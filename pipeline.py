# import packages
import time, math, os
from tqdm import tqdm
import gc
import yaml
import pickle
import random
import argparse
from datetime import datetime
from operator import itemgetter
import numpy as np
import pandas as pd
import warnings
from collections import defaultdict
import collections
warnings.filterwarnings('ignore')


# data_path = './data_raw/'
# save_path = './results/'
# data_path = '/home/admin/jupyter/data/' # 天池平台路径
# save_path = '/home/admin/jupyter/temp_results/'  # 天池平台路径


def split_train_and_test(all_df, test_rate=0.2):
    all_df['_id'] = all_df.index
    all_user_ids = all_df.user_id.unique()
    test_user_ids = np.random.choice(all_user_ids, size=int(test_rate * len(all_user_ids)), replace=False)
    #trn_df = all_df[~all_df['user_id'].isin(test_user_ids)]
    tst_df = all_df[all_df['user_id'].isin(test_user_ids)]
    #return trn_df, tst_df
    tst_df = tst_df.sort_values(['user_id', 'click_timestamp'], ascending=False).groupby('user_id').head(1)
    trn_df = all_df[~all_df['user_id'].isin(tst_df._id.unique())]
    del trn_df['_id']
    del tst_df['_id']
    return trn_df, tst_df

# debug模式：从训练集中划出一部分数据来调试代码
def get_all_click_sample(data_path, sample_rate=0.1, test_rate=0.2):
    """
        训练集中采样一部分数据调试
        data_path: 原数据的存储路径
        sample_rate: 采样率（这里由于机器的内存限制，可以采样用户做）
    """
    all_click = pd.read_csv(os.path.join(data_path, 'train_click_log.csv'))
    all_user_ids = all_click.user_id.unique()

    sample_user_ids = np.random.choice(all_user_ids, size=int(sample_rate * len(all_user_ids)), replace=False)
    all_click = all_click[all_click['user_id'].isin(sample_user_ids)]

    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    trn_click, tst_click = split_train_and_test(all_click, test_rate=test_rate)
    return trn_click, tst_click


# 读取点击数据，这里分成线上和线下，如果是为了获取线上提交结果应该讲测试集中的点击数据合并到总的数据中
# 如果是为了线下验证模型的有效性或者特征的有效性，可以只使用训练集
def get_all_click_df(data_path, offline=True, test_rate=0.2):
    if offline:
        all_click = pd.read_csv(os.path.join(data_path, 'train_click_log.csv'))
        all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
        trn_click, tst_click = split_train_and_test(all_click, test_rate=test_rate)
        return trn_click, tst_click
    else:
        trn_click = pd.read_csv(os.path.join(data_path, 'train_click_log.csv'))
        tst_click = pd.read_csv(os.path.join(data_path, 'testA_click_log.csv'))
        all_click = trn_click.append(tst_click)
        all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
        return all_click, tst_click


# 根据点击时间获取用户的点击文章序列   {user1: [(item1, time1), (item2, time2)..]...}
def get_user_item_time(click_df):
    click_df = click_df.sort_values('click_timestamp')

    def make_item_time_pair(df):
        return list(zip(df['click_article_id'], df['click_timestamp']))

    user_item_time_df = click_df.groupby('user_id')['click_article_id', 'click_timestamp'].apply(
        lambda x: make_item_time_pair(x)) \
        .reset_index().rename(columns={0: 'item_time_list'})
    user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))

    return user_item_time_dict


# 获取近期点击最多的文章
def get_item_topk_click(click_df, k):
    topk_click = click_df['click_article_id'].value_counts().index[:k]
    return topk_click


def itemcf_sim(df, params):
    """
        文章与文章之间的相似性矩阵计算
        :param df: 数据表
        :item_created_time_dict:  文章创建时间的字典
        return : 文章与文章的相似性矩阵
        思路: 基于物品的协同过滤(详细请参考上一期推荐系统基础的组队学习)， 在多路召回部分会加上关联规则的召回策略
    """

    user_item_time_dict = get_user_item_time(df)

    # 计算物品相似度
    i2i_sim = {}
    item_cnt = defaultdict(int)
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        # 在基于商品的协同过滤优化的时候可以考虑时间因素
        for i, i_click_time in item_time_list:
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})
            for j, j_click_time in item_time_list:
                if (i == j):
                    continue
                i2i_sim[i].setdefault(j, 0)

                i2i_sim[i][j] += 1 / math.log(len(item_time_list) + 1)

    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])

    # 将得到的相似性矩阵保存到本地
    pickle.dump(i2i_sim_, open(os.path.join(params['save_root'], 'itemcf_i2i_sim.pkl'), 'wb'))

    return i2i_sim_


# 基于商品的召回i2i
def item_based_recommend(user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num, item_topk_click):
    """
        基于文章协同过滤的召回
        :param user_id: 用户id
        :param user_item_time_dict: 字典, 根据点击时间获取用户的点击文章序列   {user1: [(item1, time1), (item2, time2)..]...}
        :param i2i_sim: 字典，文章相似性矩阵
        :param sim_item_topk: 整数， 选择与当前文章最相似的前k篇文章
        :param recall_item_num: 整数， 最后的召回文章数量
        :param item_topk_click: 列表，点击次数最多的文章列表，用户召回补全
        return: 召回的文章列表 {item1:score1, item2: score2...}
        注意: 基于物品的协同过滤(详细请参考上一期推荐系统基础的组队学习)， 在多路召回部分会加上关联规则的召回策略
    """

    if user_id not in user_item_time_dict:
        return []

    # 获取用户历史交互的文章
    user_hist_items = user_item_time_dict[user_id]
    user_hist_items_ = {item_id for item_id, _ in user_hist_items}

    item_rank = {}
    for loc, (i, click_time) in enumerate(user_hist_items):
        for j, wij in sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
            if j in user_hist_items_:
                continue

            item_rank.setdefault(j, 0)
            item_rank[j] += wij

    # 不足10个，用热门商品补全
    if len(item_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in item_rank.items():  # 填充的item应该不在原来的列表中
                continue
            item_rank[item] = - i - 100  # 随便给个负数就行
            if len(item_rank) == recall_item_num:
                break

    item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]

    return item_rank


def evaluate(true_df, pred_df):
    true_dict = {}
    for i, row in true_df.iterrows():
        true_dict[row['user_id']] = row['click_article_id']

    pred_dict = {}
    for i, row in pred_df.iterrows():
        pred_dict[row['user_id']] = [row['article_1'], row['article_2'], row['article_3'], row['article_4'], row['article_5']]

    scores = []
    for user_id, article_id in true_dict.items():
        if user_id in pred_dict:
            plist = pred_dict[user_id]
            for i, aid in enumerate(plist):
                if aid == article_id:
                    score = 1.0 / (i + 1)
                else:
                    score = 0
                scores.append(score)
    print("mrr: ", sum(scores), len(scores), sum(scores) / len(scores))


# 生成提交文件
# TODO: add evaluate
def submit(recall_df, params, topk=5):
    recall_df = recall_df.sort_values(by=['user_id', 'pred_score'])
    recall_df['rank'] = recall_df.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')

    ## 判断是不是每个用户都有5篇文章及以上
    #tmp = recall_df.groupby('user_id').apply(lambda x: x['rank'].max())
    #assert tmp.min() >= topk

    del recall_df['pred_score']
    submit = recall_df[recall_df['rank'] <= topk].set_index(['user_id', 'rank']).unstack(-1).reset_index()

    submit.columns = [int(col) if isinstance(col, int) else col for col in submit.columns.droplevel(0)]
    # 按照提交格式定义列名
    submit = submit.rename(columns={'': 'user_id', 1: 'article_1', 2: 'article_2',
                                    3: 'article_3', 4: 'article_4', 5: 'article_5'})

    save_name = os.path.join(params['save_root'], params['model_name'] + '_' + datetime.today().strftime('%m-%d') + '.csv')
    submit.to_csv(save_name, index=False, header=True)
    return submit


def load_config(config_file):
    params = dict()
    if not os.path.exists(config_file):
        raise RuntimeError('config_file={} not exist!'.format(config_file))

    with open(config_file, 'r') as cfg:
        config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
        if 'data' in config_dict:
            params.update(config_dict['data'])
        if 'model' in config_dict:
            params.update(config_dict['model'])

    return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/offline.yaml', help='The config file.')

    args = vars(parser.parse_args())
    params = load_config(args['config'])

    data_root = params['data_root']
    save_root = params['save_root']
    sim_item_topk = params['sim_item_topk']
    recall_item_num = params['recall_item_num']

    # 初始化
    if not os.path.exists(params['save_root']):
        os.makedirs(params['save_root'])

    # 正常训练集和测试集的用户有重复吗，有必要进行隔离划分吗？
    if params['debug']:
        # debug模式
        train_click_df, test_click_df = get_all_click_sample(data_root, sample_rate=params['sample_rate'])
    else:
        # 全量训练集
        train_click_df, test_click_df = get_all_click_df(data_root, params['offline'])

    i2i_sim = itemcf_sim(train_click_df, params)
    # i2i_sim_path = save_root + 'itemcf_i2i_sim.pkl'
    # if os.path.exists(i2i_sim_path):
    #     # 去取文章相似度
    #     i2i_sim = pickle.load(open(save_root + 'itemcf_i2i_sim.pkl', 'rb'))
    # else:
    #     i2i_sim = itemcf_sim(all_click_df, params)

    # 定义
    user_recall_items_dict = collections.defaultdict(dict)

    # 获取 用户 - 文章 - 点击时间的字典
    user_item_time_dict = get_user_item_time(train_click_df)

    # 用户热度补全
    item_topk_click = get_item_topk_click(train_click_df, k=50)

    for user in tqdm(test_click_df['user_id'].unique()):
        user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, i2i_sim,
                                                            sim_item_topk, recall_item_num, item_topk_click)

    # 将字典的形式转换成df
    user_item_score_list = []

    for user, items in tqdm(user_recall_items_dict.items()):
        for item, score in items:
            user_item_score_list.append([user, item, score])

    recall_df = pd.DataFrame(user_item_score_list, columns=['user_id', 'click_article_id', 'pred_score'])

    # 生成测试数据
    tst_users = test_click_df['user_id'].unique()
    tst_recall = recall_df[recall_df['user_id'].isin(tst_users)]
    pred_df = submit(tst_recall, params, topk=5)
    evaluate(test_click_df, pred_df)

    # # 获取测试集
    # tst_click = pd.read_csv(data_path + 'testA_click_log.csv')
    # tst_users = tst_click['user_id'].unique()
    #
    # # 从所有的召回数据中将测试集中的用户选出来
    # tst_recall = recall_df[recall_df['user_id'].isin(tst_users)]
    #
    # # 生成提交文件
    # submit(tst_recall, topk=5, model_name='itemcf_baseline')

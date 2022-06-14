import pandas as pd

trn_click = pd.read_csv('./data_raw/train_click_log.csv')
tst_click = pd.read_csv('./data_raw/testA_click_log.csv')

for i, row in tst_click.iterrows():
    print(row['user_id'])
    input()

common_users = set(trn_click['user_id']) & set(tst_click['user_id'])
print(len(common_users))


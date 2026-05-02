import pandas as pd

df_owid = pd.read_csv('data/processed/cleaned_owid.csv')
df_oxford = pd.read_csv('data/processed/cleaned_oxford.csv')
df_google = pd.read_csv('data/processed/cleaned_google.csv')
df_tweets = pd.read_csv('data/processed/cleaned_tweets.csv')

dfs = [df_owid, df_oxford, df_google, df_tweets]
for df in dfs:
    df['date'] = pd.to_datetime(df['date'])

# Inner 3 nguồn cơ bản vì chúng đầy đủ
df_master = pd.merge(df_owid, df_oxford, on=['code', 'date'], how='inner')
df_master = pd.merge(df_master, df_google, on=['code', 'date'], how='inner')

# Left với Twitter để không bị mất các ngày đầu dịch
df_master = pd.merge(df_master, df_tweets, on=['code', 'date'], how='left') 

# XỬ LÝ AN TOÀN NAN CHO RANDOM FOREST:
# Xác định các cột thuộc về Twitter
tweet_cols = df_tweets.columns.drop(['code', 'date'])
# Nếu ngày đó không có tweet nào (NaN do left join), điền 0
df_master[tweet_cols] = df_master[tweet_cols].fillna(0)

df_master = df_master.dropna()

# PHẠM VI
target_countries = ['IND', 'CAN', 'GBR', 'AUS']
df_model = df_master[df_master['code'].isin(target_countries)].copy()

print(f"Rows: {len(df_model)}")

output_path = 'data/processed/ML_READY_DATASET.csv'
df_model.to_csv(output_path, index=False)

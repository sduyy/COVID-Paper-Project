import pandas as pd

df_owid = pd.read_csv('data/processed/cleaned_owid.csv')
df_oxford = pd.read_csv('data/processed/cleaned_oxford.csv')
df_google = pd.read_csv('data/processed/cleaned_google.csv')
df_tweets = pd.read_csv('data/processed/cleaned_tweets.csv')

dfs = [df_owid, df_oxford, df_google, df_tweets]
for df in dfs:
    df['date'] = pd.to_datetime(df['date'])

df_master = pd.merge(df_owid, df_oxford, on=['code', 'date'], how='inner')
df_master = pd.merge(df_master, df_google, on=['code', 'date'], how='inner')
df_master = pd.merge(df_master, df_tweets, on=['code', 'date'], how='left') 

# Điền 0 cho những ngày không có dữ liệu Twitter (không ai tweet = tâm lý trung lập/volume = 0)
df_master['sentiment_lag_14'] = df_master['sentiment_lag_14'].fillna(0)
df_master['volume_lag_14'] = df_master['volume_lag_14'].fillna(0)

# PHẠM VI
target_countries = ['IND', 'CAN', 'GBR', 'AUS']
df_model = df_master[df_master['code'].isin(target_countries)].copy()

output_path = 'data/processed/ML_READY_DATASET.csv'
df_model.to_csv(output_path, index=False)

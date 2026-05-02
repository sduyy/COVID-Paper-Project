import pandas as pd

df_ox = pd.read_csv('data/raw/OxCGRT_compact_national_v1.csv')

df_ox = df_ox[df_ox['Jurisdiction'] == 'NAT_TOTAL'].copy()

policy_features = [
    # C: Hạn chế/Đóng cửa
    'C1M_School closing', 'C2M_Workplace closing', 'C3M_Cancel public events',
    'C4M_Restrictions on gatherings', 'C5M_Close public transport',
    'C6M_Stay at home requirements', 'C7M_Restrictions on internal movement',
    'C8EV_International travel controls',
    
    # H: Y tế công cộng
    'H1_Public information campaigns', 'H2_Testing policy', 'H3_Contact tracing',
    'H6M_Facial Coverings', 'H8M_Protection of elderly people',
    
    # E: Hỗ trợ kinh tế
    'E1_Income support', 'E2_Debt/contract relief'
]

cols_to_keep = ['CountryCode', 'Date'] + policy_features
df_ox_selected = df_ox[cols_to_keep].copy()

df_ox_selected['Date'] = pd.to_datetime(df_ox_selected['Date'].astype(str), format='%Y%m%d')

df_ox_selected = df_ox_selected.sort_values(by=['CountryCode', 'Date'])

# Forward fill: Ngày hôm nay không ghi nhận gì thì lấy chính sách của ngày hôm qua
df_ox_selected[policy_features] = df_ox_selected.groupby('CountryCode')[policy_features].ffill()

# Fill 0 cho giai đoạn đầu dịch chưa ban hành lệnh nào
df_ox_selected[policy_features] = df_ox_selected[policy_features].fillna(0)

df_ox_selected = df_ox_selected.set_index('Date')
df_ox_resampled = df_ox_selected.groupby('CountryCode')[policy_features].resample('D').asfreq().reset_index()

# Vì chính sách duy trì theo thời gian dùng ffill cho các ngày thiếu
df_ox_resampled[policy_features] = df_ox_resampled.groupby('CountryCode')[policy_features].ffill().fillna(0)

for col in policy_features:
    df_ox_resampled[f'{col}_lag_21'] = df_ox_resampled.groupby('CountryCode')[col].shift(21)

df_ox_final = df_ox_resampled.dropna().copy()
df_ox_final = df_ox_final.rename(columns={'CountryCode': 'code', 'Date': 'date'})

df_ox_final.to_csv('data/processed/cleaned_oxford.csv', index=False)

import pandas as pd

df = pd.read_csv('data/raw/owid_covid_data.csv')

df_countries = df[df['continent'].notna()].copy()

features_to_keep = [
    'code', 'country', 'date',
    
    # Target & Control
    'new_deaths_smoothed_per_million', 
    'new_cases_smoothed_per_million',
    
    'stringency_index',
    
    'people_fully_vaccinated_per_hundred',
    
    # Demographics
    'population_density', 'median_age', 'gdp_per_capita', 'hospital_beds_per_thousand'
]

df_selected = df_countries[features_to_keep].copy()

df_selected['date'] = pd.to_datetime(df_selected['date'])
df_selected = df_selected.sort_values(by=['code', 'date'])

# Giai đoạn đầu chưa có vaccine fill bằng 0, sau đó forward fill
df_selected['people_fully_vaccinated_per_hundred'] = df_selected.groupby('code')['people_fully_vaccinated_per_hundred'].ffill()
df_selected['people_fully_vaccinated_per_hundred'] = df_selected['people_fully_vaccinated_per_hundred'].fillna(0)

# Fill 0 cho ca nhiễm, ca tử vong và stringency index ở giai đoạn đầu dịch
cols_to_zero = ['new_deaths_smoothed_per_million', 'new_cases_smoothed_per_million', 'stringency_index']
df_selected[cols_to_zero] = df_selected[cols_to_zero].fillna(0)

df_selected = df_selected.set_index('date')

cols_to_resample = [col for col in features_to_keep if col not in ['code', 'date']]

df_resampled = df_selected.groupby('code')[cols_to_resample].resample('D').asfreq().reset_index()

# Những ngày mới được sinh ra sẽ bị trống (NaN). 
# Dùng ffill để kéo dài các thông tin tĩnh (tên nước, dân số, gdp, vaccine...) từ ngày hôm trước
df_resampled[cols_to_resample] = df_resampled.groupby('code')[cols_to_resample].ffill()

# Nếu vẫn còn ngày trống ở tận cùng phía trước điền 0
df_resampled = df_resampled.fillna(0)

df_resampled['cases_lag_14'] = df_resampled.groupby('code')['new_cases_smoothed_per_million'].shift(14)
df_resampled['deaths_lag_14'] = df_resampled.groupby('code')['new_deaths_smoothed_per_million'].shift(14)
df_resampled['stringency_lag_21'] = df_resampled.groupby('code')['stringency_index'].shift(21)

df_final = df_resampled.dropna().copy()

df_final.to_csv('data/processed/cleaned_owid.csv', index=False)

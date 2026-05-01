import pandas as pd
import pycountry
import numpy as np

df_mob = pd.read_csv('data/raw/google_mobility.csv', low_memory=False)

df_mob_national = df_mob[
    df_mob['sub_region_1'].isna() & 
    df_mob['sub_region_2'].isna() & 
    df_mob['metro_area'].isna()
].copy()

mobility_features = [
    'retail_and_recreation_percent_change_from_baseline',
    'grocery_and_pharmacy_percent_change_from_baseline',
    'parks_percent_change_from_baseline',
    'transit_stations_percent_change_from_baseline',
    'workplaces_percent_change_from_baseline',
    'residential_percent_change_from_baseline'
]

cols_to_keep = ['country_region_code', 'date'] + mobility_features
df_mob_selected = df_mob_national[cols_to_keep].copy()

df_mob_selected = df_mob_selected.rename(columns={'country_region_code': 'code'})

def convert_alpha2_to_alpha3(alpha2):
    try:
        # Ép kiểu string để tránh lỗi với các giá trị float/NaN
        return pycountry.countries.get(alpha_2=str(alpha2).upper()).alpha_3
    except:
        return np.nan

df_mob_selected['code'] = df_mob_selected['code'].apply(convert_alpha2_to_alpha3)
df_mob_selected = df_mob_selected.dropna(subset=['code'])

df_mob_selected['date'] = pd.to_datetime(df_mob_selected['date'])
df_mob_selected = df_mob_selected.sort_values(by=['code', 'date'])

for col in mobility_features:
    df_mob_selected[col] = df_mob_selected.groupby('code')[col].transform(lambda x: x.interpolate(method='linear'))
df_mob_selected[mobility_features] = df_mob_selected[mobility_features].fillna(0)

df_mob_selected = df_mob_selected.set_index('date')
df_mob_selected = df_mob_selected.groupby('code')[mobility_features].resample('D').asfreq()
df_mob_selected = df_mob_selected.reset_index()

for col in mobility_features:
    # Điền nội suy lại cho những ngày trống vừa được sinh ra (nếu có)
    df_mob_selected[col] = df_mob_selected.groupby('code')[col].transform(lambda x: x.interpolate(method='linear')).fillna(0)
    # Tiến hành shift an toàn
    df_mob_selected[f'{col}_lag_21'] = df_mob_selected.groupby('code')[col].shift(21)

df_mob_final = df_mob_selected.dropna().copy()

df_mob_final.to_csv('data/processed/cleaned_google.csv', index=False)

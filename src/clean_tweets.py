import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

def process_tweets():
    df_tweets = pd.read_csv('data/raw/vaccination_all_tweets.csv', low_memory=False)

    df_tweets = df_tweets[['date', 'user_location', 'text']].copy()
    df_tweets['date'] = pd.to_datetime(df_tweets['date'], format='mixed').dt.floor('D')
    df_tweets = df_tweets.dropna(subset=['text', 'user_location'])

    # Dictionary
    location_map = {
        'india': 'IND', 'delhi': 'IND', 'mumbai': 'IND', 'bengaluru': 'IND',
        
        'usa': 'USA', 'united states': 'USA', 'new york': 'USA', 'california': 'USA',
        'texas': 'USA', 'florida': 'USA', 'washington': 'USA', 'chicago': 'USA',
        'los angeles': 'USA', 'boston': 'USA', 'atlanta': 'USA', 'miami': 'USA',
        'seattle': 'USA', 'dallas': 'USA', 'houston': 'USA',
        
        'uk': 'GBR', 'united kingdom': 'GBR', 'london': 'GBR', 'england': 'GBR',
        
        'canada': 'CAN', 'toronto': 'CAN', 'ontario': 'CAN', 'vancouver': 'CAN',
        
        'australia': 'AUS', 'sydney': 'AUS', 'melbourne': 'AUS'
    }

    def map_location_to_code(loc):
        loc_str = str(loc).lower()
        for key, code in location_map.items():
            if key in loc_str:
                return code
        return np.nan

    df_tweets['code'] = df_tweets['user_location'].apply(map_location_to_code)
    df_tweets = df_tweets.dropna(subset=['code'])

    analyzer = SentimentIntensityAnalyzer()
    
    def get_sentiment(text):
        return analyzer.polarity_scores(str(text))['compound']

    df_tweets['sentiment_score'] = df_tweets['text'].apply(get_sentiment)

    # 1. Tính toán Thống kê Cơ bản theo ngày
    df_tweets_daily = df_tweets.groupby(['code', 'date']).agg(
        daily_sentiment_avg=('sentiment_score', 'mean'),
        daily_sentiment_std=('sentiment_score', 'std'), 
        daily_tweet_volume=('text', 'count')
    ).reset_index()

    # Nếu chỉ có 1 tweet, độ lệch chuẩn (std) là NaN -> Gán = 0 (không phân cực)
    df_tweets_daily['daily_sentiment_std'] = df_tweets_daily['daily_sentiment_std'].fillna(0)

    # 2. Resample để liên tục hóa trục thời gian
    df_tweets_daily = df_tweets_daily.set_index('date')
    df_filled = df_tweets_daily.groupby('code')[['daily_sentiment_avg', 'daily_sentiment_std', 'daily_tweet_volume']].resample('D').asfreq().reset_index()

    # 3. FIX LỖI 1: Xử lý giá trị trống (Missing Values) một cách trung thực
    # Volume: Ngày không ai tweet = 0 là chính xác.
    df_filled['daily_tweet_volume'] = df_filled['daily_tweet_volume'].fillna(0)
    
    # Sentiment: Forward-fill để bảo lưu cảm xúc âm ỉ. TUYỆT ĐỐI KHÔNG fillna(0) sau đó.
    # Những ngày đầu dịch (chưa ai tweet bao giờ) sẽ giữ nguyên là NaN và bị drop ở bước cuối.
    cols_to_fill = ['daily_sentiment_avg', 'daily_sentiment_std']
    df_filled[cols_to_fill] = df_filled.groupby('code')[cols_to_fill].ffill()

    # 4. FIX LỖI 2: Chuẩn hóa Z-score (Khử khác biệt quy mô quốc gia)
    df_filled['sentiment_zscore'] = df_filled.groupby('code')['daily_sentiment_avg'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6)
    )

    # 5. FIX LỖI 3: Trọng số hóa (Volume-weighted)
    df_filled['weighted_sentiment'] = df_filled['daily_sentiment_avg'] * np.log1p(df_filled['daily_tweet_volume'])

    # 6. FIX LỖI: Bắt Gia tốc Tâm lý (Momentum)
    # Giữ nguyên ở thời điểm hiện tại, KHÔNG TẠO LAG cho diff để tránh dư thừa thông tin
    df_filled['sentiment_delta_3'] = df_filled.groupby('code')['daily_sentiment_avg'].diff(3)
    df_filled['sentiment_delta_7'] = df_filled.groupby('code')['daily_sentiment_avg'].diff(7)

    # VŨ KHÍ MỚI: Rolling Average (Xu hướng tâm lý duy trì trong 7 ngày)
    df_filled['sentiment_roll7'] = df_filled.groupby('code')['daily_sentiment_avg'].transform(
        lambda x: x.rolling(7).mean()
    )

    # 7. Tạo Biến Trễ (Lags) Đa Khung Thời Gian (Đã lọc bỏ các biến gây nhiễu)
    for lag in [3, 7, 14, 21]:
        # Giữ lại các biến cốt lõi
        df_filled[f'sentiment_avg_lag_{lag}'] = df_filled.groupby('code')['daily_sentiment_avg'].shift(lag)
        df_filled[f'sentiment_std_lag_{lag}'] = df_filled.groupby('code')['daily_sentiment_std'].shift(lag)
        df_filled[f'volume_lag_{lag}'] = df_filled.groupby('code')['daily_tweet_volume'].shift(lag)
        df_filled[f'weighted_sentiment_lag_{lag}'] = df_filled.groupby('code')['weighted_sentiment'].shift(lag)
        df_filled[f'sentiment_zscore_lag_{lag}'] = df_filled.groupby('code')['sentiment_zscore'].shift(lag)
        
        # Lag cho xu hướng tâm lý (Rất mạnh)
        df_filled[f'sentiment_roll7_lag_{lag}'] = df_filled.groupby('code')['sentiment_roll7'].shift(lag)

    # 8. Chốt dữ liệu
    df_tweets_final = df_filled.dropna().copy()
    
    # BƯỚC KIỂM TRA (Sanity Check) - In ra terminal để xem phân phối
    print("\nTHỐNG KÊ PHÂN PHỐI DỮ LIỆU TWITTER:")
    print(df_tweets_final[['daily_sentiment_avg', 'daily_sentiment_std', 'daily_tweet_volume']].describe())

    df_tweets_final.to_csv('data/processed/cleaned_tweets.csv', index=False)

if __name__ == "__main__":
    process_tweets()

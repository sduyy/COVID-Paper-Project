import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import os

# nltk.download('vader_lexicon')

def process_tweets():
    df_tweets = pd.read_csv('data/raw/vaccination_all_tweets.csv', low_memory=False)

    df_tweets = df_tweets[['date', 'user_location', 'text']].copy()
    df_tweets['date'] = pd.to_datetime(df_tweets['date'], format='mixed').dt.floor('D')
    df_tweets = df_tweets.dropna(subset=['text', 'user_location'])

    # Cập nhật thêm dictionary này tùy theo phạm vi nghiên cứu
    location_map = {
        'india': 'IND', 'delhi': 'IND', 'mumbai': 'IND', 'bengaluru': 'IND',
        
        # Thêm hàng loạt các bang và thành phố lớn của Mỹ
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
            if key in loc_str: # Bắt keyword trực tiếp từ chuỗi chữ thường
                return code
        return np.nan

    df_tweets['code'] = df_tweets['user_location'].apply(map_location_to_code)
    df_tweets = df_tweets.dropna(subset=['code'])

    analyzer = SentimentIntensityAnalyzer()
    
    def get_sentiment(text):
        return analyzer.polarity_scores(str(text))['compound']

    df_tweets['sentiment_score'] = df_tweets['text'].apply(get_sentiment)

    df_tweets_daily = df_tweets.groupby(['code', 'date']).agg(
        daily_sentiment_avg=('sentiment_score', 'mean'),
        daily_tweet_volume=('text', 'count')
    ).reset_index()

    df_tweets_daily = df_tweets_daily.set_index('date')

    df_filled = df_tweets_daily.groupby('code')[['daily_sentiment_avg', 'daily_tweet_volume']].resample('D').asfreq()

    df_filled = df_filled.reset_index()

    df_filled['daily_sentiment_avg'] = df_filled['daily_sentiment_avg'].fillna(0)
    df_filled['daily_tweet_volume'] = df_filled['daily_tweet_volume'].fillna(0)

    df_filled['sentiment_lag_14'] = df_filled.groupby('code')['daily_sentiment_avg'].shift(14)
    df_filled['volume_lag_14'] = df_filled.groupby('code')['daily_tweet_volume'].shift(14)

    df_tweets_final = df_filled.dropna().copy()

    df_tweets_final.to_csv('data/processed/cleaned_tweets.csv', index=False)

if __name__ == "__main__":
    process_tweets()

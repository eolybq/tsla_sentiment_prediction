import pandas as pd
from pandas.tseries.offsets import MonthEnd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

indicator_data = pd.read_csv('../../cleandata/individual_data/indicator_data.csv')
tesla_trends = pd.read_csv('../../cleandata/individual_data/tesla_trends.csv')
daily_tesla_trends = pd.read_csv('../../cleandata/individual_data/daily_tesla_trends_data.csv')
sent_surv_clean = pd.read_csv('../../cleandata/individual_data/sent_surv_clean.csv')
sentiment_daily = pd.read_csv('../../cleandata/individual_data/sentiment_daily.csv')
sentiment_not_agr = pd.read_csv('../../cleandata/individual_data/sentiment_not_agr.csv')
vix = pd.read_csv('../../cleandata/individual_data/vix_data.csv')


# Ocisteni pouze sledovanych sloupcu
indicator_data = indicator_data.drop(['symbol', 'open', 'high', 'low', 'close'], axis = 1)
sent_surv_clean = sent_surv_clean.loc[:, ['date', 'bullish_surv', 'neutral_surv', 'bearish_surv', 'bull_bear_spread_surv']]
vix = vix.loc[:, ['date', 'adjusted']]
vix = vix.rename(columns={'adjusted': 'vix'})
tweets_sentiment = sentiment_daily.loc[:, ['sentiment', 'date']]


# Prevod trends na DAILY frekvenci
# 1. Vytvoř date jako první den měsíce (předpoklad: 'date' je string ve formátu "YYYY-MM")
tesla_trends['date'] = pd.to_datetime(tesla_trends['date'] + '-01')

# 2. Vytvoř kompletní rozsah dat (denní frekvence)
date_range_trends = pd.date_range(
    start=tesla_trends['date'].min(),
    end=tesla_trends['date'].max() + MonthEnd(1) - pd.Timedelta(days=1),
    freq='D'
)

# zajisti, že pro každý den bude řádek – merge s plným kalendářem
full_trends_df = pd.DataFrame({'date': date_range_trends})
tesla_trends_daily = pd.merge(full_trends_df, tesla_trends, on='date', how='left')

# vyplň chybějící g_trends směrem dolů (downward fill)
tesla_trends_daily['g_trends'] = tesla_trends_daily['g_trends'].ffill()


sent_surv_clean['date'] = pd.to_datetime(sent_surv_clean['date'])

date_range_surv = pd.date_range(
    start=sent_surv_clean['date'].min(),
    end=sent_surv_clean['date'].max() + MonthEnd(1) - pd.Timedelta(days=1),
    freq='D'
    )

full_surv_df = pd.DataFrame({'date': date_range_surv})
surv_daily = pd.merge(full_surv_df, sent_surv_clean, on='date', how='left')

surv_daily = surv_daily.ffill()


# Merge všech dat do jednoho DataFrame

# všude je date typu datetime
indicator_data['date'] = pd.to_datetime(indicator_data['date'])
vix['date'] = pd.to_datetime(vix['date'])
tweets_sentiment['date'] = pd.to_datetime(tweets_sentiment['date'])
tesla_trends_daily['date'] = pd.to_datetime(tesla_trends_daily['date'])
surv_daily['date'] = pd.to_datetime(surv_daily['date'])

data = pd.merge(vix, tweets_sentiment, on='date', how='left')
data = pd.merge(data, tesla_trends_daily, on='date', how='left')
data = pd.merge(data, surv_daily, on='date', how='left')
data = pd.merge(data, indicator_data, on='date', how='left')

data['sentiment'] = data['sentiment'].fillna('none')
data = pd.concat([data, pd.get_dummies(data['sentiment'], prefix='sentiment').astype(float)], axis=1)
data = data.drop(columns=['sentiment'])


# Omezení vzorku dle sentiment_daily
data = data[
    (data['date'] >= sentiment_daily['date'].min()) &
    (data['date'] <= sentiment_daily['date'].max())
]

data = data.dropna()

data.to_csv('../cleandata/processed_data.csv', index=False)

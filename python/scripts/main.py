import pandas as pd

indicator_data = pd.read_csv('cleandata/indicator_data.csv')
tesla_trends = pd.read_csv('cleandata/tesla_trends.csv')
daily_tesla_trends = pd.read_csv('cleandata/daily_tesla_trends_data.csv')
sent_surv_clean = pd.read_csv('cleandata/sent_surv_clean.csv')
sentiment_daily = pd.read_csv('cleandata/sentiment_daily.csv')
sentiment_not_agr = pd.read_csv('cleandata/sentiment_not_agr.csv')
vix = pd.read_csv('cleandata/vix_data.csv')

# TODO: prevest tesla trends na MONTHLY - asi last obs. nebo ten trendecon?

# TODO: WEEKLY surv_sentiment převedeno na DAILY způsobem last obs.

# TODO: vymyslet co s NA v tweets sentiment a taky jak z toho udelat oprp faktor nebo tak neco??? ty NA co vziknou pri merge s ostatnimi vlastne nejak asi predelat na tu uroven



# NOTE: Omezení vzorku dle sentiment_daily
# TODO: POKUD TWEETS SENTIMENT DATA K NIČEMU TAK ZRUŠIT



# Odstranění sloupců, které nejsou potřeba
# tibble_data <- tibble_data_all_adj |>
#     select(-symbol, -adjusted, -open, -high, -low) |>
#     drop_na()

import pandas as pd
from pandas.tseries.offsets import MonthEnd

indicator_data = pd.read_csv('cleandata/indicator_data.csv')
tesla_trends = pd.read_csv('cleandata/tesla_trends.csv')
daily_tesla_trends = pd.read_csv('cleandata/daily_tesla_trends_data.csv')
sent_surv_clean = pd.read_csv('cleandata/sent_surv_clean.csv')
sentiment_daily = pd.read_csv('cleandata/sentiment_daily.csv')
sentiment_not_agr = pd.read_csv('cleandata/sentiment_not_agr.csv')
vix = pd.read_csv('cleandata/vix_data.csv')

# TODO: prevest tesla trends na MONTHLY - asi last obs. nebo ten trendecon?

# 1. Vytvoř date jako první den měsíce (předpoklad: 'date' je string ve formátu "YYYY-MM")
tesla_trends['date'] = pd.to_datetime(tesla_trends['date'] + '-01')

# 2. Vytvoř kompletní rozsah dat (denní frekvence)
date_range = pd.date_range(
    start=tesla_trends['date'].min(),
    end=tesla_trends['date'].max() + MonthEnd(1) - pd.Timedelta(days=1),
    freq='D'
)

# 3. Zajisti, že pro každý den bude řádek – merge s plným kalendářem
full_df = pd.DataFrame({'date': date_range})
tesla_trends_daily = pd.merge(full_df, tesla_trends, on='date', how='left')

# 4. Vyplň chybějící g_trends směrem dolů (downward fill)
tesla_trends_daily['g_trends'] = tesla_trends_daily['g_trends'].ffill()




# TODO: WEEKLY surv_sentiment převedeno na DAILY způsobem last obs.

# TODO: vymyslet co s NA v tweets sentiment a taky jak z toho udelat oprp faktor nebo tak neco??? ty NA co vziknou pri merge s ostatnimi vlastne nejak asi predelat na tu uroven



# NOTE: Omezení vzorku dle sentiment_daily
# TODO: POKUD TWEETS SENTIMENT DATA K NIČEMU TAK ZRUŠIT



# Odstranění sloupců, které nejsou potřeba
# tibble_data <- tibble_data_all_adj |>
#     select(-symbol, -adjusted, -open, -high, -low) |>
#     drop_na()


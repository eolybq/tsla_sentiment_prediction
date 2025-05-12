library(tidyverse)
library(tidyquant)
library(TTR)

rm(list = ls())


# STOCK DATA + INDICATORS ===============
# Stažení historických dat pro TSLA
# rawdata <- tq_get("TSLA", from = "2010-06-01", to = "2025-05-01")

# save(rawdata, file = "data/rawdata/tsla_quant.RData")

load("data/rawdata/tsla_quant.RData")


# ============================
# Trend – SMA, EMA, MACD
#
# Momentum – RSI, StochRSI
#
# Volatilita / Riziko – ATR, BBands, Basic Volatility
#
# Objem – OBV
#
# Cenová hladina – samotné close, return apod.
# ============================


indicator_data <- rawdata

# Průměr zavíracích cen za posledních X dní Sleduje trend. Pokud je cena nad SMA, je to býčí signál, pod SMA je medvědí signál.
indicator_data$sma_20 <- zoo::rollapply(rawdata$close, 20, mean, fill = NA)
indicator_data$sma_50 <- zoo::rollapply(rawdata$close, 50, mean, fill = NA)

# Vážený průměr, který dává větší důraz na novější data. Rychleji reaguje na změny trendu než SMA.
indicator_data$ema_20 <- EMA(rawdata$close, n = 20)

# Rychlý odhad volatility bez složitého modelování.
indicator_data$basic_volatility <- rawdata$high - rawdata$low

# Průměrný rozsah cenového pohybu včetně gapů, přesnější měření volatility než basic_volatility. Měří tržní riziko; vyšší ATR = vyšší riziko.
indicator_data$atr <- ATR(cbind(rawdata$high, rawdata$low, rawdata$close), n = 14)$atr

# Indikátor momenta, který měří poměr kladných vs. záporných výnosů za určité období
indicator_data$rsi <- RSI(rawdata$close, n = 14)

# Rozdíl mezi dvěma EMA (např. 12 a 26 dní) + signální čára (9denní EMA). Sleduje sílu a směr trendu. MACD překročí signal line směrem nahoru = nákupní signál. Směrem dolů = prodejní signál.
macd <- MACD(rawdata$close, nFast = 12, nSlow = 26, nSig = 9)
indicator_data$macd <- macd$macd
indicator_data$macd_signal <- macd$signal

# Horní a dolní pásmo kolem klouzavého průměru. Určuje volatilitu. Vypočítává se jako SMA ± 2× směrodatná odchylka. Cena dotýkající se horního pásma = potenciální překoupení. Dotýkající se spodního = přeprodanost.
bb <- BBands(indicator_data$close, n = 20)
indicator_data$bb_up <- bb$up
indicator_data$bb_dn <- bb$dn

# Kumulativní součet objemu obchodování s ohledem na směr ceny. Pokud roste OBV spolu s cenou, potvrzuje trend. Divergence (např. cena roste, OBV klesá) varuje před obratem.
indicator_data$obv <- OBV(rawdata$close, rawdata$volume)

# Aplikuje stochastický oscilátor na RSI, takže výsledná hodnota je mezi 0 a 1.
indicator_data$stochrsi <- stochRSI(rawdata$close, n = 14, maType = "SMA")

# Sila trendu bez smeru
adx <- ADX(cbind(rawdata$high, rawdata$low, rawdata$close), n = 14)
indicator_data$adx <- adx[, 4]


save(indicator_data, file = "data/cleandata/tsla_quant_indicators.RData")


# TRENDS + TWEETS DATA ==================

# TODO: DAILY normalizovane hodnoty z baliku trendecon z Google trends api
library(trendecon)
ts_gtrends_mwd("tesla", geo = "US", from = "2010-01-01")


tesla_trends <- read_csv(
    "data/rawdata/tesla_trends.csv",
    skip = 3,
    col_names = c("date", "value")
)

save(tesla_trends, file = "data/cleandata/tesla_trends.RData")


# NOTE: Korelace s tesla_trends = 0.9042762 takže asi whatever co použiju
#tsla_trends <- read_csv("data/rawdata/tsla_trends.csv", skip = 1)
#tsla_trends[[2]] <- as.numeric(str_replace_all(tsla_trends[[2]], "\\D", ""))



elon_tweets <- read_csv("data/rawdata/elon_2010_2025/all_musk_posts.csv")
# RETWEETS DATA
# elon_retweets <- read_csv("data/rawdata/elon_2010_2025/musk_quote_tweets.csv")


# Tweets cleanup od hastags mention emojis answers atd.
clean_tweet <- function(text) {
    text |>
        str_remove_all("http[s]?://\\S+") |>         # URL
        str_remove_all("#\\S+") |>                   # Hashtagy
        str_remove_all("@\\w+") |>                   # Zmínky
        str_remove_all("[\U0001F600-\U0001F64F]") |> # Emoji (základní range)
        str_remove_all("[\U0001F300-\U0001F5FF]") |>
        str_remove_all("[\U0001F680-\U0001F6FF]") |>
        str_remove_all("[\U0001F1E0-\U0001F1FF]") |>
        str_squish()                                # Nadbytečné mezery
}

tweets_tsla <- elon_tweets |>
    mutate(
        date = as.Date(createdAt),
        cleanText = clean_tweet(fullText)
    ) |>

    # TODO: zajistit i engagement metriky - pocet like, retweet, reply
    select(date, cleanText) |>
    filter(str_detect(
        cleanText,
        regex(
            paste(
                "tesla",
                "tsla",
                "elon musk",
                "model [s3xy]",
                "cybertruck",
                "roadster",
                "semi",
                "giga\\w*",
                "autopilot",
                "fsd",
                "self[- ]?driving",
                "autonomous",
                "ev\\b",
                "electric car",
                "battery",
                "batteries",
                "supercharger",
                "charging",
                "solar roof",
                "solar panels?",
                "powerwall",
                "energy storage",
                "tesla bot",
                "humanoid robot",
                "tesla stock",
                "stock price",
                "earnings",
                "deliveries",
                "production",
                "supply chain",
                "recall",
                "safety",
                "tesla insurance",
                sep = "|"
            ),
            ignore_case = TRUE
        )
    )) |>
    arrange(date)

# agregace vsech tweetu na denní frekvenci
tweets_tsla_daily <- tweets_tsla |>
    group_by(date) |>
    summarise(
        cleanText = paste(cleanText, collapse = " "),
        .groups = "drop"
    )

write_csv(tweets_tsla_daily, "data/rawdata/tweets_tsla_daily.csv")

# TODO: Znovu nacist po finBERT upravit do struktury R a ulozit save(RData) do /cleandata
rm(list = ls())

library(tidyverse)
library(tseries)
library(corrr)

load("data/cleandata/tibble_data.RData")


# NOTE: EDA

# Descriptive statistics
summary(tibble_data)
colSums(is.na(tibble_data))
table(tibble_data$tweets_sentiment)
tibble_data |>
    select_if(is.numeric) |>
    correlate() |>
    shave() |>
    rplot()


# Visualizations
ggplot(tibble_data, aes(x = date)) +
    geom_line(aes(y = close), color = "blue") +
    labs(title = "Vývoj závěrečné ceny", y = "Cena", x = "Datum")

ggplot(tibble_data, aes(x = date)) +
    geom_line(aes(y = volume), color = "darkgreen") +
    labs(title = "Objem obchodů", y = "Volume")

ggplot(tibble_data, aes(x = date)) +
    geom_line(aes(y = basic_volatility), color = "red") +
    labs(title = "Základní volatilita", y = "Volatilita")

ggplot(tibble_data, aes(x = date)) +
    geom_line(aes(y = close), color = "black") +
    geom_line(aes(y = bb_up), color = "blue", linetype = "dashed") +
    geom_line(aes(y = bb_dn), color = "red", linetype = "dashed") +
    labs(title = "Bollinger Bands")

ggplot(tibble_data, aes(x = date)) +
    geom_line(aes(y = close), color = "black") +
    geom_area(aes(y = bullish_surv * max(close), fill = "Bullish"), alpha = 0.3) +
    geom_area(aes(y = -bearish_surv * max(close), fill = "Bearish"), alpha = 0.3) +
    scale_fill_manual(values = c("Bullish" = "green", "Bearish" = "red")) +
    labs(title = "Cena a sentiment (plošně)", x = "Datum", y = "Cena / Sentiment") +
    theme_minimal()

# TODO: lepší tweets sentiment graf
ggplot(tibble_data, aes(x = date)) +
    geom_line(aes(y = close), color = "black") +
    geom_point(aes(y = close, color = tweets_sentiment), size = 1.8, alpha = 0.7) +
    scale_color_manual(
        values = c(
            "positive" = "green",
            "negative" = "red",
            "neutral"  = "blue",
            "none"     = "gray80"
        ),
        name = "Sentiment tweetu"
    ) +
    labs(title = "Cena akcie vs. sentiment Muskova tweetu", y = "Cena (close)", x = "Datum") +
    theme_minimal()





# PCA ================
# FIX: Mozna oddelat ema_20 - podobny sma_20, bb_up a bb_dn odvozeno od sma obou a podobne atr
technical_vars <- tibble_data %>%
    select(
        sma_20,
        sma_50,
        # ema_20,
        # basic_volatility,
        atr,
        rsi,
        macd,
        macd_signal,
        # bb_up,
        # bb_dn,
        obv,
        stochrsi,
        adx
    ) |>
    na.omit()

# TODO: Poradit si s autokorelaci - diff() asi
# TODO: ASI i stacionarni? takze fakt ten diff a testovat
adf_results <- sapply(technical_vars, function(x) {
    adf.test(na.omit(x))$p.value
})
print(adf_results)

tech_vars_adj <- technical_vars |>
    mutate(
        across(
            c(
                atr,
                obv
            ),
            ~ diff(.)
        )
    )

adf_results <- sapply(technical_vars, function(x) {
    adf.test(na.omit(x))$p.value
})
print(adf_results)


# NOTE: scaling - normalizovano na pruemr = 0 sd = 1 argumenty
# FIX: ????! Z-score normalizace
pca_result <- prcomp(technical_vars, center = TRUE, scale. = TRUE)

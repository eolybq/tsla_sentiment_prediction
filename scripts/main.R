rm(list = ls())

library(tidyverse)
library(tseries)
library(corrr)
library(factoextra)
library(car)
library(forecast)

load("data/cleandata/tibble_data.RData")


# NOTE: Stacionarita a tabulka p hodnot testu
check_stationarity <- function(data) {
    data <- data |>
        drop_na()

    results <- tibble(
        Variable = character(),
        KPSS_p_value = numeric(),
        ADF_p_value = numeric(),
        PP_p_value = numeric(),
    )

    for (var in names(data)) {
        var_data <- data[[var]]
        kpss_test <- kpss.test(var_data, null = "Level")
        adf_test <- adf.test(var_data, alternative = "stationary")
        pp_test <- pp.test(var_data, alternative = "stationary")
        ndiffs_val <- ndiffs(var_data, test = "adf", max.d = 10, alpha = 0.05, type = "level")

        results <- rbind(results, tibble(
            Variable = var,
            KPSS_p_value = kpss_test$p.value,
            ADF_p_value = adf_test$p.value,
            PP_p_value = pp_test$p.value,
        ))
    }

    return(results)
}

# NOTE: acf pacf fce
# ACF - stacionarita | PACF - sezonnost ===========
acf_pacf <- function(data, maxlag) {
    if (class(data)[1] %in% c("numeric", "ts")) {
        acf(as.numeric(data), lag.max = maxlag, main = paste("ACF for", deparse(substitute(data))))
        pacf(as.numeric(data), lag.max = maxlag, main = paste("PACF for", deparse(substitute(data))))
    } else if (class(data)[1] %in% c("tbl_df", "tbl", "data.frame")) {
        imap(data[, -1], function(x, y) {
            # par(mfrow = c(1, 2))

            acf(as.numeric(x), lag.max = maxlag, main = paste("ACF for", y))
            pacf(as.numeric(x), lag.max = maxlag, main = paste("PACF for", y))
        })
    }
}


# ========================================================
# DATA
# ========================================================


# EDA ================

# Descriptive statistics
summary(tibble_data)
# Počet NA
colSums(is.na(tibble_data))
# Počet unikátních hodnot sentimentu tweetu Elona
table(tibble_data$tweets_sentiment)
# Korelace plot
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
# NOTE: oddelano ema_20 - podobny sma_20, bb_up a bb_dn odvozeno od sma obou a podobne atr
technical_vars <- tibble_data |>
    select(
        date,
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
    drop_na()

# NOTE: Poradit si s autokorelaci - diff() asi
# NOTE: stacionarni - takze fakt ten diff a testovat
technical_vars |>
    select(-date) |>
    check_stationarity()

tech_vars_adj <- technical_vars |>
    mutate(
        across(
            c(
                sma_20,
                sma_50,
                atr,
                obv
            ),
            ~ c(NA, diff(.))
        )
    ) |>
    drop_na()

tech_vars_adj |>
    select(-date) |>
    check_stationarity()


# NOTE: scaling - normalizovano na mean = 0 sd = 1 argumenty
# FIX: ????! Z-score normalizace
pca_result <- prcomp(
    select(tech_vars_adj, -date),
    center = TRUE, scale. = TRUE
)

# Scree plot - zobrazuje % vysvětlené variance každou hlavní komponentou
fviz_eig(pca_result, addlabels = TRUE, ylim = c(0, 50))

# korelace s původních s PC's
print(pca_result$rotation)

# Graf zátěží (loadings) prvních dvou komponent - ukazuje, jak proměnné přispívají
fviz_pca_var(
    pca_result,
    col.var = "contrib", # barvení podle příspěvku proměnných
    gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
    repel = TRUE
)

# Graf skóre prvních dvou hlavních komponent (PC1 vs PC2)
fviz_pca_ind(
    pca_result,
    col.ind = "cos2", # barvení podle kvality reprezentace
    gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
    repel = TRUE
) # zabrání překrývání popisků


# NOTE: na zaklade scree plot -> VZIT 4 A NEBO 5-6 PC - SPIS 4 - usporne a 80,8 % var
pc_scores <- as_tibble(pca_result$x[, 1:4])
pc_scores$date <- tech_vars_adj$date


tibble_data_all <- tibble_data |>
    select(
        -c(
            # vynechavam promenne co jsou v PCA
            sma_20,
            sma_50,
            atr,
            rsi,
            macd,
            macd_signal,
            obv,
            stochrsi,
            adx,
            # bb_ nestacionarni - asi problem a jsou odvozeny od sma a stejne uz ATR v PCA, VIX index
            bb_up,
            bb_dn,
            # ema_20 je podobná sma_20 a 50 ktere jsou uz v PCA
            ema_20,
            # NOTE: zatim zahrnout idk
            # basic_volatility,
            # kvuli multikolinearitě zanechavam jen bull - bear spread
            bullish_surv,
            bearish_surv,
            # NOTE: zanechavam neutral_surv - neni problem s korelaci
            # neutral_surv,
        )
    ) |>
    left_join(
        pc_scores,
        by = "date"
    )


# Korelace
tibble_data_all |>
    select(
        -date,
        -close,
        -tweets_sentiment
    ) |>
    cor(use = "pairwise.complete.obs")




# TRANSFORMATIONS ================
# Stacionarita
tibble_data_all |>
    select(-date) |>
    check_stationarity()
tibble_data_all |>
    drop_na() |>
    acf_pacf(40)


trans_tdata <- tibble_data_all |>
    mutate(
        # close = log(close)
        close = c(NA, diff(log(close))),
        # NOTE: zpozdeni sentiment, tweets, trend vars
    ) |>
    drop_na()

# Stacionarita
trans_tdata |>
    select(-date) |>
    check_stationarity()
trans_tdata |>
    drop_na() |>
    acf_pacf(40)


# Korelace
trans_tdata |>
    select(
        -date,
        -close,
        -tweets_sentiment
    ) |>
    cor(use = "pairwise.complete.obs")

# Multikolinearita
lm(close ~ ., data = select(trans_tdata, -date)) |>
    vif()


# ========================================================
# MODELS
# ========================================================

# ESTIMATIONS ================




# DIAGNOSTICS ================
residuals_h <- residuals(res_var_model_h)

as_tibble(residuals_h) |>
    add_column(new_col = NA, .before = 1) |>
    acf_pacf(20)

serial.test(res_var_model_h) # Autokorelace
arch.test(res_var_model_h) # Heteroskedasticita
normality.test(res_var_model_h) # Normalita

# KPSS test
kpss_results_h <- apply(residuals_h, 2, kpss.test)
# ADF test
adf_results_h <- apply(residuals_h, 2, adf.test)
# PP test
pp_results_h <- apply(residuals_h, 2, pp.test)
# Print the results
print(kpss_results_h)
print(adf_results_h)
print(pp_results_h)




# PREDICTIONS ================

rm(list = ls())

library(conflicted)
conflict_prefer("select", "dplyr")
conflict_prefer("VAR", "vars")
library(progress)
library(tidyverse)
library(tseries)
library(corrr)
library(factoextra)
library(car)
library(forecast)
library(vars)
library(writexl)
library(gridExtra)

load("cleandata/tibble_data.RData")


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





# PCA CELY VZOREK ================
# NOTE: I Priprava dat pro PCA vstupujici do modelu
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
    )

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

# JEN PRO VYBER LAGU JINAK PCA V KAZDEM WINDOW
# NOTE: na zaklade scree plot celeho vzorku? -> VZIT 4 A NEBO 5-6 PC - SPIS 4 - usporne a 80,8 % var
pc_scores_whole <- as_tibble(pca_result$x[, 1:4])
pc_scores_whole$date <- tech_vars_adj$date





# TRANSFORMATIONS ================
# Stacionarita
tibble_data |>
    select(-date) |>
    check_stationarity() |>
    print(n = 22)
tibble_data |>
    drop_na() |>
    acf_pacf(40)


trans_tdata <- tibble_data |>
    mutate(
        # NOTE: diff log na cenu v kazdem rolling window kazdeho modelu
        close = c(NA, diff(log(close))),


        # NOTE: Zpozdeni tweets, trend, sentiment vars, vix index proste exog vars
        tweets_sentiment = c(
            factor(NA, levels = levels(tweets_sentiment)),
            tweets_sentiment[-length(tweets_sentiment)]
        ),
        g_trends = c(NA, g_trends[-length(g_trends)]),
        neutral_surv = c(NA, neutral_surv[-length(neutral_surv)]),
        bull_bear_spread_surv = c(NA, bull_bear_spread_surv[-length(bull_bear_spread_surv)]),
        vix_close = c(NA, vix_close[-length(vix_close)]),

    ) |>
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
    drop_na()


ggplot(trans_tdata, aes(x = date)) +
    geom_line(aes(y = close), color = "blue") +
    labs(title = "Vývoj závěrečné ceny po transformaci", y = "Cena", x = "Datum")


# NOTE: rozdeleni na VAR a ARIMA data a exog data
var_data <- trans_tdata |>
    select(
        -tweets_sentiment,
        -g_trends,
        -bull_bear_spread_surv,
        -neutral_surv,
        -vix_close
    )

arima_data <- trans_tdata |>
    select(close)

exog_data <- trans_tdata |>
    select(
        tweets_sentiment,
        g_trends,
        bull_bear_spread_surv,
        neutral_surv,
        vix_close
    )

# NOTE: Matice pro exogenní proměnné pro odhady
# NOTE: vix_close kvuli tomu ze je to index celeho trhu
xreg_mat <- model.matrix(~ tweets_sentiment + vix_close + g_trends + bull_bear_spread_surv + neutral_surv, data = exog_data)
xreg_mat <- xreg_mat[, colnames(xreg_mat) != "(Intercept)"]

xreg_mat_full <- model.matrix(~ tweets_sentiment + vix_close + g_trends + bull_bear_spread_surv + neutral_surv, data = tibble_data |> 
    select(
        tweets_sentiment,
        g_trends,
        bull_bear_spread_surv,
        neutral_surv,
        vix_close
    ))
xreg_mat_full <- xreg_mat_full[, colnames(xreg_mat_full) != "(Intercept)"]



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


# PCA fce pro VAR modely - spoji se s all_data
pca_func <- function(tech_vars, all_data) {
    # NOTE: scaling - normalizovano na mean = 0 sd = 1 argumenty
    pca_result <- prcomp(
        select(tech_vars, -date),
        center = TRUE, scale. = TRUE
    )

    # NOTE: na zaklade scree plot celeho vzorku? -> VZIT 4 A NEBO 5-6 PC - SPIS 4 - usporne a 80,8 % var
    pc_scores <- as_tibble(pca_result$x[, 1:4])
    pc_scores$date <- tech_vars$date

    tibble_data_all <- all_data |>
        left_join(
            pc_scores,
            by = "date"
        ) |>
        select(
            -date
        )

    return(tibble_data_all)
}



# Lag selection ================

# ARIMA
# NOTE: ARIMA(2, 0, 2) optimální na celém vzorku
arima_data |>
    auto.arima()


# ARIMAX
# NOTE: ARIMA(2, 0, 2) optimální na celém vzorku
arima_data |>
    auto.arima(
        xreg = xreg_mat
    )


# VAR
# NOTE: VAR(36 / 16 / 4 / 36) optimální na celém vzorku
var_data |>
    # NOTE: pridani jen docasne PCA na celem vzorku pro vyber lagu
    left_join(
        pc_scores_whole,
        by = "date"
    ) |>
    select(
        -date
    ) |>
    VARselect(
        lag.max = 40,
        type = "const",
    )


# VARX
# NOTE: VARX(36 / 16 / 4 / 36) optimální na celém vzorku
var_data |>
    # NOTE: pridani jen docasne PCA na celem vzorku pro vyber lagu
    left_join(
        pc_scores_whole,
        by = "date"
    ) |>
    select(
        -date
    ) |>
    VARselect(
        lag.max = 40,
        type = "const",
        exogen = xreg_mat
    )

arima_lag <- 1
var_lag <- 4




# PREDICTIONS ================

window_size <- 1204
n <- nrow(trans_tdata)
n_iter <- n - window_size
dates <- trans_tdata$date


# Výstupní tibble pro predikce a skutečné hodnoty
results <- tibble(
    date = as.Date(character()),
    actual = numeric(),
    arima_pred = numeric(),
    arimax_pred = numeric(),
    var_pred = numeric(),
    varx_pred = numeric()
)





pb <- progress_bar$new(
    total = n_iter,
    format = "  [:bar] :percent ETA: :eta",
    clear = FALSE,
    width = 60
)

# MAIN LOOP
for (i in 1:n_iter) {
    actual_value <- var_data[i + window_size, "close", drop = TRUE]
    pred_date <- dates[i + window_size]

    window_tech_v <- tech_vars_adj[i:(i + window_size - 1), ]
    window_var <- var_data[i:(i + window_size - 1), ]

    window_var_pca <- pca_func(
        window_tech_v,
        window_var
    )


    window_exog <- xreg_mat[i:(i + window_size - 1), ]


    window_arima <- arima_data[i:(i + window_size - 1), ]
    # ZAROVNANI DAT ABY BYLA VSTUPNI DATA STEJNA PRO ARIMA A VAR MODELY S JINYM LAGEM
    train_arima <- window_arima[(var_lag - arima_lag):nrow(window_arima), ]
    xreg_arima <- window_exog[(var_lag - arima_lag):nrow(window_exog), ]


    # TODO: mozna osetrit kvuli error
    # fit <- tryCatch({
    # arima(train_arima, order = c(p, 0, q), method = "ML")
    # }, error = function(e) {
    # return(NA)  # nebo jiná logika, např. zkus jiný model
    # })
    # ARIMA
    arima_model <- arima(
        train_arima,
        order = c(arima_lag, 0, arima_lag),
        method = "CSS-ML"
    )
    arima_pred <- forecast(
        arima_model,
        h = 1
    )$mean

    # ARIMAX
    arimax_model <- arima(
        train_arima,
        order = c(arima_lag, 0, arima_lag),
        xreg = xreg_arima,
        method = "CSS-ML"
    )
    newreg <- xreg_mat_full[i + window_size, , drop = FALSE]
    arimax_pred <- predict(
        arimax_model,
        n.ahead = 1,
        newxreg = newreg
    )$pred

    # VAR
    var_model <- VAR(
        window_var_pca,
        p = var_lag,
        type = "const"
    )
    var_pred <- predict(var_model, n.ahead = 1)$fcst$close[, "fcst"]

    # VARX
    varx_model <- VAR(
        window_var_pca,
        p = var_lag,
        type = "const",
        exogen = window_exog
    )
    dumvar <- xreg_mat_full[i + window_size, , drop = FALSE]
    varx_pred <- predict(varx_model, n.ahead = 1, dumvar = dumvar)$fcst$close[, "fcst"]



    results <- results |>
        add_row(
            date = pred_date,
            actual = actual_value,
            arima_pred = as.numeric(arima_pred),
            arimax_pred = as.numeric(arimax_pred),
            var_pred = as.numeric(var_pred),
            varx_pred = as.numeric(varx_pred)
        )



    # Tick progress
    # pb$tick()
}


# VYHODNOCENÍ PREDIKCÍ =============
naive_pred <- c(NA, results$actual[-length(results$actual)])

# Funkce pro výpočet metrik
calc_metrics <- function(actual, predicted) {
    # Odstraníme případy, kde actual je 0 pro výpočet MAPE
    nonzero_idx <- actual != 0
    
    mse <- mean((actual - predicted)^2, na.rm = TRUE)
    rmse <- sqrt(mse)
    mae <- mean(abs(actual - predicted), na.rm = TRUE)
    
    # NOTE: Extremni hodnoty asi protoze actuals blizko 0, nedava smysl
    # mape <- if (any(nonzero_idx)) {
    #     mean(abs((actual[nonzero_idx] - predicted[nonzero_idx]) / actual[nonzero_idx]), na.rm = TRUE) * 100
    # } else {
    #     NA_real_
    # }
    # 
    # # SMAPE
    # denominator <- (abs(actual) + abs(predicted)) / 2
    # smape_vals <- ifelse(denominator == 0, 0, abs(predicted - actual) / denominator)
    # smape <- mean(smape_vals, na.rm = TRUE) * 100
    
    # MASE (porovnání s naivní jednofázovou predikcí)
    naive_errors <- abs(diff(actual))  # rozdíl mezi actual[t] a actual[t-1]
    scale <- mean(naive_errors, na.rm = TRUE)
    
    mase <- if (!is.na(scale) && scale != 0) {
        mae / scale
    } else {
        NA_real_
    }
    
    tibble(MSE = mse, RMSE = rmse, MAE = mae, MASE = mase)
}

metrics <- tibble(
    model = c("arima", "arimax", "var", "varx"),
    MSE = NA_real_,
    RMSE = NA_real_,
    MAE = NA_real_,
    # MAPE = NA_real_,
    # SMAPE = NA_real_
    MASE = NA_real_,
)

metrics <- tibble(model = c("arima", "arimax", "var", "varx")) |>
    rowwise() |>
    mutate(
        metrics = list(calc_metrics(results$actual, results[[paste0(model, "_pred")]]))
    ) |>
    unnest_wider(metrics) |>
    ungroup()

naive_metrics <- calc_metrics(results$actual, naive_pred) |>
    mutate(model = "naive") |>
    select(model, everything())

metrics <- bind_rows(metrics, naive_metrics)


# WRITE:
#„Model se mýlí přibližně tolik, kolik samotná data kolísají.“
# Což je OK – lepší by bylo být pod tím.
print(metrics)
sd(trans_tdata$close)

write_xlsx(metrics, "plots_tabs/metrics.xlsx")



# 1. Grafy zvlášť pro každý model (skutečné vs predikované)
for(model_name in c("arima", "arimax", "var", "varx")) {
    p <- ggplot(results, aes(x = date)) +
        geom_line(aes(y = actual), color = "black", size = 1, alpha = 0.7) +
        geom_line(aes_string(y = paste0(model_name, "_pred")), color = "blue", size = 1, alpha = 0.7) +
        labs(title = paste("Actual vs Predicted -", toupper(model_name)),
             y = "Value",
             x = "Date") +
        theme_minimal()
    
    print(p)
}

# 2. Společný graf všech modelů najednou
# Převod do long formátu pro ggplot
results_long <- results %>%
    select(date, actual, arima_pred, arimax_pred, var_pred, varx_pred) %>%
    pivot_longer(
        cols = -date,
        names_to = "series",
        values_to = "value"
    )

# Pro lepší popisky upravíme název série
results_long <- results_long %>%
    mutate(series = case_when(
        series == "actual" ~ "Actual",
        series == "arima_pred" ~ "ARIMA",
        series == "arimax_pred" ~ "ARIMAX",
        series == "var_pred" ~ "VAR",
        series == "varx_pred" ~ "VARX"
    ))

# Vykreslení
ggplot(results_long, aes(x = date, y = value, color = series)) +
    geom_line(size = 1, alpha = 0.8) +
    labs(title = "Actual vs Predicted for All Models",
         x = "Date",
         y = "Value",
         color = "Series") +
    theme_minimal()





# Připravíme residuals dlouhý formát
residuals_long <- results |>
    pivot_longer(
        cols = ends_with("_pred"),
        names_to = "model",
        values_to = "predicted"
    ) |>
    mutate(
        model = sub("_pred$", "", model),  # odstraníme '_pred' na konci názvu
        residual = actual - predicted
    )

# Facetovaný graf pro každý model zvlášť
ggplot(residuals_long, aes(x = date, y = residual)) +
    geom_line(alpha = 0.7) +
    facet_wrap(~ model, scales = "free_y") +
    labs(
        title = "Residuals (chyby predikcí) podle modelu v čase",
        x = "Datum",
        y = "Residual (actual - predicted)"
    ) +
    theme_minimal()



# ACF REZIDUI PREDIKCI
acf(results$actual - results$arima_pred, main = "ACF reziduí ARIMA")
Box.test(results$arima_pred - results$actual, lag = 20, type = "Ljung-Box")
acf(results$actual - results$arimax_pred, main = "ACF reziduí ARIMAX")
acf(results$actual - results$var_pred, main = "ACF reziduí VAR")
acf(results$actual - results$varx_pred, main = "ACF reziduí VARX")

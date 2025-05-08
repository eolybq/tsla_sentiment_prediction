library(tidyverse)
library(tidyquant)
library(TTR)

rawdata <- tq_get("TSLA", from = "2010-06-01", to = "2025-05-01")

rawdata$sma_20 <- zoo::rollapply(rawdata$close, 20, mean, fill = NA)
rawdata$sma_50 <- zoo::rollapply(rawdata$close, 50, mean, fill = NA)

rawdata$ema_20 <- TTR::EMA(rawdata$close, n = 20)

rawdata$basic_volatility <- rawdata$high - rawdata$low

rawdata$atr <- ATR(cbind(rawdata$high, rawdata$low, rawdata$close), n = 14)$atr

rawdata$rsi <- TTR::RSI(rawdata$close, n = 14)

macd <- TTR::MACD(rawdata$close, nFast = 12, nSlow = 26, nSig = 9)
rawdata$macd <- macd$macd
rawdata$macd_signal <- macd$signal

bb <- TTR::BBands(rawdata$close, n = 20)
rawdata$bb_up <- bb$up
rawdata$bb_dn <- bb$dn

rawdata$obv <- TTR::OBV(rawdata$close, rawdata$volume)

rawdata$stochrsi <- TTR::stochRSI(rawdata$close, n = 14, maType = "SMA")

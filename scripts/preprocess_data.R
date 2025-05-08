library(tidyverse)
library(tidyquant)
library(TTR)

rawdata <- tq_get("TSLA", from = "2010-06-01", to = "2025-05-01")



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



# Průměr zavíracích cen za posledních X dní Sleduje trend. Pokud je cena nad SMA, je to býčí signál, pod SMA je medvědí signál.
rawdata$sma_20 <- zoo::rollapply(rawdata$close, 20, mean, fill = NA)
rawdata$sma_50 <- zoo::rollapply(rawdata$close, 50, mean, fill = NA)

# Vážený průměr, který dává větší důraz na novější data. Rychleji reaguje na změny trendu než SMA.
rawdata$ema_20 <- EMA(rawdata$close, n = 20)

# Rychlý odhad volatility bez složitého modelování.
rawdata$basic_volatility <- rawdata$high - rawdata$low

# Průměrný rozsah cenového pohybu včetně gapů, přesnější měření volatility než basic_volatility. Měří tržní riziko; vyšší ATR = vyšší riziko.
rawdata$atr <- ATR(cbind(rawdata$high, rawdata$low, rawdata$close), n = 14)$atr

# Indikátor momenta, který měří poměr kladných vs. záporných výnosů za určité období
rawdata$rsi <- RSI(rawdata$close, n = 14)

# Rozdíl mezi dvěma EMA (např. 12 a 26 dní) + signální čára (9denní EMA). Sleduje sílu a směr trendu. MACD překročí signal line směrem nahoru = nákupní signál. Směrem dolů = prodejní signál.
macd <- MACD(rawdata$close, nFast = 12, nSlow = 26, nSig = 9)
rawdata$macd <- macd$macd
rawdata$macd_signal <- macd$signal

# Horní a dolní pásmo kolem klouzavého průměru. Určuje volatilitu. Vypočítává se jako SMA ± 2× směrodatná odchylka. Cena dotýkající se horního pásma = potenciální překoupení. Dotýkající se spodního = přeprodanost.
bb <- BBands(rawdata$close, n = 20)
rawdata$bb_up <- bb$up
rawdata$bb_dn <- bb$dn

# Kumulativní součet objemu obchodování s ohledem na směr ceny. Pokud roste OBV spolu s cenou, potvrzuje trend. Divergence (např. cena roste, OBV klesá) varuje před obratem.
rawdata$obv <- OBV(rawdata$close, rawdata$volume)

# Aplikuje stochastický oscilátor na RSI, takže výsledná hodnota je mezi 0 a 1.
rawdata$stochrsi <- stochRSI(rawdata$close, n = 14, maType = "SMA")

# Sila trendu bez smeru
adx <- ADX(cbind(rawdata$high, rawdata$low, rawdata$close), n = 14)
rawdata$adx <- adx$ADX

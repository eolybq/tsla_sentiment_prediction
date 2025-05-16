rm(list = ls())

library(tidyverse)

load("data/cleandata/tibble_data.RData")


# PCA ================
# TODO: Poradit si s autokorelaci - diff() asi
# TODO: ASI i stacionarni? takze fakt ten diff a testovat
# TODO: scaling - normalizovat na pruemr = 0 sd = 1 scale()
# ! Z-score normalizace
# TODO: seasonality - ocistit treba seas()?
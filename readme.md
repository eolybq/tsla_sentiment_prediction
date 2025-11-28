# Tesla Price Sentiment Based ML Prediction

- **Part 1 of this project has its own report available (in Czech) in this file: [Report in PDF](r/report.pdf) or [Report in DOCX](r/report.docx)**


This project explores and evaluates different approaches for predicting stock prices using financial and sentiment time series data. Its main goal is to identify the most effective models and assess their practical performance in trading scenarios.
This project also introduces unusual data - sentiment indicator derived from Elon Musk's tweets from 2010 to 2025 which contains keywords related to "Tesla" company. Elon's tweets can be influential to markets, so sentiment indicated by each tweet related to Tesla might incorporate relevant data to my models. FinBERT model is used to make NLP analysis to get sentiment indicators.


# Project Overview

1. **Analysis framework:**  
   - Analyze a large unique dataset to identify the best-performing models (ARIMA, VAR, basic ML models, etc.).  
   - Incorporate NLP assesed sentiment from Elon's tweets with 4 different levels - positive, negative, neutral or none (Elon didn't tweet about tesla)
   - Dataset also contains survey data representing spread between % of bullish or bearish investors, google trends data of search term "tesla", financial indicators (VIX, TSLA volume, SMA, ATR, stochRSI, MACD, ADX etc.)
   - Evaluate models on prediction accuracy metrics and visualizations.

2. **Prediction Approaches:**  
   - Implementing ARIMA or VAR with exogenous variables for improved accuracy, as pure endogenous approach might be less suitable when some of exogenous indicators (VIX, bulish bearish spread..) are not directly influenced by the target stock price.  
   - Try alternative basic ML methods to compare them with traditional time series models
   - Try to incorporate classification prediction task as predicting wheter log return will be positive or neutral / negative


## Part 1 (directory r/)
**Unconventional data, conventional models**
- Main programming language: R (python used to get FinBERT from HuggingFace)

- Project evaluates time-series models (ARIMA, ARIMAX, VAR, VARX) on unique data to get answer on hypothesis, that unique Elon Musk's tweets sentiment indicator data, volatility data and market sentiment data bring a lot of information to model predicting financial stock data. 
- I also assume, that usage of more features such as financial indicators would improve accuracy
- Projects uses different model architectures to assess different approaches and compare their evaluation metrics -> their performance.

## Part 2 (directory python/)
**Unconventional data, uncoventional models**
- Main programming language: Python

- Project aims to extend Part 1 of this project. Extension lies in using python instead of R to create machine-learning models and their prediction evaluations. Then compare traditional time-series models and these models to get result of perfomance on financial data.
- Part 2 also explores classification ML methods which cannot be directly compared to regression task models from Part 1
- Classification is based on predicting artificialy created binary variable, which takes on value 1 if stock log return went over some "significance" treshold (0.005)
- At the moment there is Gradient Descent Linear Regression and Gradient Descend Logistic Regression done, written from scratch.
- Projects also uses regression and classification Decision Trees, Random Forests, XGBoost and LightGBM models.
- All models are trained recursively on rolling window -> walk forward prediction validation on same data
- LSTM is planned aswell.


# Tools used
- Python (Pandas, Numpy, SK-learn, XGBoost, LightGBM, Statsmodels, Seaborn, Matplotlib, Transformers - HuggingFace)
- R (Tidyverse, Tidyquant)

- HuggingFace - model FinBERT -> NLP sentiment analysis
- PCA



# Outcomes

## Traditional time series models (Part 1)

**Predictions evaluation across models:**

| Model     | MSE       | RMSE      | MAE       | MASE     |
|-----------|-----------|-----------|-----------|----------|
| **ARIMA** | **0.0014** | **0.0381** | **0.0267** | **0.6823** |
| ARIMAX    | 0.0015    | 0.0383    | 0.0268    | 0.6858   |
| VAR       | 0.0015    | 0.0383    | 0.0271    | 0.6925   |
| VARX      | 0.0015    | 0.0384    | 0.0272    | 0.6962   |
| Naive     | 0.0029    | 0.0540    | 0.0391    | 1.0000   |

- Basic univariate **ARIMA** model without unique data used in this project beats ARIMAX with this data aswell as multivariate models.
- Every model beats Naive model

## Unconventional models (ML models) (Part 2)

**Regression:**

| Model                     | MSE       | RMSE     | MAE      | MASE   | R²          |
|---------------------------|-----------|----------|----------|--------|-------------|
| **RandomForestRegressor** | **0.00146** | **0.03822** | **0.02815** | **0.681** | **0.00208** |
| GDLinearRegression        | 0.00151   | 0.03880  | 0.02852  | 0.690  | -0.0282     |
| DecisionTreeRegressor     | 0.00154   | 0.03919  | 0.02865  | 0.693  | -0.0493     |
| LightGBMRegressor         | 0.00167   | 0.04092  | 0.03026  | 0.732  | -0.1438     |
| XGBoostRegressor          | 0.00168   | 0.04097  | 0.03060  | 0.740  | -0.1464     |
| Naive                     | 0.00294   | 0.05419  | 0.04133  | 1.000  |             |

- Best model **RandomForest** gets approximately same results as best model from Part 1 (univariate ARIMA)
- Most sofisticated models (XGBoost, LightGBM) perform the worst on these metrics, but visual inspection of predictions show they really capture the data patterns better than others
- Plots provided below show that some models get quite good predictions
- All models beat Naive model 


**Classification:**

| Model                    | Accuracy    | Log Loss    | ROC AUC     | Precision (weighted) | Recall (weighted) | F1-score (weighted) |
|--------------------------|-------------|-------------|-------------|----------------------|-------------------|---------------------|
| **GDLogisticRegression** | **0.57365** | 0.68822     | **0.56045** | **0.58240**          | **0.57365**       | **0.57591**         |
| RandomForestClassifier   | 0.55807     | **0.68024** | 0.54499     | 0.54962              | 0.55807           | 0.55129             |
| LightGBMClassifier       | 0.56374     | 0.82974     | 0.52811     | 0.54943              | 0.56374           | 0.54777             |
| XGBoostClassifier        | 0.54249     | 0.84326     | 0.51572     | 0.53507              | 0.54249           | 0.53715             |
| DecisionTreeClassifier   | 0.52408     | 0.91675     | 0.51313     | 0.52002              | 0.52408           | 0.52164             |

- Models show accuracy slightly better than randomness (0.5) although **GD Logistic Regression** performs quite good on all metrics
- Weighted recall and precision indicate models have limited ability to correctly identify all positive cases and avoid false positives
- Log Loss metric indicates models have limited confidence in their predictions
- On ROC AUC metric, models show relatively weak ability to differentiate between classes across different thresholds


### Short-term stock returns exhibit very low predictability. All tested models achieved only marginal improvements over randomness, which is consistent with the known difficulty of short-horizon financial forecasting. Moreover, sentiment extracted from tweets does not provide strong predictive power in these settings.
- **Though regression models perform quite well on visualizations, metrics show they aren't really reliable**   
- **There aren't any significant patterns/relationships in this data finded, which when modeled with these models, could be reliably used to predict log return, or rise/other binary variable of TSLA. All classificator models shows accuracy slightly above 0.50 and regressor models show similar MSE to basic ARIMA of low order, which also by plot show weak oscilation around mean (~0) of the log return timeseries.**   


- **NOTE**:  
When trying also simple Decision Trees and Random Forest only with past values of log return (1-10 lags) to avoid using irelevant features (all of the used features also doesn't really show high abs corellation with y), results are similar, ofter worse

## Predictions in plot:
### Part 1  
![TS models](r/plots_tabs/preds.png)  


### Part 2  
**Regression:**  

![GD LinearRegression](python/plots_tabs/GDLinearRegression_preds.png)
![DecisionTreeRegressor](python/plots_tabs/DecisionTreeRegressor_preds.png)
![RandomForestRegressor](python/plots_tabs/RandomForestRegressor_preds.png)
![XGBoostRegressor](python/plots_tabs/XGBoostRegressor_preds.png)
![LightGBMRegressor](python/plots_tabs/LightGBMRegressor_preds.png)

**Classification:**
![GD Logit Confusion matrix](python/plots_tabs/GDLogisticRegression_conf_m.png)
![DecisionTreeClassifier Confusion Matrix](python/plots_tabs/DecisionTreeClassifier_conf_m.png)
![RandomForestClassifier Confusion Matrix](python/plots_tabs/RandomForestClassifier_conf_m.png)
![XGBoostClassifier Confusion Matrix](python/plots_tabs/XGBoostClassifier_conf_m.png)
![LightGBMClassifier Confusion Matrix](python/plots_tabs/LightGBMClassifier_conf_m.png)

- **NOTE:**  
1 express that log_return is > 0.005 (log return rises atleast by this "significance" threshold) and 0 otherwise (log return is constant or negative)

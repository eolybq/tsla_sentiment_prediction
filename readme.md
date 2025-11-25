# Tesla Price Prediction Project

- **Part 1 of this project has its own report available (in Czech) in this file: [Report in PDF](r/report.pdf) or [Report in DOCX](r/report.docx)**


This project explores and evaluates different approaches for predicting stock prices using financial time series data. Its main goal is to identify the most effective models and assess their practical performance in trading scenarios.
This project also introduces unusual data - sentiment indicator derived from Elon Musk's tweets from 2010 to 2025 which contains keywords related to "Tesla" company. Elon's tweets can be influential to markets, so sentiment indicated by each tweet related to Tesla might incorporate relevant data to my models. FinBERT model is used to make NLP analysis to get sentiment indicators.


## Project Overview

1. **Analysis framework:**  
   - Analyze a large unique dataset to identify the best-performing models (ARIMA, VAR, basic ML models, etc.).  
   - Incorporate NLP assesed sentiment from Elon's tweets with 4 different levels - positive, negative, neutral or none (Elon didn't tweet about tesla)
   - Dataset also contains survey data representing spread between % of bullish or bearish investors, google trends data of search term "tesla", financial indicators (VIX, TSLA volume, SMA, ATR, stochRSI, MACD, ADX etc.)
   - Evaluate models on prediction accuracy

2. **Prediction Approaches:**  
   - Implementing ARIMA or VAR with exogenous variables for improved accuracy, as pure endogenous approach might be less suitable when some of exogenous indicators (VIX, bulish bearish spread..) are not directly influenced by the target stock price.  
   - Try alternative basic ML methods to compare them with traditional time series models
   - Try to incorporate classification prediction task as predicting wheter log return is positive or neutral / negative


### Part 1 (directory r/)
**Unconventional data, conventional models**
- Main programming language: R (python used to get FinBERT from HuggingFace)

- Project evaluates time-series models (ARIMA, ARIMAX, VAR, VARX) on unique data to get answer on hypothesis, that unique Elon Musk's tweets sentiment indicator data, volatility data and market sentiment data bring a lot of information to model predicting financial stock data. 
- I also assume, that usage of more features such as financial indicators would improve accuracy
- Projects uses different model architectures to assess different approaches and compare their evaluation metrics -> their performance.

### Part 2 (directory python/)
**Unconventional data, uncoventional models**
- Main programming language: Python

- Project aims to extend Part 1 of this project. Extension lies in using python instead of R to create machine-learning models and their prediction evaluations. Then compare traditional time-series models and these models to get result of perfomance on financial data.
- Part 2 also explores classification ML methods which cannot be directly compared to regression task models from Part 1
- Classification is based on predicting artificialy created binary variable, which takes on value 1 if stock log return went over some "significance" treshold (0.005)
- At the moment there is Gradient Descent Linear Regression and Gradient Descend Logistic Regression done, written from scratch.
- Also Decision Tree classification and regression, Random Forest and XGBoost classification using sk-learn, xgboost libraries are done. (trained recursively on rolling window - walk forward prediction validation)
- LSTM is planned aswell.


## Tools used
- Python (Pandas, Numpy, SK-learn, Statsmodels, Seaborn, Matplotlib, Transformers - HuggingFace)
- R (Tidyverse, Tidyquant)
- HuggingFace - model FinBERT

- Time series modeling: ARIMA, VAR, ARIMAX, VARX, Gradient descent linear regression, Gradient descent LOGIT.  
- Feature engineering with financial indicators, market and sentiment data. PCA is also used to reduce dimensionality. 


## Outcomes

### Traditional time series models (Part 1)

**Predictions evaluation across models:**

| Model     | MSE       | RMSE      | MAE       | MASE     |
|-----------|-----------|-----------|-----------|----------|
| **ARIMA** | **0.0014** | **0.0381** | **0.0267** | **0.6823** |
| ARIMAX    | 0.0015    | 0.0383    | 0.0268    | 0.6858   |
| VAR       | 0.0015    | 0.0383    | 0.0271    | 0.6925   |
| VARX      | 0.0015    | 0.0384    | 0.0272    | 0.6962   |
| Naive     | 0.0029    | 0.0540    | 0.0391    | 1.0000   |

- Basic univariate ARIMA model without unique data used in this project beats ARIMAX with this data aswell as multivariate models.
- Every model beats Naive model

### Unconventional models (ML models) (Part 2)

**Regression:**

| Model                   | MSE        | RMSE       | MAE        | MASE       | R^2        |
|-------------------------|------------|------------|------------|------------|------------|
| **GD LinearRegression** | **0.0014** | **0.0368** | **0.0277** | **0.6765** | **0.0669** |
| DecisionTreeRegressor   | 0.0016     | 0.0398     | 0.0289     | 0.6991     | -0.0808    |
| Naive                   | 0.0029     | 0.0537     | 0.0410     | 1.0000     | -          |

- Model gets approximately same results as best model from Part 1 (univariate ARIMA)
- R^2 is relatively small (small % of log_return variability explained) which could be anticipated for linear model predicting complex relationships in stock data
- Part 2 uses slightly different indexes of train/test data (train_test_split vs rolling window in Part 1)
- Model beats Naive model 

**Classification:**

| Model                     | Accuracy   | Log Loss   | ROC AUC    |
|---------------------------|------------|------------|------------|
| **GD LogisticRegression** | **0.5680** | **0.6721** | **0.6090** |
| DecisionTreeClassifier    | 0.5382     | 16.6434    | 0.5165     |
| RandomForestClassifier    | 0.5326     | 0.6814     | 0.5381     |
| XGBoostClassifier         | 0.5326     | 0.8270     | 0.5268     |

- Models show accuracy slightly better than randomness (0.5)
- Log loss for Decision Tree is very high, which means weakly calibrated predicted probabilities - models is often sure with prediction that is wrong
- On ROC AUC metric, models show relatively weak ability to differentiate between classes across different thresholds


### Short-term stock returns exhibit very low predictability. All tested models achieved only marginal improvements over randomness, which is consistent with the known difficulty of short-horizon financial forecasting. Moreover, sentiment extracted from tweets does not provide strong predictive power in these settings.
**There aren't any significant patterns/relationships in this data finded, which when modeled with these models, could be reliably used to predict log return, or rise/other binary variable of TSLA. All classificator models shows accuracy slightly above 0.50 and regressor shows similar MSE to basic ARIMA of low order, which also by plot show weak oscilation around mean (~0) of the log return ts.**
- **NOTE**:  
When trying also simple Decision Trees and Random Forest only with past values of log return (1-10 lags) to avoid using irelevant features (all of the used features also doesn't really show high abs corellation with y), results are similar, ofter worse

### Predictions in plot:
**Part 1**
![TS models](r/plots_tabs/preds.png)


**Part 2**
![GD LinearRegression](python/plots_tabs/gd_lr.png)
![GD Logit Confusion matrix](python/plots_tabs/gd_log_conf_m.png)
![DecisionTreeRegressor](python/plots_tabs/dtr.png)
![DecisionTreeClassifier Confusion Matrix](python/plots_tabs/dtc_conf_m.png)
![RandomForestClassifier Confusion Matrix](python/plots_tabs/rfc_conf_m.png)
![XGBoostClassifier Confusion Matrix](python/plots_tabs/xgbc_conf_m.png)

- **NOTE:**  
1 express that log_return is > 0.005 (log return rises atleast by this "significance" threshold) and 0 otherwise (log return is constant or negative)

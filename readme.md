# Stock Price Prediction Project

- ! THIS PROJECT IS CURRENTLY IN-PROGRESS - SOME FEATURES AREN'T IMPLEMENTED YET.

This project explores and evaluates different approaches for predicting stock prices using financial time series data. Its main goal is to identify the most effective models and assess their practical performance in trading scenarios.
This project also introduce unusual data - sentiment indicator derived from Elon Musk's tweets from 2010 to 2025 which contains keywords related to "Tesla" company. Elon's tweets can be influential to markets, so sentiment indicated by each tweet related to Telsa might incorporate relevant data to my models. FinBERT model is used to make NLP analysis to get sentiment indicators.


## Project Overview

1. **Model Comparison:**  
   - Analyze a large dataset to identify the best-performing models (ARIMA, VAR, ML regression-based models, etc.).  
   - Evaluate models on prediction accuracy

2. **Prediction Approaches:**  
   - Naive comparison of models (e.g., VAR vs AR)
   - Focus on ARIMA or VAR with exogenous variables for improved accuracy, as pure endogenous approach might be less suitable when exogenous indicators do not directly influence the target price.  
   - Consider potential interactions such as volume and price, which may still influence each other.


## Tools & Techniques
- Python (Pandas, Numpy, Matplotlib), R (Tidyverse, Tidyquant)
- HuggingFace - model FinBERT

- Time series modeling: ARIMA, VAR, ARIMAX, VARX, Gradient descent linear regression, Gradient descent LOGIT.  
- Feature engineering with financial indicators and exogenous data.  
- Performance evaluation including prediction error and trading profit/loss with backtest (currently in-progress).  

The project aims to provide a comprehensive evaluation of both predictive power and practical applicability for stock price forecasting.

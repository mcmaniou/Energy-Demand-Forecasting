# Energy-Demand-Forecasting
An overview of different methods for energy demand forecasting.

This project presents a comparative evaluation of different timeseries prediction alforithms for energy demand forecasting. 

The repo is organized in the following folders:
- code : the python notebooks
- data : the data used for training and testing the models. The raw data were recovered from [1].

The algorithms implemented include:
- ARIMA: Autoregressive integrated moving average [2].
- PROPHET: An additive regression forecasting model released by Meta’s research team [3]. 
- LightGBM: A gradient boosting framework that uses tree-based learning algorithms [4].
- Transformer: Deep learning attention-based model originally designed for influenza ratio forecasting [5].


[1] Jhana N. 2019. Hourly energy demand generation and weather. Retrieved April 8, 2022 from https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather/metadata?select=energy_dataset.csv 
[2] Box GEP, Jenkins GM, Reinsel GC, Ljung GM. 2015. Time Series Analysis: Forecasting and Control. John Wiley and Sons Inc, pp. 712. ISBN: 978‐1‐118‐67502‐1.
[3] Letham B, Taylor SJ. 2017. Prophet: forecasting at scale. [Blog] Meta Research. https://research.facebook.com/blog/2017/02/prophet-forecasting-at-scale/ [Accessed April 8, 2022]
[4] Wu N, Green B, Ben X, O'Banion S. 2020. Deep transformer models for time series forecasting: The influenza prevalence case. arXiv preprint arXiv:2001.08317.
[5] Ke G, Meng Q, Finley T, Wang T, Chen W, Ma W, Ye Q, Liu TY. 2017. Lightgbm: A highly efficient gradient boosting decision tree. Advances in neural information processing systems, 30.


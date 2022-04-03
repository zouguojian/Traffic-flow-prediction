# -- coding: utf-8 --

import seaborn as sns
sns.set_style("whitegrid",{"font.sans-serif":['KaiTi', 'Arial']})

import pandas as pd
import numpy as np
from model.hyparameter import parameter
from model.utils import metric
import seaborn as sns
import argparse
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
sns.set_style("whitegrid", {"font.sans-serif": ['KaiTi', 'Arial']})

para = parameter(argparse.ArgumentParser())
para = para.get_para()

import warnings
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.8)
    train, test = X[int(len(X) * 0.6):train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    prediction = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast(steps=1)[0]
        prediction.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, prediction)
    return error, test,prediction


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    label, predict = 0, 0
    dataset = dataset.astype('float32')
    mse, test,prediction = evaluate_arima_model(dataset, [p_values, d_values, q_values])
    return label,predict

if __name__ == "__main__":
    # series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    data = pd.read_csv('/Users/guojianzou/Traffic-flow-prediction/MT-STNet/data/train.csv', encoding='utf-8')  # 数据读取
    # evaluate parameters
    # p_values = [1, 2, 4, 6]
    # d_values = range(0, 3)
    # q_values = range(0, 3)
    labels = []
    predicts = []

    warnings.filterwarnings("ignore")
    for site in range(66):
        series=data[(data['station']==site)]

        label, predict=evaluate_models(series.values[:,-1], p_values=6, d_values=0, q_values=1)
        labels.append(label)
        predicts.append(predict)

    labels=np.reshape(np.array(labels),newshape=[-1])
    predicts=np.reshape(np.array(predicts),newshape=[-1])
    metric(pred=predicts,label=labels)
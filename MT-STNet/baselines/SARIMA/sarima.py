# -- coding: utf-8 --

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm_notebook
from itertools import product
import warnings
import datetime

def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)
log = open('log(SARIMA)', 'w')
log_string(log, "loading data....")

font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18.,
}

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}

def metric(pred, label):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)

        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label.astype(np.float32))
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape

def read_data():
    file = 'data/ads.csv'
    df = pd.read_csv(file, index_col=['Time'], parse_dates=['Time'])
    ts = df.Ads
    return ts

def plot(ts):
    results = adfuller(ts)
    results_str = 'ADF test, p-value is: {}'.format(results[1])

    grid = plt.GridSpec(2, 2)
    ax1 = plt.subplot(grid[0, :])
    ax2 = plt.subplot(grid[1, 0])
    ax3 = plt.subplot(grid[1, 1])

    ax1.plot(ts)
    ax1.set_title(results_str)
    plot_acf(ts, lags=int(len(ts) / 2 - 1), ax=ax2)
    plot_pacf(ts, lags=int(len(ts) / 2 - 1), ax=ax3)
    plt.show()

# 1. read the original data from database and observe the results of visualization
data = pd.read_csv("train.csv")
filtered_data = data[data.iloc[:, 0] == 3]
y = filtered_data.iloc[:, 5]

# 2. visualization
print('Visual display, including stationarity test, autocorrelation, and partial autocorrelation plots')
# If p is less than 0.05, why no difference is needed?
# It means that the data is relatively stable, but seasonal difference is needed.
# plot(y.values[:2016])

from statsmodels.tsa.arima.model import ARIMA
def find_pq(ts, d=0, max_p=5, max_q=5):
    best_p, best_q = 0, 0
    best_aic = np.inf

    for p in range(max_p):
        for q in range(max_q):
            model = ARIMA(ts, order=(p, d, q)).fit()
            aic = model.aic

            if aic < best_aic:
                best_aic = aic
                best_p = p
                best_q = q

    return best_p, best_q, best_aic

from pmdarima.model_selection import train_test_split
def version_arima_with_manual(ts):
    """
    ARIMA（手动季节差分）
    """
    # period 周期大小
    periods = 12 * 24 * 7
    # seasonal difference季节差分
    ts_diff = ts - ts.shift(periods)
    # 再次差分（季节差分后p值小于0.05-接近，可认为平稳，若要严格一点也可再做一次差分）
    # ts_diff = ts_diff - ts_diff.shift(1)

    # （训练数据中不能有缺失值，这里差分后前几个值为nan，故去除）
    # ts_diff = ts_diff[~pd.isnull(ts_diff)]

    # data splitting 数据拆分
    train, test = train_test_split(ts_diff, train_size=0.8)
    test_list = test.values.tolist()
    label_list = ts.values.tolist()[train.shape[0]:]
    print(train.shape, test.shape)

    # model training 模型训练（训练数据为差分后的数据-已平稳，所以d=0）
    p, q, _ = find_pq(train)
    # p, q = 4, 3
    start_time = datetime.datetime.now()
    model = ARIMA(train.tolist(), order=(p, 0, q)).fit()
    end_time = datetime.datetime.now()
    total_time = end_time - start_time
    print("Total training times is : %f" % total_time.total_seconds())

    start_time = datetime.datetime.now()
    yhat = model.forecast(test.shape[0])
    end_time = datetime.datetime.now()
    total_time = end_time - start_time
    print("Total test times is : %f" % total_time.total_seconds())
    yhat = yhat.tolist()
    print(model.summary())

    # model predicting 模型预测
    # pre_s = []
    # label_s = []
    # for t in range(0, test.shape[0]-12, 12):
    #     try:
    #         pre_s += yhat[t:t+12]
    #         label_s += label_list[t:t+12]
    #     except:
    #         pass
    # print(np.array(pre_s).shape)
    # print(np.array(label_s).shape)
    history = train.values.tolist()
    pre_s = []
    label_s = []
    total = 0
    for t in range(0, test.shape[0]-12, 12):
        try:
            start_time = datetime.datetime.now()
            yhat = model.forecast(steps=12)
            pre_s += yhat.tolist()
            label_s += label_list[t:t+12]
            history = history[-500:] + test_list[t:t+12]
            model = ARIMA(history, order=(p, 0, q)).fit()
            end_time = datetime.datetime.now()
            total_time = end_time - start_time
            if total==0:
                total = total_time
            else:
                total+=total_time
        except:
            pass
    print(np.array(pre_s).shape)
    print("Total running times is : %f" % total.total_seconds())

    # forecasting results 差分还原（预测结果）
    prior = ts.values.tolist()[:train.shape[0]][-periods:]
    for i in range(0, len(pre_s), 12):
        for j in range(12):
            pre_s[i + j] = pre_s[i + j] + prior[j] # adding period to predicted values
        prior = prior[12:] + pre_s[i: i + 12] # last value adds into prior list, and removes first index from prior

    # model evaluation on three metrics 模型评估
    mae, rmse, mape = metric(pred = np.array(pre_s), label = np.array(label_s))
    log_string(log,'average:         %.3f\t\t%.3f\t\t%.3f%%' % (mae, rmse, mape * 100))

    # visualization 可视化
    plt.figure(figsize=(12, 4))
    plt.plot(label_s, label='Observed', color='black')
    plt.plot(pre_s, label='ARIMA', color='red')
    plt.ylabel("Traffic flow", font1)
    plt.title("Monitoring station 43", font1)
    plt.legend()
    plt.grid(True)
    plt.show()

    return mae, rmse, mape, np.reshape(pre_s, [-1, 12]), np.reshape(label_s,[-1, 12])  # [None, 12], [None, 12]


'''arima 实现过程'''
filtered_data = data[data.iloc[:, 0] == 43]
y = filtered_data.iloc[:, 5]
version_arima_with_manual(y)

def find_pq_PQ(ts, m, d, D, max_p=4, max_q=4, max_P=2, max_Q=2):
    best_p, best_q = 0, 0
    best_P, best_Q = 0, 0
    best_aic = np.inf

    for p in range(max_p):
        print('p is : ', p)
        for q in range(max_q):
            for P in range(max_P):
                for Q in range(max_Q):
                    model = SARIMAX(ts, order=(p, d, q), seasonal_order=(P, D, Q, m)).fit(disp=-1)
                    aic = model.aic

                    if aic < best_aic:
                        best_aic = aic
                        best_p = p
                        best_q = q
                        best_P = P
                        best_Q = Q

    return best_p, best_q, best_P, best_Q, best_aic

def version_sarima_with_manual(ts):
    """
    SARIMA（statsmodels）
    """
    # period 周期大小
    periods = 12 * 24 * 7

    # data splitting 数据拆分
    train, test = train_test_split(ts, train_size=0.8)
    test_list = test.values.tolist()

    # model training 模型训练
    d, D = 0, 1
    p, q, P, Q, _ = find_pq_PQ(ts, periods, d=d, D=D)
    # p, q, P, Q = p, q, P, Q = 4, 3, 0, [1] 
    model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, periods)).fit(disp=-1)
    yhat = model.forecast(test.shape[0])
    print(model.summary())

    pre_s = []
    label_s = []
    for t in range(0, test.shape[0]-12, 1):
        try:
            pre_s += yhat[t:t+12]
            label_s += test_list[t:t+12]
        except:
            pass
    print(np.array(pre_s).shape)

    # model evaluation 模型评估
    metric(pred = np.array(pre_s), label = np.array(label_s))

    # visualization 可视化
    plt.figure(figsize=(12, 4))
    plt.plot(label_s, label='Ads', color='blue')
    plt.plot(pre_s, label='forecast', color='red')
    plt.legend()
    plt.grid(True)
    plt.show()

'''sarima 实现过程'''
# version_sarima_with_manual(y)

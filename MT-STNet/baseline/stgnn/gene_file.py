# -- coding: utf-8 --
import pandas as pd
import numpy as np

data = pd.read_csv('train.csv')
data['datatime'] = pd.to_datetime(data.date) + pd.to_timedelta(data.hour, unit='h') + pd.to_timedelta(data.minute,
                                                                                                      unit='m')
# print(data.tail(100))
data_flow = data['flow'].values
datetime = data['datatime'].values
data_flow = data_flow.reshape(-1, 66,1)
datetime = datetime.reshape(-1, 66,1)
np.savez('train.npz', data=data_flow,index = datetime)
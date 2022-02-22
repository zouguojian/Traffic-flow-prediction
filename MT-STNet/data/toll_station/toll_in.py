# -- coding: utf-8 --
import pandas as pd
import csv
keys=['入口站点', '日期', '小时', '分钟', '车流量'] # keys
months=[-1,31,29,31,30,31,30,31,31,30,31,30,31] # -1 represents a sentinel
hours=24    # 24 h
minutes=60  # 60 minutes
toll_station='in1.5.csv' # address

data=pd.read_csv(filepath_or_buffer=toll_station,encoding='gb2312')

print(data.keys(),data.values[:10])
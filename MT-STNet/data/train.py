# -- coding: utf-8 --
# python 3.6
import pandas as pd
import csv
import os
import datetime

in_toll_stations={'2002':0, '2005':1, '2007':2, '2008':3, '2009':4, '2011':5, '2012':6,
                  '101001':7, '101007':8, '102004':9, '102005':10, '106006':11 ,'106007':12}
out_toll_stations={'2002':13, '2005':14, '2007':15, '2008':16, '2009':17, '2011':18, '2012':19,
                  '101001':20, '101007':21, '102004':22, '102005':23, '106006':24 ,'106007':25}
dragon_stations={'G008564001000310010':26, 'G008564001000320010':27,
                 'G008564001000210010':28, 'G008564001000220010':29,
                 'G002064001000410010':30, 'G002064001000420010':31,
                 'G002064001000320010':32, 'G002064001000310010':33,
                 'G002064001000210010':34, 'G002064001000220010':35,
                 'G000664001001610010':36, 'G000664001001620010':37,
                 'G000664001001510010':38, 'G000664001001520010':39,
                 'G000664001004110010':40, 'G000664001004120010':41,
                 'G000664001001410010':42, 'G000664001001420010':43,
                 'G000664001001310010':44, 'G000664001001320010':45,
                 'G000664001004010010':46, 'G000664001004020010':47,
                 'G000664001003910010':48, 'G000664001003920010':49,
                 'G000664001001210010':50, 'G000664001001220010':51,
                 'G000664001000910010':52, 'G000664001000920010':53,
                 'G000664001000820010':54, 'G000664001000810010':55,
                 'G000664001003810010':56, 'G000664001003820010':57,
                 'G000664001001720010':58, 'G000664001001710010':59,
                 'G000664001000720010':60, 'G000664001000710010':61,
                 'G008564001000410010':62, 'G008564001000420010':63,
                 'G002064001000520010':64, 'G002064001000510010':65}

keys=['station','date','hour','minute','flow'] # keys
months=[-1,31,29,31,30,31,30,31,31,30,31,30,31] # -1 represents a sentinel
hours=24    # 24 h
minutes=60  # 60 minutes

# data1=pd.read_csv('in1.5.csv',encoding='gb2312')
# print(data1.values[:10])
# print(data1.loc[(data1['日期']=='2021/6/1')])

# 思路，先取出每个站点的所有数据，然后，按照时间的顺序遍历每个站点的数据，累积相加即可！！！

def read_source(file_paths, beg_month=6,end_month=9,year=2021,encoding='utf-8'):
    '''
    :param file_paths: list:[file1, file2,...], that is all paths
    :param beg_month: begin month
    :param end_month: end month
    :param year:
    :param encoding: decoding methods
    :return:
    '''
    for in_toll_station in in_toll_stations:
        print('the in_toll_station name is: ',in_toll_station)
        in_toll_station_data_list=list()
        # used to store the DataFrame data of each station
        for in_file_path in file_paths:
            in_toll_station_data=pd.read_csv(filepath_or_buffer=in_file_path,encoding=encoding)
            # read each station data
            in_toll_station_data_list.append(in_toll_station_data.loc[(in_toll_station_data['入口站点'] == in_toll_station)])
            # use list to store each station data

        for month in range(beg_month,end_month):
            # to traverse the input months list
            for day in range(1, months[month]+1):
                # to traverse the input days of each month
                current_date=str(year)+'/'+str(month)+'/'+str(day)
                for hour in range(hours):
                    for minute in range(0, minutes, 5):
                        sum_flow=0
                        for data in in_toll_station_data_list: # read data form the DataFrom list
                            if not data.loc[(data['日期'] == current_date) & (data['小时'] == hour) & (data['分钟'] == minute)].empty:
                                sum_flow+=int(data.loc[(data['日期'] == current_date) & (data['小时'] == hour) & (data['分钟'] == minute)].values[-1][-1])
                        print(in_toll_station, current_date,hour,minute,sum_flow)
                        yield in_toll_station, current_date,hour,minute,sum_flow

def data_combine(file_paths, out_path, beg_month=6,end_month=9,year=2021,encoding='utf-8'):
    '''
    :param file_paths: a list, contain original data sets
    :param out_path: write path, used to save the training set
    :return:
    '''
    file = open(out_path, 'w', encoding='utf-8')
    writer = csv.writer(file)
    writer.writerow(['station','date','hour','minute','flow'])
    for line in read_source(file_paths=file_paths, beg_month=beg_month,end_month=end_month, year=year, encoding=encoding):
        writer.writerow(line)
    file.close()

if __name__=='__main__':
    print('hello')
    # data_combine(file_paths=['in1.5.csv', 'in1.csv', 'in2.csv', 'in3.csv'], out_path='in_flow.csv', encoding='gb2312')


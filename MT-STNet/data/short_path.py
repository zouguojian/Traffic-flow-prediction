# -- coding: utf-8 --

edge = {('in_101001', '78001D'): 0, ('in_101001', '780079'): 1, ('in_101001', '79001C'): 2, ('in_101007', '78001F'): 3,
               ('in_101007', '79001E'): 4, ('in_102004', '780067'): 5, ('in_102004', '78007D'): 6, ('in_102004', '79007C'): 7,
               ('in_102005', '78007F'): 8, ('in_102005', '79007E'): 9, ('in_106006', '780069'): 10, ('in_106006', '790068'): 11,
               ('in_106007', '78006B'): 12, ('in_106007', '79006A'): 13, ('in_2002', '780011'): 14, ('in_2002', '790010'): 15,
               ('in_2005', '780023'): 16, ('in_2005', '790022'): 17, ('in_2007', '78001B'): 18, ('in_2007', '79001A'): 19,
               ('in_2008', '780019'): 20, ('in_2008', '790014'): 21, ('in_2009', '780019'): 22, ('in_2009', '790014'): 23,
               ('in_2011', '78005F'): 24, ('in_2011', '79005E'): 25, ('in_2012', '780061'): 26, ('in_2012', '790060'): 27,
               ('78000F', '780011'): 28, ('780011', '780013'): 29, ('780011', '78005D'): 30, ('780013', '780019'): 31,
               ('780019', '78001B'): 32, ('78001B', '78001D'): 33, ('78001B', '780079'): 34, ('78001D', '78001F'): 35,
               ('78001F', '780021'): 36, ('780021', '780023'): 37, ('78005D', '78005F'): 38, ('78005F', '780061'): 39,
               ('780061', '780063'): 40, ('780061', '78007B'): 41, ('780061', '79007A'): 42, ('780063', '780021'): 43,
               ('780067', '780069'): 44, ('780069', '78006B'): 45, ('780079', '780063'): 46, ('780079', '78007B'): 47,
               ('780079', '790062'): 48, ('78007B', '780067'): 49, ('78007B', '78007D'): 50, ('78007D', '78007F'): 51,
               ('790012', '790010'): 52, ('790014', '790012'): 53, ('79001A', '790014'): 54, ('79001C', '79001A'): 55,
               ('79001E', '780079'): 56, ('79001E', '79001C'): 57, ('790020', '79001E'): 58, ('790022', '790020'): 59,
               ('790022', '790064'): 60, ('790024', '790022'): 61, ('79005E', '790012'): 62, ('790060', '79005E'): 63,
               ('790062', '790060'): 64, ('790064', '78007B'): 65, ('790064', '790062'): 66, ('790064', '79007A'): 67,
               ('790068', '78007D'): 68, ('790068', '79007C'): 69, ('79006A', '790068'): 70, ('79006C', '79006A'): 71,
               ('79007A', '78001D'): 72, ('79007A', '79001C'): 73, ('79007C', '780063'): 74, ('79007C', '790062'): 75,
               ('79007C', '79007A'): 76, ('79007E', '780067'): 77, ('79007E', '79007C'): 78, ('790080', '79007E'): 79,
               ('78000F', 'out_2002'): 80, ('780013', 'out_2008'): 81, ('780013', 'out_2009'): 82, ('780019', 'out_2007'): 83,
               ('78001B', 'out_101001'): 84, ('78001D', 'out_101007'): 85, ('780021', 'out_2005'): 86, ('78005D', 'out_2011'): 87,
               ('78005F', 'out_2012'): 88, ('780067', 'out_106006'): 89, ('780069', 'out_106007'): 90, ('78007B', 'out_102004'): 91,
               ('78007D', 'out_102005'): 92, ('790014', 'out_2002'): 93, ('79001A', 'out_2008'): 94, ('79001A', 'out_2009'): 95,
               ('79001C', 'out_2007'): 96, ('79001E', 'out_101001'): 97, ('790020', 'out_101007'): 98, ('790024', 'out_2005'): 99,
               ('790060', 'out_2011'): 100, ('790062', 'out_2012'): 101, ('790068', 'out_102004'): 102, ('79006A', 'out_106006'): 103,
               ('79006C', 'out_106007'): 104, ('79007A', 'out_101001'): 105, ('79007E', 'out_102004'): 106, ('790080', 'out_102005'): 107}

# Dijkstra.狄杰斯特拉
import heapq
import math
import numpy as np

def init_distance(graph, s):
    distance = {s: 0}
    for vertex in graph:
        if vertex != s:
            distance[vertex] = math.inf
    return distance

def distance_normalization(A):
    distances = A[~np.isinf(A)].flatten()
    std = distances.std()   # 计算距离的方差
    A = np.exp(-np.square(A / std))
    return A

def dijkstra(graph, s):
    pqueue = []
    heapq.heappush(pqueue, (0, s))
    seen = set()
    parent = {s: None}
    distance = init_distance(graph, s)

    while len(pqueue) > 0:
        pair = heapq.heappop(pqueue)
        dist = pair[0]
        vertex = pair[1]
        seen.add(s)
        nodes = graph[vertex].keys()
        for w in nodes:
            if w not in seen:
                if dist + graph[vertex][w] < distance[w]:
                    heapq.heappush(pqueue, (dist + graph[vertex][w], w))
                    parent[w] = vertex
                    distance[w] = dist + graph[vertex][w]
    return parent, distance

def distance_path(graph,s,end):
    parent, distance = dijkstra(graph, s)
    path=[end]
    while parent[end] !=None:
        path.append(parent[end])
        end=parent[end]
    path.reverse()
    return path

import pandas as pd
import csv
import numpy as np
if __name__ == '__main__':
    site_num=66
    roads_num = 108
    distance=pd.read_csv('YINCHUAN/distances.csv',usecols=['id1','id2','distance'], encoding='utf-8')
    distance['id1'] = distance['id1'].astype(str)
    distance['id2'] = distance['id2'].astype(str)
    distance['distance'] = distance['distance'].values #使用指数归一化方法
    distance_dict={(line[0], line[1]):line[2] for line in distance.values} # 两点之间直接距离

    # 站点映射到index，字典存放
    station_index = pd.read_csv('YINCHUAN/station_index.csv', usecols=['station', 'index'])
    station_index_dict={line[0]:line[1] for line in station_index.values}

    # 初始化站点间的直接距离，无连接的即为inf， 如：A到各个站点间的直接距离
    # graph_dict = {
    #     "A": {"B": 5, "C": 1},
    #     "B": {"A": 5, "C": 2, "D": 1},
    #     "C": {"A": 1, "B": 2, "D": 4, "E": 8},
    #     "D": {"B": 1, "C": 4, "E": 3, "F": 6},
    #     "E": {"C": 8, "D": 3},
    #     "F": {"D": 4},
    # }
    graph_dict = {station_i: {station_j: math.inf for station_j in station_index.values[:,0]} for station_i in station_index.values[:,0]}
    for key in distance_dict:
        graph_dict[key[0]][key[1]] = distance_dict[key]

    parent_dict, distance_dict = dijkstra(graph_dict, 'in_101001')
    print(parent_dict)
    print(distance_dict)
    print(distance_path(graph_dict,'in_101001','out_106007'))
    
    # 用于存储最短路径
    '''
    file = open('YINCHUAN/sp.csv', 'w', encoding='utf-8')
    writer = csv.writer(file)
    writer.writerow([i for i in range(site_num)])
    for station_index_i in station_index.values[:,0]:
        parent_dict, distance_dict = dijkstra(graph_dict, station_index_i)
        for station_index_j in station_index.values[:,0]:
            if distance_dict[station_index_j]<200:
                sp=distance_path(graph_dict, station_index_i, station_index_j)
                left_sp=[edge[(sp[i],sp[i+1])] for i in range(len(sp)-1)]   # 最短路径
                right_sp = [roads_num for _ in range(site_num-len(left_sp))]
                writer.writerow(left_sp+right_sp)
            else:
                writer.writerow([roads_num for _ in range(site_num)])
    file.close()
    '''


    # 用于存储最短路径长度
    '''
    sp_matrix=[[math.inf for _ in range(site_num)] for _ in range(site_num)]
    file = open('YINCHUAN/dis_A.csv', 'w', encoding='utf-8')
    writer = csv.writer(file)
    writer.writerow([i for i in range(site_num)])
    for station_i in station_index.values[:,0]:
        parent_dict, distance_dict = dijkstra(graph_dict, station_i)
        for key in distance_dict:
            sp_matrix[station_index_dict[station_i]][station_index_dict[key]]=distance_dict[key]
    sp_matrix = distance_normalization(np.array(sp_matrix))
    for i in range(site_num):
        writer.writerow(list(sp_matrix[i]))
    file.close()
    '''
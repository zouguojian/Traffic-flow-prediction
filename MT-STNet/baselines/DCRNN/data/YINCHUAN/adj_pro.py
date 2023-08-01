# -- coding: utf-8 --
import pandas as pd
import csv
import numpy as np

def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None): # 后续研究，建议使用
    if 'npy' in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        return adj_mx, None
    else:
        import csv
        A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
        if id_filename:
            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引
            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA
        else:
            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA


def triple_adjacent(adjacent_file=None, road_segments=66, target_file='adjacent.txt'):
    '''
    :return: [N*N, 3], for example [i, j, distance]
    '''
    adjacent=pd.read_csv(adjacent_file, encoding='utf-8').values

    triple_adjacent = np.zeros(shape=[road_segments**2, 3])
    full_adjacent = np.zeros(shape=[road_segments, road_segments])
    for i in range(road_segments):
        for j in range(road_segments):
            if i!=j:
                triple_adjacent[i * road_segments + j] = np.array([i, j, 0])
            else:
                full_adjacent[i, j] =1
                triple_adjacent[i * road_segments + j] = np.array([i, j, 1])

    for pair in adjacent:
        triple_adjacent[pair[0]*road_segments+pair[1]] = np.array([pair[0], pair[1], 1])
        full_adjacent[pair[0],pair[1]]=1

    np.savez('adjacent.npz', data=full_adjacent)
    np.savetxt(target_file, triple_adjacent, '%d')

def train_npz(source_file=None, site_num=108, target_file='train.npz'):
    '''
    :param source_file:
    :param site_num:
    :param target_file:
    :return:
    '''
    data = pd.read_csv(source_file, encoding='utf-8').values
    print(data.shape)
    data = np.reshape(data, [-1, site_num, data.shape[-1]])
    print(data.shape)
    np.savez(target_file, data=data)

if __name__=='__main__':
    print('hello')
    # 生成三元组形式的邻接矩阵
    triple_adjacent(adjacent_file='adjacent.csv',road_segments=66)

    # 训练集生成为.npz格式
    # train_npz(source_file='train.csv')

    print('finished')
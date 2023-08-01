import numpy as np
import os

# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

# metric
def metric(pred, label):
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape

def seq2instance(data, P, Q):
    num_step, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape = (num_sample, P, dims))
    y = np.zeros(shape = (num_sample, Q, dims))
    for i in range(num_sample):
        x[i] = data[i : i + P]
        y[i] = data[i + P : i + P + Q]
    return x, y

def load_dataset(args):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(args.traffic_file + args.name+'/', category + '.npz'), allow_pickle=True)
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
        print(data['x_' + category][...,0].shape, data['y_' + category][...,0].shape)
    mean=data['x_train'][..., 0].mean()
    std=data['x_train'][..., 0].std()
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = (data['x_' + category][..., 0]-mean)/std

    # spatial embedding
    # f = open(args.traffic_file+args.name+'/'+args.SE_file, mode = 'r')
    # lines = f.readlines()
    # temp = lines[0].split(' ')
    # N, dims = int(temp[0]), int(temp[1])
    # SE = np.zeros(shape = (N, dims), dtype = np.float32)
    # for line in lines[1 :]:
    #     temp = line.split(' ')
    #     index = int(temp[0])
    #     SE[index] = temp[1 :]

    # temporal embedding
    trainTE = None
    valTE = None
    testTE = None

    return data['x_train'][..., 0], trainTE, data['y_train'][..., 0], data['x_val'][..., 0], valTE, data['y_val'][..., 0], data['x_test'][..., 0], testTE, data['y_test'][..., 0], mean, std
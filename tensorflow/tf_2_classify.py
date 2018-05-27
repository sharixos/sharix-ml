import numpy as np 




data_path = '../data/'
if __name__ == '__main__':
    raw_input_data = np.loadtxt(data_path + 'test/2-classify.txt')
    x = raw_input_data[:, :2]
    y = raw_input_data[:, 2]



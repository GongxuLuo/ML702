#from lcy.CondGen.discriminate import test
import scipy.io as scio
import random
import numpy as np

random.seed(321)
data = scio.loadmat('BP.mat')

'''
data['fmri']: shape(82,82,97)
data['dti']: shape(82,82,97)
data['label']: shape(97,1)
'''

data['fmri'] = np.transpose(data['fmri'], [2,0,1])
data['dti'] = np.transpose(data['dti'], [2,0,1])

split_data = []
for i in range(len(data['fmri'])):
    split_data.append({
        'view1': data['fmri'][i:i+1],
        'view2': data['dti'][i:i+1],
        'label': data['label'][i:i+1],
        })
index = list(range(len(split_data)))
random.shuffle(index)
train_index = index[:int(len(index) * 0.8)]
test_index = index[int(len(index) * 0.8):]
print(test_index)
for item in data['fmri']:
    for i in range(len(item)):
        for j in range(len(item)):
            if item[i,j] < np.mean(item):
                item[i,j] = 0
            else:
                item[i,j] = 1

for item in data['dti']:
    for i in range(len(item)):
        for j in range(len(item)):
            if np.absolute(item[i,j]) < np.mean(np.absolute(item)):
                item[i,j] = 0
            else:
                item[i,j] = np.sign(item[i,j]) * 1


# train_index = range(int(data_length * 0.5))
# valid_index = range(int(data_length * 0.5), int(data_length * 0.6))
# test1_index = range(int(data_length * 0.6), int(data_length * 0.8))
# test2_index = range(int(data_length * 0.8), data_length)


def save_file(data_type: str, index):
    view1, view2, label = [], [], []
    for i in index:
        view1.append(split_data[i]['view1'])
        view2.append(split_data[i]['view2'])
        label.append(split_data[i]['label'])
    np.save(f'./view1_{data_type}', np.concatenate(view1, axis=0))
    np.save(f'./view2_{data_type}', np.concatenate(view2, axis=0))
    np.save(f'./label_{data_type}', np.concatenate(label, axis=0))


save_file('train', train_index)
save_file('test', test_index)


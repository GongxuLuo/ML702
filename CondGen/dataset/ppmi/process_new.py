#from lcy.CondGen.discriminate import test
import scipy.io as scio
import random
import numpy as np

random.seed(321)
data = scio.loadmat('PPMI.mat')

'''
data['X']: shape(90,90,70)
data['label']: shape(70,1)
'''

graphs = data['X']
labels = data['label']

split_data = []
for graph, label in zip(graphs, labels):
    graph = np.transpose(graph[0], [2,0,1])
    view1, view2 = graph[0:1], graph[2:3]
    mean1, mean2 = np.mean(view1), np.mean(view2)
    view1, view2 = np.where(view1 < mean1, 0, view1), np.where(view2 < mean2, 0, view2)
    split_data.append({
        'view1': view1,
        'view2': view2,
        'label': label,
        })

index = list(range(len(split_data)))
random.shuffle(index)
train_index = index[:int(len(index) * 0.8)]
test_index = index[int(len(index) * 0.8):]
print(test_index)

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

import numpy as np
import matplotlib.pyplot as plt
import sys
#plt.style.use('ggplot')
import matplotlib.style
import matplotlib as mpl
mpl.style.use('seaborn-whitegrid')

import sys

for line in sys.stdin:
    label = line.split(':')[0]
    data = list(map(float, line.split(':')[1].strip().split()))
    plt.plot(np.arange(len(data)), data, label=label)
plt.xticks(np.arange(0, len(data), 5), np.arange(1, len(data)+1, 5))
#plt.grid()
#plt.ylim(0.98, 1.0)
plt.xlabel('epoch')
plt.ylabel('AUC')
plt.legend()
plt.show()

# # data = np.load(sys.argv[1])
# #
# # steps = data['steps']
# # logits = data['detector_logits']
# #
# # for i in range(logits.shape[0]):
# #   plt.plot(steps, logits[:, i])
# #
# # plt.show()
# 
# 
# epoch_map = {'checkpoint-1': 0, 'checkpoint-10': 1, 'checkpoint-20': 2,
#              'checkpoint-30': 3, 'checkpoint-40': 4, 'checkpoint-50': 5,
#              'checkpoint-60': 6, 'checkpoint-70': 7, 'checkpoint-80': 8,
#              'checkpoint-90': 9, 'checkpoint-100': 10}
# 
# # config_map = {'steps-100 stepsize-0.01': 0, 'steps-200 stepsize-0.01': 1,'steps-500 stepsize-0.01': 2,
# #               'steps-1000 stepsize-0.001': 3, 'steps-2000 stepsize-0.001': 4, 'steps-5000 stepsize-0.001': 5}
# 
# config_map = {'steps-100 stepsize-0.01': 0, 'steps-500 stepsize-0.01': 1,
#               'steps-1000 stepsize-0.001': 2, 'steps-5000 stepsize-0.001': 3}
# 
# reverse_config_map = {v: k for k, v in config_map.items()}
# data = np.zeros((len(config_map), len(epoch_map)))  # num step configs, num checkpoints
# for line in open(sys.argv[1], 'r').readlines():
#   tokens = line.split()
#   val = float(tokens[-1])
#   j = epoch_map[tokens[1]]
#   key_i = tokens[2] + ' ' + tokens[3]
#   if key_i in config_map:
#     i = config_map[key_i]
#     data[i, j] = val
# 
# for i in range(data.shape[0]):
#   plt.plot(np.arange(data.shape[1]), data[i], label=reverse_config_map[i])
# 
# plt.xticks(np.arange(data.shape[1]), ['1', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100'])
# plt.grid()
# plt.xlabel('epoch')
# plt.ylabel('AUC')
# plt.legend()
# plt.show()
# 

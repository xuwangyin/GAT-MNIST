import sys
import matplotlib.pyplot as plt
#plt.style.use('seaborn')

from sklearn.metrics import roc_curve, auc
import numpy as np

#colors = [plt.cm.viridis(i) for i in np.linspace(0, 1.0, 10)]
colors = plt.cm.Paired.colors
pos = [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]
neg = [9020, 8865, 8968, 8990, 9018, 9108, 9042, 8972, 9026, 8991]

for i in range(10):
  data = np.load('adv_stats/test_adv_roc_class{}_steps5000_stepsize0.001.npz'.format(i))
  fpr, tpr = data['fpr'], data['tpr']
  roc_auc = np.round(auc(fpr, tpr), 5)
  plt.plot(fpr, tpr, color=colors[i], label='k = {} (AUC={:.5f})'.format(i, roc_auc, pos[i], neg[i]))
plt.xlim([0, 0.2])
plt.ylim([0.95, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('ROC of ')
plt.grid(True, alpha=0.3)
plt.legend(loc="lower right")
plt.savefig('mnist_roc_linf03_classes.pdf')
plt.show()

#sys.exit(0)

#colors = plt.cm.tab10.colors
for i in range(3):
  style=['-', '--', ':'][i]

  data = np.load('adv_stats/test_adv_roc_class{}_steps100_stepsize0.01.npz'.format(i))
  fpr, tpr = data['fpr'], data['tpr']
  roc_auc = np.round(auc(fpr, tpr), 5)
  plt.plot(fpr, tpr, style, color='C0', label='k={}, iters=100,   lr=0.01   (AUC = {})'.format(i, roc_auc))

  data = np.load('adv_stats/test_adv_roc_class{}_steps500_stepsize0.01.npz'.format(i))
  fpr, tpr = data['fpr'], data['tpr']
  roc_auc = np.round(auc(fpr, tpr), 5)
  plt.plot(fpr, tpr, style, color='C1', label='k={}, iters=500,   lr=0.01   (AUC = {})'.format(i, roc_auc))


  data = np.load('adv_stats/test_adv_roc_class{}_steps5000_stepsize0.001.npz'.format(i))
  fpr, tpr = data['fpr'], data['tpr']
  roc_auc = np.round(auc(fpr, tpr), 5)
  plt.plot(fpr, tpr, style, color='C2', label='k={}, iters=5000, lr=0.001 (AUC = {})'.format(i, roc_auc))


plt.xlim([0, 0.1])
plt.ylim([0.98, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('ROC of ')
plt.grid(True, alpha=0.3)
plt.legend(loc="lower right")
plt.savefig('mnist_roc_linf03_attacks.pdf')
plt.show()


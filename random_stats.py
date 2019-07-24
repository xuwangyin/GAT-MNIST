import numpy as np
from sklearn.metrics import roc_curve, auc


nat_logits = []
adv_logits = []
for i in range(50):
  #filename = 'random_test/epsilon=0.5-norm=Linf-optimizer=adam-steps=200-step_size=0.01-method=detector-adv-tag={0}-class0_samples_rand{0}.npz'.format(i)
  filename = 'random_test/epsilon=5.0-norm=L2-optimizer=adam-steps=200-step_size=0.1-method=detector-adv-tag={0}-class0_samples_rand{0}.npz'.format(i)
  print(filename)
  data = np.load(filename)
  logits = data['logits']
  y = data['y']
  nat_logits.append(logits[y==1])
  adv_logits.append(logits[y==0])
  fpr, tpr, th = roc_curve(y, logits)
  score = np.round(auc(fpr, tpr), 5)
  print('rand{}, score {}'.format(i, score))

nat_logits = nat_logits[0]
adv_logits = np.max(np.array(adv_logits), axis=0)
fpr, tpr, th = roc_curve(y, np.concatenate([nat_logits, adv_logits]))
score = np.round(auc(fpr, tpr), 5)
print('worst score {}'.format(score))

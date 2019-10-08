# Asymmetrical Adversarial Training on MNIST

## Training
Use train_classifier.py to train a standard classifier.

Use train_detector.py to train base detectors. As an example, the following kicks off the training of the first base detector.
    ```
    $ python train_detector.py --target_class 0 --epsilon 0.3 --norm Linf --train_steps 100 --step_size 0.01
    epoch 1, 0/50000| train auc 0.0000, f-score 0.1351, precision 0.1351, recall 0.1351, acc 0.0725, balanced_acc 0.0676 tpr 0.1351 fpr 1.0000, pos 37, neg 32
    epoch 1, 320/50000| train auc 0.7868, f-score 0.6800, precision 0.5152, recall 1.0000, acc 0.5152, balanced_acc 0.5000 tpr 1.0000 fpr 1.0000, pos 34, neg 32
    epoch 1, 640/50000| train auc 0.8672, f-score 0.6800, precision 0.5667, recall 0.8500, acc 0.6923, balanced_acc 0.7219 tpr 0.8500 fpr 0.4062, pos 20, neg 32
    ```
## Evaluation

First download and extract model checkpoints.

**Robustness test.** Use eval_base_detector.py to evaluate base detectors. As an example, the following tests the first eps0.3 base detector.
    ```
    $ python eval_base_detector.py \
    --target_class 0 --epsilon 0.3 --steps 200 --step_size 0.01 --optimizer adam \
    -p checkpoints/mnist/detector_Linf_0.3/ovr-steps100-adam-noclip-balanced/ovr_class0_Linf_distance0.3-91
    roc_auc: 0.9995914973528214
    ```
    
**Detection performance.** Use eval_detection.py to test the detection performances of integrated detection and generated detection.

**Robust classification performance.** Use eval_generative_classifier.py and eval_integrated_classifier.py to test the classification performances of generative classification and integrated classification.


## Model checkpoints

Pretrained models include Linf-eps0.3, Linf-eps0.5, L2-eps2.5 and L2-eps5.0 constrained detectors, a standard classifier, and a robust classifier.

Download and extract pretrained models:
```
 $ wget https://asymmetrical-adversarial-training.s3.amazonaws.com/mnist/checkpoints.tar.gz
 $ tar zxf checkpoints.tar.gz
```



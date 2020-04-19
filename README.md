# Generative Adversarial Training on MNIST

## Training
Use train_classifier.py to train a standard classifier.

Use train_detector.py to train base detectors. As an example, the following kicks off the training of the first base detector.
```
$ python train_detector.py --target_class 0 --epsilon 0.3 --norm Linf --train_steps 100 --step_size 0.01
```
## Evaluation

First download and extract model checkpoints.

**Robustness test.** Use eval_base_detector.py to evaluate base detectors. As an example, the following tests the first eps0.3 base detector.
```
$ python eval_base_detector.py \
--target_class 0 --epsilon 0.3 --steps 200 --step_size 0.01 --optimizer adam \
checkpoints/mnist/detector_Linf_0.3/ovr-steps100-adam-noclip-balanced/ovr_class0_Linf_distance0.3-91
```
    
**Detection performance.** Use eval_detection.py to test the detection performances of integrated detection and generated detection.

**Robust classification performance.** Use eval_generative_classifier.py and eval_integrated_classifier.py to test the classification performances of generative classification and integrated classification.

**Minimum mean L2 distance.** Use min_L2_perturb.py to reproduce the minimum mean L2 distance results.


## Model checkpoints

Pretrained models include Linf-eps0.3, Linf-eps0.5, L2-eps2.5 and L2-eps5.0 constrained detectors, a standard classifier, and a robust classifier.

Download and extract pretrained models:
```
 $ wget https://asymmetrical-adversarial-training.s3.amazonaws.com/mnist/checkpoints.tar.gz
 $ tar zxf checkpoints.tar.gz
```

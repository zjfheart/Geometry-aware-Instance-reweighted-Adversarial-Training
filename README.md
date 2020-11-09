# Geometry-aware Instance-reweighted Adversarial Training

This repository provides codes for geometry-aware instance-reweighted adversarial training methods,
based on the paper **Geometry-aware Instance-reweighted Adversarial Training** 
(https://arxiv.org/abs/2010.01736) <br/>
*Jingfeng Zhang, Jianing Zhu, Gang Niu, Bo Han, Masashi Sugiyama and Mohan Kankanhalli*

## What is the nature of adversarial training?
Adversarial training employs adversarial data for updating the models. 
For more details of the nature of adversarial training, refer to this [FAT's GitHub](https://github.com/zjfheart/Friendly-Adversarial-Training) for the preliminary. <br/> 
In this repo, you will know: 
## FACT 1: Model Capacity is NOT enough for adversarial training.
<p align="center">
    <img src="images/diff_net_error_white.png" width="400"\>
   <img src="images/eps_error_white.png" width="400"\>
</p>
<p align="left">
We plot standard training error (Natural) and adversarial training error (PGD-10) over the training epochs of the standard AT (Madry's) on CIFAR-10 dataset. 
  *Left panel*: AT on different sizes of network. 
  *Right panel*: AT on ResNet-18 under different perturbation bounds eps_train. </p>
  
  
Refer to [FAT's GitHub](https://github.com/zjfheart/Friendly-Adversarial-Training) for the standard AT by setting 

```python FAT.py --epsilon 0.031 --net 'resnet18' --tau 10 --dynamictau False``` 

OR using codes in this repo by setting

```python GAIRAT.py --epsilon 0.031 --net 'resnet18' --Lambda_max float('inf')``` 

to recover the standard AT (Madry's).


The over-parameterized models that fit nataral data entirely in the standard training are still far from enough for fitting adversarial data in adversarial training.
Compared with standard training fitting the natural data points, adversarial training smooths the neighborhoods of natural data, so that adversarial data consume significantly more model capacity than natural data. 

The volume of this neighborhood is exponentially ![](http://latex.codecogs.com/gif.latex?|1+\epsilon_{train}|^{X}) large w.r.t. the input dimension ![](http://latex.codecogs.com/gif.latex?|X|), even if ![](http://latex.codecogs.com/gif.latex?\epsilon_{train}) is small.

Under the computational budget of 100 epochs, the networks hardly reach zero error on the adversarial training data.


## FACT 2: Data points are inherently different. 
More attackable data are closer to the decision boundary.

More guarded data are farther away from the decision boundary.

<p align="center">
    <img src="images/motivation.png" width="400"\>
    <img src="images/pca01.png" width="350"\>
</p>
<p align="left">
More attackable data (lighter red and blue) are closer to the decision boundary; more guarded data (darker red and blue) are farther away from the decision boundary. *Left panel*: Two toy examples. *Right panel*: The model’s output distribution of two randomly selected classes from the CIFAR-10 dataset. The degree of robustness (denoted by the color gradient) of a data point is calculated based on the least number of iterations κ that PGD needs to find its misclassified adversarial variant. </p>


## Therefore, given the limited model capacity, we should treat data differently for updating the model in adversarial training.
**IDEA**: Geometrically speaking, a natural data point closer to/farther from the class boundary is less/more robust, and the corresponding adversarial data point should be assigned with larger/smaller weight.<br/>
To implement the idea, we propose geometry-aware instance-reweighted adversarial training (GAIRAT), where the weights are based on how difficult it is to attack a natural data point.<br/>
"how difficult it is to attack a natural data point" is approximated by the number of PGD steps that the PGD method requires to generate its misclassified adversarial variant.
<p align="center">
    <img src="images/GAIRAT_learning_obj.png" width="800"\>
</p>
<p align="left">
The illustration of GAIRAT. GAIRAT explicitly gives larger weights on the losses of adversarial data (larger red), whose natural counterparts are closer to the decision boundary (lighter blue). GAIRAT explicitly gives smaller weights on the losses of adversarial data (smaller red), whose natural counterparts are farther away from the decision boundary (darker blue). </p>

## GAIRAT's Implementation
For updating the model, GAIRAT assigns instance dependent weight on the loss of the adversarial data (found in ```GAIRAT.py```). <br/>
The instance dependent weight depends on ```num_steps```, which indicates the least PGD step numbers for the misclassified adversarial variant. <br/>

The details will be determined once the name are fixed.........

## Preferred Prerequisites

* Python (3.6)
* Pytorch (1.2.0)
* CUDA
* numpy

## Running GAIRAT, GAIR-FAT on benchmark datasets  (CIFAR-10 and SVHN)


### White-box evaluations on WRN-32-10
    

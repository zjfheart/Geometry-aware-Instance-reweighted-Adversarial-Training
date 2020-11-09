# Geometry-aware Instance-reweighted Adversarial Training

This repository provides codes for geometry-aware instance-reweighted adversarial training methods,
based on the paper **Geometry-aware Instance-reweighted Adversarial Training** 
(https://arxiv.org/abs/2010.01736) <br/>
*Jingfeng Zhang, Jianing Zhu, Gang Niu, Bo Han, Masashi Sugiyama and Mohan Kankanhalli*

## What is the nature of adversarial training?
Adversarial training employs adversarial data for updating the models. 
For more details of the nature of adversarial training, refer to this [FAT's GitHub](https://github.com/zjfheart/Friendly-Adversarial-Training) for the preliminary. <br/> 
In this repo, you will know: 
## Model Capacity is NOT enough for adversarial training.
<p align="center">
    <img src="images/diff_net_error_white.png" width="450"\>
   <img src="images/eps_error_white.png" width="450"\>
</p>
<p align="left">
We plot standard training error (Natural) and adversarial training error (PGD-10) over the training epochs of the standard AT (Madry's) on CIFAR-10 dataset. 
  Left panel: AT on different sizes of network. 
  Right panel: AT on ResNet-18 under different perturbation bounds eps_train. </p>
  
  
Refer to [FAT's GitHub](https://github.com/zjfheart/Friendly-Adversarial-Training) for the standard AT by setting 

```python FAT.py --epsilon 0.031 --net 'resnet18' --tau 10 --dynamictau False``` 

OR using codes in this repo by setting

```python GAIRAT --epsilon 0.031 --net 'resnet18' --beta_max =????``` 

to recover the standard AT (Madry's).


The over-parameterized models that fit nataral data entirely in the standard training are still far from enough for fitting adversarial data in adversarial training.
Compared with standard training fitting the natural data points, adversarial training smooths the neighborhoods of natural data, so that adversarial data consume significantly more model capacity than natural data. 

The volume of this neighborhood is exponentially ![](http://latex.codecogs.com/gif.latex?|1+\epsilon_{train}|^{X}) large w.r.t. the input dimension ![](http://latex.codecogs.com/gif.latex?|X|), even if ![](http://latex.codecogs.com/gif.latex?\epsilon_{train}) is small.

Under the computational budget of 100 epochs, the networks hardly reach zero error on the adversarial training data.


## Data are inherently different. 
More attackable/guarded data are closer to/farther away from the decision boundary.

## Given the limited model capacity, we should treat adversarial data differently for updating the model. 

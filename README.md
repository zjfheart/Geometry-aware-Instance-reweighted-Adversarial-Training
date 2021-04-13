# Geometry-aware Instance-reweighted Adversarial Training

This repository provides codes for **Geometry-aware Instance-reweighted Adversarial Training** 
(https://openreview.net/forum?id=iAX0l6Cz8ub) (ICLR oral)<br/>
*Jingfeng Zhang, Jianing Zhu, Gang Niu, Bo Han, Masashi Sugiyama and Mohan Kankanhalli*

## What is the nature of adversarial training?
Adversarial training employs adversarial data for updating the models. 
For more details of the nature of adversarial training, refer to this [FAT's GitHub](https://github.com/zjfheart/Friendly-Adversarial-Training) for the preliminary. <br/> 
In this repo, you will know: 
## FACT 1: Model Capacity is NOT enough for adversarial training.
<p align="center">
    <img src="images/diff_net_error_with_ST_white.png" width="400"\>
   <img src="images/eps_error_white.png" width="400"\>
</p>
<p align="left">
We plot standard training error (Natural) and adversarial training error (PGD-10) over the training epochs of the standard AT (Madry's) on CIFAR-10 dataset. 
  *Left panel*: AT on different sizes of network (blue lines) and standard training (ST, the red line) on ResNet-18. 
  *Right panel*: AT on ResNet-18 under different perturbation bounds eps_train. </p>
  
  
Refer to [FAT's GitHub](https://github.com/zjfheart/Friendly-Adversarial-Training) for the standard AT by setting 

```python FAT.py --epsilon 0.031 --net 'resnet18' --tau 10 --dynamictau False``` 

OR using codes in this repo by setting

```python GAIRAT.py --epsilon 0.031 --net 'resnet18' --Lambda 'inf'``` 

to recover the standard AT (Madry's).


The over-parameterized models that fit nataral data entirely in the standard training (ST) are still far from enough for fitting adversarial data in adversarial training (AT).
Compared with ST fitting the natural data points, AT smooths the neighborhoods of natural data, so that adversarial data consume significantly more model capacity than natural data. 

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
**IDEA**: Geometrically speaking, a natural data point closer to/farther from the class boundary is less/more robust, and the corresponding adversarial data point should be assigned with larger/smaller weight for updating the model.<br/>
To implement the idea, we propose geometry-aware instance-reweighted adversarial training (GAIRAT), where the weights are based on how difficult it is to attack a natural data point.<br/>
"how difficult it is to attack a natural data point" is approximated by the number of PGD steps that the PGD method requires to generate its misclassified adversarial variant.
<p align="center">
    <img src="images/GAIRAT_learning_obj.png" width="800"\>
</p>
<p align="left">
The illustration of GAIRAT. GAIRAT explicitly gives larger weights on the losses of adversarial data (larger red), whose natural counterparts are closer to the decision boundary (lighter blue). GAIRAT explicitly gives smaller weights on the losses of adversarial data (smaller red), whose natural counterparts are farther away from the decision boundary (darker blue). </p>

## GAIRAT's Implementation
For updating the model, GAIRAT assigns instance dependent weight (```reweight```) on the loss of the adversarial data (found in ```GAIR.py```). <br/>
The instance dependent weight depends on ```num_steps```, which indicates the least PGD step numbers needed for the misclassified adversarial variant. <br/>



## Preferred Prerequisites

* Python (3.6)
* Pytorch (1.2.0)
* CUDA
* numpy
* foolbox

## Running GAIRAT, GAIR-FAT on benchmark datasets  (CIFAR-10 and SVHN)

Here are examples:

* Train GAIRAT and GAIR-FAT on WRN-32-10 model on CIFAR-10 and compare our results with [AT](https://arxiv.org/abs/1706.06083), [FAT](https://arxiv.org/abs/2002.11242)
```bash
CUDA_VISIBLE_DEVICES='0' python GAIRAT.py 
CUDA_VISIBLE_DEVICES='0' python GAIR_FAT.py 
```
* How to recover the original FAT and AT using our code? 

```bash
CUDA_VISIBLE_DEVICES='0' python GAIRAT.py --Lambda 'inf' --output_dir './AT_results' 
CUDA_VISIBLE_DEVICES='0' python GAIR_FAT.py --Lambda 'inf' --output_dir './FAT_results' 
```

* Evaluations
After running, you can find ```./GAIRAT_result/log_results.txt``` and ```./GAIR_FAT_result/log_results.txt``` for checking Natural Acc. and PGD-20 test Acc. <br/>
We also evaluate our models using PGD+. PGD+ is the same as ``PG_ours`` in [RST repo](https://github.com/yaircarmon/semisup-adv). ``PG_ours`` is PGD with 5 random starts, and each start has 40 steps with step size 0.01 (It has 40 × 5 = **200** iterations for each test data).
Since PGD+ is computational defense, we only evaluate the best checkpoint ```bestpoint.pth.tar``` and the last checkpoint ```checkpoint.pth.tar``` in the folders ```GAIRAT_result``` and ```GAIR_FAT_result``` respectively. 
```bash
CUDA_VISIBLE_DEVICES='0' python eval_PGD_plus.py --model_path './GAIRAT_result/bestpoint.pth.tar' --output_suffix='./GAIRAT_PGD_plus'
CUDA_VISIBLE_DEVICES='0' python eval_PGD_plus.py --model_path './GAIR_FAT_result/bestpoint.pth.tar' --output_suffix='./GAIR_FAT_PGD_plus'
```

### White-box evaluations on WRN-32-10

 Defense （best checkpoint）        	| Natural Acc. 	| PGD-20 Acc. | PGD+ Acc. | 
|-----------------------|-----------------------|------------------|-----------------|
|[AT(Madry)](https://arxiv.org/abs/1706.06083)		| 86.92 % ![](https://latex.codecogs.com/gif.latex?\pm) 0.24%	|  51.96% ![](https://latex.codecogs.com/gif.latex?\pm) 0.21%	|    51.28%	![](https://latex.codecogs.com/gif.latex?\pm) 0.23%    |
| [FAT](https://arxiv.org/abs/2002.11242)  		|  **89.16% ![](https://latex.codecogs.com/gif.latex?\pm) 0.15%**  	|     51.24% ![](https://latex.codecogs.com/gif.latex?\pm) 0.14%     |     46.14% ![](https://latex.codecogs.com/gif.latex?\pm) 0.19%     |
| GAIRAT  |  85.75% ![](https://latex.codecogs.com/gif.latex?\pm) 0.23%  	|**57.81% ![](https://latex.codecogs.com/gif.latex?\pm) 0.54%**| **55.61% ![](https://latex.codecogs.com/gif.latex?\pm) 0.61%**|
| GAIR-FAT		|  88.59% ![](https://latex.codecogs.com/gif.latex?\pm) 0.12%   	|   56.21% ![](https://latex.codecogs.com/gif.latex?\pm) 0.52%  		|     53.50% ![](https://latex.codecogs.com/gif.latex?\pm) 0.60%    	|

 Defense （last checkpoint）        	| Natural Acc. 	| PGD-20 Acc. | PGD+ Acc. | 
|-----------------------|-----------------------|------------------|-----------------|
|[AT(Madry)](https://arxiv.org/abs/1706.06083)		| 86.62 % ![](https://latex.codecogs.com/gif.latex?\pm) 0.22%	|  46.73% ![](https://latex.codecogs.com/gif.latex?\pm) 0.08%	|    46.08%	![](https://latex.codecogs.com/gif.latex?\pm) 0.07%    |
| [FAT](https://arxiv.org/abs/2002.11242)  		|  88.18% ![](https://latex.codecogs.com/gif.latex?\pm) 0.19%  	|     46.79% ![](https://latex.codecogs.com/gif.latex?\pm) 0.34%     |     45.80% ![](https://latex.codecogs.com/gif.latex?\pm) 0.16%     |
| GAIRAT  |  85.49% ![](https://latex.codecogs.com/gif.latex?\pm) 0.25%  	|**53.76% ![](https://latex.codecogs.com/gif.latex?\pm) 0.49%**| **50.32% ![](https://latex.codecogs.com/gif.latex?\pm) 0.48%**|
| GAIR-FAT		|  **88.44% ![](https://latex.codecogs.com/gif.latex?\pm) 0.10%**   	|   50.64% ![](https://latex.codecogs.com/gif.latex?\pm) 0.56%  		|     47.51% ![](https://latex.codecogs.com/gif.latex?\pm) 0.51%    	|

For more details, refer to Table 1  and Appendix C.8 in the paper. 

## Benchmarking robustness with additional 500K unlabeled data on CIFAR-10 dataset.

In this repo, we unleash the full power of our geometry-aware instance-reweighted methods by incorporating 500K unlabeled data (i.e., **GAIR-RST**). 
In terms of both evaluation metrics, i.e., generalization and robustness, we can obtain the best WRN-28-10 model among all public available robust models. <br/>

* How to create the such the superior model from scratch? 
1. Download ```ti_500K_pseudo_labeled.pickle``` containing our 500K pseudo-labeled TinyImages from this [link](https://drive.google.com/file/d/1LTw3Sb5QoiCCN-6Y5PEKkq9C9W60w-Hi/view) (Auxillary data provided by Carmon et al. 2019). Store ```ti_500K_pseudo_labeled.pickle``` into the folder ```./data```<br/>
2. You may need mutilple GPUs for running this. 
```bash
chmod +x ./GAIR_RST/run_training.sh
./GAIR_RST/run_training.sh
```
3. We evaluate the robust model using natural test accuracy on natural test data and roubust test accuracy by [Auto Attack](https://github.com/fra31/auto-attack). 
Auto Attack is combination of two white box attacks and two black box attacks. 
```bash
chmod +x ./GAIR_RST/autoattack/examples/run_eval.sh
```

### White-box evaluations on WRN-28-10

We evaluate the robustness on CIFAR-10 dataset under [auto-attack](https://github.com/fra31/auto-attack) [(Croce & Hein, 2020)](https://arxiv.org/abs/2003.01690). 

Here we list the results using WRN-28-10 on the [leadboard](https://github.com/fra31/auto-attack/blob/master/README.md) and our results. In particular, we use the test `eps = 0.031` which keeps the same as the training `eps` of our GAIR-RST.


## CIFAR-10 - Linf
The robust accuracy is evaluated at `eps = 8/255`, except for those marked with * for which `eps = 0.031`, where `eps` is the maximal Linf-norm allowed for the adversarial perturbations. The `eps` used is the same set in the original papers.\
**Note**: ‡ indicates models which exploit additional data for training (e.g. unlabeled data, pre-training).

|#    |method/paper           |model     |architecture |clean         |report. |AA  |
|:---:|---|:---:|:---:|---:|---:|---:|
|**1**| [(Gowal et al., 2020)](https://arxiv.org/abs/2010.03593)‡| *authors*| WRN-28-10| 89.48| 62.76| 62.80|
|**2**| [(Wu et al., 2020b)](https://arxiv.org/abs/2004.05884)‡| *available*| WRN-28-10| 88.25| 60.04| 60.04|
|**-**| **[GAIR-RST (Ours)\*](https://arxiv.org/abs/2010.01736)**‡| *available*| WRN-28-10| 89.36| 59.64| 59.64|
|**3**| [(Carmon et al., 2019)](https://arxiv.org/abs/1905.13736)‡| *available*| WRN-28-10| 89.69| 62.5| 59.53|
|**4**| [(Sehwag et al., 2020)](https://github.com/fra31/auto-attack/issues/7)‡| *available*| WRN-28-10| 88.98| -| 57.14|
|**5**| [(Wang et al., 2020)](https://openreview.net/forum?id=rklOg6EFwS)‡| *available*| WRN-28-10| 87.50| 65.04| 56.29|
|**6**| [(Hendrycks et al., 2019)](https://arxiv.org/abs/1901.09960)‡| *available*| WRN-28-10| 87.11| 57.4| 54.92|
|**7**| [(Moosavi-Dezfooli et al., 2019)](http://openaccess.thecvf.com/content_CVPR_2019/html/Moosavi-Dezfooli_Robustness_via_Curvature_Regularization_and_Vice_Versa_CVPR_2019_paper)| *authors*| WRN-28-10| 83.11| 41.4| 38.50|
|**8**| [(Zhang & Wang, 2019)](http://papers.nips.cc/paper/8459-defense-against-adversarial-attacks-using-feature-scattering-based-adversarial-training)| *available*| WRN-28-10| 89.98| 60.6| 36.64|
|**9**| [(Zhang & Xu, 2020)](https://openreview.net/forum?id=Syejj0NYvr&noteId=Syejj0NYvr)| *available*| WRN-28-10| 90.25| 68.7| 36.45|

The results show our GAIR-RST method can facilitate a competitive model by utilizing additional unlabeled data.



### Wanna download our superior model for other purposes? Sure! 

We welcome various attack methods to attack our defense models. For cifar-10 dataset, we normalize all images into ```[0,1]```. <br/>

Download our pretrained models ```checkpoint-epoch200.pt``` into the folder ``./GAIR_RST/GAIR_RST_results`` through this [Google Drive link](https://drive.google.com/drive/folders/1Ry7q_NbCgeJsjSwxXpRfi1zSc_jdVJf6?usp=sharing).

You can evaluate this pretrained model through ```./GAIR_RST/autoattack/examples/run_eval.sh```

## Reference
```
@inproceedings{
zhang2021_GAIRAT,
title={Geometry-aware Instance-reweighted Adversarial Training},
author={Jingfeng Zhang and Jianing Zhu and Gang Niu and Bo Han and Masashi Sugiyama and Mohan Kankanhalli},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=iAX0l6Cz8ub}
}
```

# Contact
Please contact jingfeng.zhang@riken.jp or j-zhang@comp.nus.edu.sg and zhujianing9810@gmail.com if you have any question on the codes.


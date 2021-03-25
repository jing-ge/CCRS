# Corporate Credit Rating System

## 1.Introduction
This is a corporate credit system based on deep learning models. The whole model concludes three moudules including Financial Data,Credit Rating models and Ensemble model  which shows in the following figure.
![avatar](./images/architecture.jpg)
* Financial Data : This layer process the financial data as input.
* Credit Rating models: This moudules contains three rating models:CCR-CNN,CCR-GNN,ASSL4CCR
* Ensemble model: This layer enselbe the three models to predict final results by the way of bagging.


## 2.Credit Rating models
* **CCR-CNN** :  corporate credit ratings via convolution neural networks
![avatar](./images/ccrcnn.jpg)
[[pdf]](https://arxiv.org/abs/2012.03744)
* **CCR-GNN** : corporate credit ratings via Graph neural networks
![avatar](./images/ccrgnn.jpg)
[[pdf]](https://arxiv.org/abs/2012.01933)
* **ASSL4CCR** : Adversarial semi-supervised learning for corporate credit rating 
![avatar](./images/assl4ccr.jpg)
[[pdf]]()
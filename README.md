
# On Learning Contrastive Representations for Learning With Noisy Labels

This repository contains the implementation code for paper: <br>
__On Learning Contrastive Representations for Learning With Noisy Labels__ <br>
Li Yi, Sheng Liu, Qi She, A. Ian McLeod, Boyu Wang <br>
_IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR 2022)_ <br>
[[Paper](https://arxiv.org/pdf/2203.01785.pdf)]

## Running Experiments
Please refer to [[run.sh](run.sh)] for running different experiments. 
This repository implements our method with [BYOL](https://arxiv.org/pdf/2006.07733.pdf) framework for better performance as indicated in our paper (Tab.5).


## Results
Here we show the accuracy of our method + **CE** on CIFAR-10 in 3 trails:

|   | 0%  | SYM 20% | SYM 40% | SYM 60% | SYM 80% | SYM 90% | ASYM 40%
|-----------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Seed 2021 (Best/Last)    |  94.40/94.31   |  93.19/93.19   |  92.14/91.91   |  89.18/89.08   |  88.00/87.99   | 84.58/84.49 |  89.23/89.23 |
| Seed 2022 (Best/Last)  |  94.22/94.15   |  93.54/93.29   |  92.36/92.22   |  89.46/88.38   |  87.56/87.39   | 83.48/83.12 |  89.43/88.16|
| Seed 2023 (Best/Last)  |  94.36/94.24   |  93.22/93.09   |  92.16/92.08   |  88.87/88.61   |  87.10/86.87   | 84.32/84.10 | 89.71/89.60|
| Average  |  94.32/94.23   |  93.31/93.19   |  92.21/92.07   | 89.17/88.69   |  87.55/87.41   | 84.12/83.90 | 89.45/89.00|


Here we show the accuracy of our method + **CE** on CIFAR-100 in 3 trails:

|   | 0%  | SYM 20% | SYM 40% | SYM 60% | SYM 80%| ASYM 40%
|-----------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Seed 2021 (Best/Last)    |  75.75/75.54   |  71.80/71.52   |  69.21/69.06   |  62.95/62.70   |  55.07/54.94   | 55.76/54.81 | 
| Seed 2022 (Best/Last)  |  76.03/75.93   |  71.75/71.50  |    68.09/67.97   |  62.80/62.33   | 56.04/55.93 | 54.57/54.10 |
| Seed 2023 (Best/Last)  |  76.35/76.14   |  71.84/71.63     |  67.97/67.72   |  63.43/63.18   | 54.83/54.42 | 55.46/54.68 |
| Average  |  76.04/75.87   |  71.79/71.55   |  68.42/68.25   |  63.06/62.73   |  55.31/55.09   | 54.93/54.49| 


Here we show the accuracy of our method + **GCE** on CIFAR-10 in 3 trails:

|    | SYM 20% | SYM 40% | SYM 60% | SYM 80% | SYM 90% | 
|-----------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Seed 2021 (Best/Last)    |  94.40/94.36   |  93.85/93.76   |  93.09/92.96   |  91.81/91.63   |   89.29/88.92   |  
| Seed 2022 (Best/Last)  |  94.51/94.38   |  93.49/93.44   |  93.15/92.98   |  91.20/91.03   |  89.79/89.65   |  
| Seed 2023 (Best/Last)  |  94.34/94.25   |  93.70/93.36   |  93.20/93.14   |  92.00/91.81   |  90.44/90.34   | 
| Average  |  94.41/94.33   |  93.67/93.52   |  93.14/93.02   |  91.67/91.49   |  89.83/87.63   |


Here we show the accuracy of our method + **GCE** on CIFAR-100 in 3 trails:

|    | SYM 20% | SYM 40% | SYM 60% | SYM 80%
|-----------|:-------:|:-------:|:-------:|:-------:|:-------:|
| Seed 2021 (Best/Last)    |  74.26/73.90   |  72.65/72.37   |  70.66/70.29   |   64.59/64.38   | 
| Seed 2022 (Best/Last)  |  74.67/74.55   |  73.12/72.90  |    70.47/70.13   |   64.05/63.81   | 
| Seed 2023 (Best/Last)  |  74.66/74.41   |  72.85/72.71     |  70.97/70.70   |  63.47/63.29   | 
| Average  |  74.53/74.28   |  72.87/72.66   | 70.70/70.46   |  64.03/63.82   | 


## Acknowledgement
This code inherits some codes from [SimSiam](https://github.com/Reza-Safdari/SimSiam-91.9-top1-acc-on-CIFAR10).

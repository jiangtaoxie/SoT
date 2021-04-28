# So-ViT: Mind Visual Tokens for Vision Transformer


<div>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src="https://jiangtaoxie.oss-cn-beijing.aliyuncs.com/overview.jpg?versionId=CAEQARiBgICg0OPxxxciIDVlNTJiZWJhYjJmZjRmZDQ5ZDM0OTU2ZWY1MDg3OGQy" width="100%"/>
</div>

## Introduction

This repository contains the source code under **PyTorch** framework and models trained on ImageNet-1K dataset for the following [paper](https://arxiv.org/pdf/2104.10935.pdf):

    @articles{So-ViT,
        author = {Jiangtao Xie, Ruiren Zeng, Qilong Wang, Ziqi Zhou, Peihua Li},
        title = {So-ViT: Mind Visual Tokens for Vision Transformer},
        booktitle = {arXiv:2104.10935},
        year = {2021}
    }

The Vision Transformer ([ViT](https://arxiv.org/pdf/2010.11929.pdf)) heavily depends on pretraining using ultra large-scale datasets (e.g. ImageNet-21K or JFT-300M) to achieve high performance, while significantly underperforming on *ImageNet-1K if trained from scratch*. We propose a novel So-ViT model toward addressing this problem, by carefully considering the *role of visual tokens*. 


***Above all***, for classification head, the ViT only exploits class token while entirely neglecting rich semantic information inherent in high-level visual tokens. Therefore, we propose a new   classification paradigm, where the second-order, cross-covariance pooling of visual tokens is combined with class token for final classification. Meanwhile, a fast singular value power normalization is proposed for improving the second-order pooling. 

  
***Second***, the ViT employs the naïve method of one linear projection of fixed-size image patches for visual token embedding, lacking the ability to model translation equivariance and locality. To alleviate this problem, we develop a light-weight, hierarchical module based on off-the-shelf convolutions for visual token embedding. 

## Classification results

#### Classification results (single crop 224x224, %) on ImageNet-1K validation set
 <table>
         <tr>
             <th rowspan="2" style="text-align:center;">Network</th>
             <th colspan="2" style="text-align:center;">Top-1 Accuracy </th>
             <th colspan="2" style="text-align:center;">Pre-trained models</th>
         </tr>
         <tr>
         <td style="text-align:center;">Paper reported</td>
             <td style="text-align:center;"> Upgrade </td>
             <td style="text-align:center;">GoogleDrive</td>
             <td style="text-align:center;">Aliyun</td>
         </tr>
         <tr>
             <td style="text-align:center">So-ViT-7</td>
             <td style="text-align:center;">76.2</td>
             <td style="text-align:center;">76.8</td>
             <td style="text-align:center;"><a href="https://drive.google.com/file/d/1-MTuAbQ-T6skfqHEYrhb9BmitG1iu5SY/view?usp=sharing">85.08MB</a></td>
             <td style="text-align:center;"><a href="https://jiangtaoxie.oss-cn-beijing.aliyuncs.com/So-ViT/So-ViT-7.pth.tar?versionId=CAEQAhiBgIDLutbayBciIGI0ZDEwN2ZiZjk3MjQxMjY5ODQxMTU3OGYxZjYzZWYz">85.08MB</a></td>
         </tr>
         <tr>
             <td style="text-align:center">So-ViT-10</td>
             <td style="text-align:center;">77.9</td>
             <td style="text-align:center;">78.7</td>
             <td style="text-align:center;"><a href="https://drive.google.com/file/d/1biRaP_m8NR48buFEQQwCRQphR_Nz5Sjw/view?usp=sharing">109.24MB</a></td>
             <td style="text-align:center;"><a href="https://jiangtaoxie.oss-cn-beijing.aliyuncs.com/So-ViT/So-ViT-10.pth.tar?versionId=CAEQAhiBgMDHutbayBciIDViOTY5ZDFlNTI0ZDQ5NWNhM2IxMTA3OWU4OTlmMmUw">109.24MB</a></td>
         </tr>
         <tr>
             <td style="text-align:center">So-ViT-14</td>
             <td style="text-align:center;">81.8</td>
             <td style="text-align:center;">82.3</td>
             <td style="text-align:center;"><a href="https://drive.google.com/file/d/1A85Rlnaaya9GEpraaEKJsSaGsdqHU5YO/view?usp=sharing">377.09MB</a></td>
             <td style="text-align:center;"><a href="https://jiangtaoxie.oss-cn-beijing.aliyuncs.com/So-ViT/So-ViT-14.pth.tar?versionId=CAEQAhiBgIDDutbayBciIGM1YjBlOTIxMjYyNDRhMjY4MjE1MzgxMDcyMjEyNTM1">377.09MB</a></td>
         </tr>
         <tr>
             <td style="text-align:center">So-ViT-19</td>
             <td style="text-align:center;">82.4</td>
             <td style="text-align:center;">82.8</td>
             <td style="text-align:center;"><a href="https://drive.google.com/file/d/1Jv2lYVzJ5lY909MHYP3YORhAVsKxUAC9/view?usp=sharing">669.60MB</a></td>
             <td style="text-align:center;"><a href="https://jiangtaoxie.oss-cn-beijing.aliyuncs.com/So-ViT/So-ViT-19.pth.tar?versionId=CAEQAhiCgIDDutbayBciIGFkNGQ1ODUzNmQxZDRmYmU5YjQ1NzkxNDVmMTJjZDg0">669.60MB</a></td>
         </tr>
</table>




## Installation and Usage

1. Install [PyTorch](https://github.com/pytorch/pytorch) (`>=1.6.0`)
2. Install [timm](https://https://github.com/rwightman/pytorch-image-models) (`==0.3.4`)
3. `pip install thop`
4. type `git clone https://github.com/jiangtaoxie/So-ViT`
5. prepare the dataset as follows
```
.
├── train
│   ├── class1
│   │   ├── class1_001.jpg
│   │   ├── class1_002.jpg
|   |   └── ...
│   ├── class2
│   ├── class3
│   ├── ...
│   ├── ...
│   └── classN
└── val
    ├── class1
    │   ├── class1_001.jpg
    │   ├── class1_002.jpg
    |   └── ...
    ├── class2
    ├── class3
    ├── ...
    ├── ...
    └── classN
```

#### for training from scracth

```shell
sh model_name.sh  # model_name = {So_vit_7/10/14/19}
```

## Acknowledgment
pytorch: https://github.com/pytorch/pytorch

timm: https://github.com/rwightman/pytorch-image-models

T2T-ViT: https://github.com/yitu-opensource/T2T-ViT

## Contact

**If you have any questions or suggestions, please contact me**

`jiangtaoxie@mail.dlut.edu.cn`

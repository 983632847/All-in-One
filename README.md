All-in-One
=========================================
Official Code for [All in One: Exploring Unified Vision-Language Tracking with Multi-Modal Alignment](https://dl.acm.org/doi/10.1145/3581783.3611803) accepted by ACM MM 2023.


## Requirements
- python==3.8.18
- torch==1.13.0
- torchvision==0.14.0
- torchaudio==0.13.0
- timm==0.9.10

## Results (AUC)
|  method   | LaSOT | LaSOTEXT | OTB99-L | TNL2K | WebUAV3M  | Model |
|:---------:|:-----:|:-----:|:------:|:------:|:------:|:------:|
| All-in-One  | 72.8 |  55.8 | 71.0 | 55.9 | 58.5 | [All-in-One](https://pan.baidu.com/s/1OgAFG_LPh9ti4SCt88ILWQ)|
|Raw Results| [LaSOT]() | [LaSOTEXT]()  | [OTB99-L]() | [TNL2K]() | [WebUAV3M]() | - |

It should be noted that the above pretrained model is trained on an Ubuntu 18.04 server with multiple NVIDIA RTX A6000 GPUs. The above results are reported using [analysis_results.py](./tracking/analysis_results.py). For WebUAV-3M, we recommend the official [evaluation toolkit](https://github.com/983632847/WebUAV-3M). 

## Evaluation   
Download the model [All-in-One](https://pan.baidu.com/s/1OgAFG_LPh9ti4SCt88ILWQ), extraction code: `alli`. Add the model to `$PROJECT_ROOT$/All-in-One/lib/models/pretrained_models`.
```
python tracking/test.py --dataset webuav3m --threads 8
python tracking/analysis_results.py
```

Before evaluation, please make sure the data path in [***local.py***](./lib/test/evaluation/local.py) is correct.

## Training
1.Training with one GPU.
```
cd /$PROJECT_ROOT$/All-in-One/lib/train
python run_training_all_in_one.py --save_dir ./output
```

2.Training with multiple GPUs.
```
cd /$PROJECT_ROOT$/All-in-One
python tracking/train.py --save_dir ./output --mode multiple --nproc_per_node 8
```

Before training, please make sure the data path in [***local.py***](./lib/train/admin/local.py) is correct.


## Thanks
This implementation is based on [OSTrack](https://github.com/botaoye/OSTrack). Please ref to their reposity for more details.

## Citation
If you find that this project helps your research, please consider citing our paper:
```
@inproceedings{zhang2023all,
  title={All in One: Exploring Unified Vision-Language Tracking with Multi-Modal Alignment},
  author={Zhang, Chunhui and Sun, Xin and Yang, Yiqian and Liu, Li and Liu, Qiong and Zhou, Xi and Wang, Yanfeng},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={5552--5561},
  year={2023}
}

# LPDGAN


### A Dataset and Model for Realistic License Plate Deblurring [[Paper Link]]
[Haoyan Gong], [Yuzheng Feng], [Zhenrong Zhang], [Xianxu Hou], [Jingxin Liu], [Siqi Huang] and [Hongbin Liu]


## Real-World Motion-blurred License Plate Results

**Comparison with the other methods.**

<img src="https://github.com/haoyGONG/LPDGAN/blob/main/figures/results.jpg" width="800"/>


## Citations
#### BibTeX



## Environment


### Installation
Install Pytorch first.
Then,
```
pip install -r requirements.txt
```
 
## Download Dataset
- The LPBlur dataset is available at [Baidu Netdisk](https://pan.baidu.com/s/1RbffG9eCPDYEa-I96wFx-A) (access code: 7ylj).  


## How To Train
- Refer to `./main.py` for the configuration of the model to train.
- The training command is like
```
python main.py --mode train --dataroot ./dataset
```

The training logs and weights will be saved in the `./checkpoints` folder.

## Acknowledgement
This project is built mainly based on the excellent [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet) and [pix2pix](https://github.com/phillipi/pix2pix) codeframe. We appreciate it a lot for their developers.

## Contact
If you have any question, please email m.g.haoyan@gmail.com.




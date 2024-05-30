# LPDGAN


### A Dataset and Model for Realistic License Plate Deblurring [[Paper Link]](https://arxiv.org/abs/2404.13677)
[Haoyan Gong], [Yuzheng Feng], [Zhenrong Zhang], [Xianxu Hou], [Jingxin Liu], [Siqi Huang] and [Hongbin Liu]


## Real-World Motion-blurred License Plate Results

**Comparison with the other methods.**

<img src="https://github.com/haoyGONG/LPDGAN/blob/main/figures/results.jpg" width="800"/>


## Citations
#### BibTeX



## Environment
torch>=2.1.1+cu121

### Installation
Install Pytorch first.
Then,
```
pip install -r requirements.txt
```
 
## Download Dataset
- The LPBlur dataset is available at [Google Drive](https://drive.google.com/file/d/1ygdpsYdFVUbDoBtcTpb-D0ag9g-ONi0S/view?usp=sharing) and [Baidu Netdisk](https://pan.baidu.com/s/1RbffG9eCPDYEa-I96wFx-A) (access code: 7ylj).  


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




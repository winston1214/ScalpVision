# ScalpVision: A Comprehensive Diagnostic System for Scalp Diseases and Alopecia with Unsupervised Masks and Diffusion Model
<a href='https://mirlab.yonsei.ac.kr/'>MIR Lab in Yonsei University</a> and PAI Lab in Ehwa Women's University

<a href='https://github.com/winston1214'>Youngmin Kim*</a>, <a href='https://github.com/0110tpwls'>Saejin Kim*</a>, <a href='https://github.com/HoyeonM'>Hoyeon Moon</a>, Youngjae Yu, Junhyug Noh

<img src='https://github.com/winston1214/TALMO/blob/master/picture/ScalpVision.png'></img>

## Installation & Preparation
- DIffuseIT-M
1. Please download <a href='https://drive.google.com/file/d/1kfCPMZLaAcpoIcvzTHwVVJ_qDetH-Rns/view?usp=sharing'>256x256 image generation weight</a> or <a href='https://openaipublic.blob.core.windows.net/diffusion/jul-2021/512x512_diffusion.pt'>512x512 image generation weight</a> in `DiffuseIT-M/checkpoints` folder
2. Please install these modules.
```
pip install ftfy regex matplotlib lpips kornia opencv-python torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install color-matcher
pip install git+https://github.com/openai/CLIP.git
```
- SAM
1. Please download <a href='https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'>SAM weight file</a> (ViT-H SAM).
2. Please install these modules.
```
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

## Dataset

- <a href='https://aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=realm&dataSetSn=216'>Dataset Download</a>

## Classifier training
```
python train.py --data_dir $DATA_PATH --epoch $EPOCH --batch_size $BATCH_SIZE --save_dir $SAVE_DIR
```

## Pseudo Image & Mask
- <a href='https://drive.google.com/file/d/1GKpF2Z4Q74_inqkR91z5oW2tnK1x9hwN/view?usp=drive_link'>Pseudo Image & Mask</a> For training $\text{U}^2$-Net

## Weights
- <a href='https://drive.google.com/file/d/11ISRNPL4K1kF7AS3Xy8-mDG9JDImWMhb/view?usp=drive_link'>Pseudo segmentation weights</a> ($\text{U}^{2}$-Net)

## Reference
```
# U^2 Net
@article{qin2020u2,
  title={U2-Net: Going deeper with nested U-structure for salient object detection},
  author={Qin, Xuebin and Zhang, Zichen and Huang, Chenyang and Dehghan, Masood and Zaiane, Osmar R and Jagersand, Martin},
  journal={Pattern recognition},
  volume={106},
  pages={107404},
  year={2020},
  publisher={Elsevier}
}

# DiffuseIT
@article{kwon2022diffusion,
  title={Diffusion-based image translation using disentangled style and content representation},
  author={Kwon, Gihyun and Ye, Jong Chul},
  journal={arXiv preprint arXiv:2209.15264},
  year={2022}
}
```


# Scalp Diagnostic System With Label-Free Segmentation and Training-Free Image Translation (MICCAI 2025)

**Authors:**
**[Youngmin Kim*](https://winston1214.github.io)** ,
[Saejin Kim*](https://0110tpwls.github.io/),
[Hoyeon Moon](https://github.com/HoyeonM),
[Youngjae Yu](https://yj-yu.github.io/home/),
[Junhyeok Noh](https://junhyug.github.io/)

<img src='https://github.com/winston1214/ScalpVision/blob/master/picture/ScalpVision.png'></img>

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

## Hair Segmentation
※ Please check your workspace directory in code

First, you can train $\text{U}^{2}$-Net.

```python segmentation/u2net_train.py```

Next, run this prompt with your pretrained weight file and you can get segmentation labels, $\hat{M}$.

```python segmentation/u2net_test.py```

Next, run this prompt to get point prompt for SAM guidance.

```python segmentation/sam_guide.py```

Next, you can get $M_\text{AP}$.

```python segmentation/sam_predict.py```

Finally, you can get final masks, $M$.

```python segmentation/make_final_mask.py```

## Alopecia Prediction 
⚠️ Note: This module is implemented in code for practical use, but not described in the paper.

- Hair thickness estimation
  
```python alopecia/calculate_hair_thickness.py --img_folder $source_image --save_path $save_folder```

- Hair counting

```python alopecia/calculate_hair_count.py --img_folder $source_image --label_csv $label_csv --save_path $save_folder```

- Alopecia Prediction

```python alopecia/alopecia_prediction.py```

## DiffuseIT-M

```
cd DiffuseIT-M
python main.py -i $source  --output_path $output_path -tg $target --diff_iter 100 --timestep_respacing 200 --skip_timesteps 80 --model_output_size 256 --init_mask $mask --use_range_restart True --use_colormatch True --use_noise_aug_all True --iterations_num 1 --output_file $output_file_name
```


## Classifier training
```
python train.py --data_dir $DATA_PATH --epoch $EPOCH --batch_size $BATCH_SIZE --save_dir $SAVE_DIR
```

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


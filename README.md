# ScalpVision: A Comprehensive Diagnostic System for Scalp Diseases and Alopecia with Unsupervised Masks and Diffusion Model
<img src='https://github.com/winston1214/TALMO/blob/master/picture/ScalpVision.png'></img>

## Installation
```
pip install requirements.txt
```
If you occur "module not found error", please contact @winston1214

## Dataset

- <a href='https://aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=realm&dataSetSn=216'>Dataset Download</a>

### label
- Original
```
- 미세각질 -> microkeratin (value_1)
- 피지 과다 -> sebum (value_2)
- 모낭사이 홍반 -> erythema (value_3)
- 모낭홍반농포 -> pustule (value_4)
- 비듬 -> dandruff (value_5)
- 탈모 -> hair_loss (value_6)
```
- Merge

```python merge_label.py --file_path data --csv_file train_label.csv```

```python merge_label.py --file_path data --csv_file val_label.csv```

```python merge_label.py --file_path data --csv_file test_label.csv```

=> output : data/merge_train_label.csv, data/merge_val_label.csv, data/merge_test_label.csv
```
- microkeartin + dandruff (value_1)
- sebum (value_2)
- erythema + pustule (value_3)
```

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


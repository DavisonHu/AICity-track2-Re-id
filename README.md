#  Pyramid Granularity Attentive Model
Implement of paper:[Learning Discriminative Features with Multiple Granularities for Person Re-Identification](https://arxiv.org/abs/1804.01438v1)

## Dependencies

- Python >= 3.6
- PyTorch >= 1.0.0
- torchvision
- scipy
- numpy
- scikit_learn
- tqdm
- glob
- pandas
- visdom
- PIL
- h5py




## Data

VERI
AICity track 2


## Weights

weights/
	pretrain weight: VERI.pth
	best weight: pyramid.pth

## Train

You can specify more parameters in opt.py

```
python main.py --mode train --data_path <your dataset dir> --class_num <dataset id number>
```

## Evaluate

```
python3 main.py --mode evaluate -data_path <your dataset dir> --weight <weights/pyramid.pth> 
```

## notice

```
when you run code please put the label.csv to the file, ex: put label/AICity/train/label.csv to track 2 train data file
label.csv index is id, image name , pose, color
the txt will save in save/
please run 'python -m visdom.server -port 8098' before you train
```

## Citation

```text
@ARTICLE{Chang19AI,
    author = {{Ming-Ching Chang and Jiayi Wei and Zheng-An Zhu and Yan-Ming Chen and Chan-Shuo Hu and Ming-Xiu Jiang and Chen-Kuo Chiang},
    title = "{{AI} {C}ity {C}hallenge 2019 -- {C}ity-Scale Video Analytics for Smart Transportation}",
    year = 2019,
}
```

# Introduction
1. 키티 데이터의 모노큘라 시퀀스를 위한 데이터로더  
2. 시티스케이프 데이터의 모노큘라 시퀀스를 위한 데이터로더  
(향후 포인트 클라우드를 활용한 핸들링도 포함 예정)  
# Requirements
```
torch
Pillow
skimage
albumentations == 0.5.2
```
# Usage
## KITTI
### Download
```
wget -i splits_kitti/archives2download.txt -P dataset/
```
### Unzip
```
cd dataset
unzip "*.zip"
```
### *.png 2 *.jpg
install moreutils  
```
apt-get update -y
apt-get install moreutils
or
apt-get install -y moreutils
```
install parallel  
```
apt-get install parallel
```
from png to jpg  
```
find dataset/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
find dataset/ -name '*.png' | parallel 'convert {.}.png {.}.jpg && rm {}'
```
### Folder
```
./dataset
    ㄴKITTI
        ㄴ2011_09_26/2011_09_26_drive_0001_sync
                ㄴimage_00/data
                ㄴimage_01/data
                ㄴimage_02/data
                    ㄴ0000000000.jpg
                    ㄴ0000000001.jpg
                ㄴimage_03/data
                    ㄴ0000000000.jpg
                    ㄴ0000000001.jpg
                ㄴvelodyne_points/data
                    ㄴ0000000000.bin
                    ㄴ0000000001.bin
                    ... ...
        ㄴ2011_09_26
        ㄴ2011_09_28
        ㄴ2011_09_29
        ㄴ2011_09_30
        ㄴ2011_10_03
./model_loader
    __init__.py
    kitti.py
./splits
    /kitti_benchmark
        /train_files.txt
        /val_files.txt
    ... ...
```
### Example
```
splits     = ["kitti_benchmark", "kitti_eigen_full", "kitti_eigen_zhou"]
datapath   = "./dataset/kitti"
filepath   = "./splits" + "/" + splits[1] + "/" + "{}_files.txt"

batch_size = 8
scale      = 1
frame_ids  = [0, -2, -1, 1, 2]
key_frame  = 2

train_filename = readlines(filepath.format("train")
train_dataset  = KITTIMonoDataset(
    datapath, train_filename, True, frame_ids, ".jpg", 192, 640, 4)
train_loader   = DataLoader(
    train_dataset, batch_size, True, num_workers = 4, pin_memory = True, drop_last = True)
```
## Cityscapes
### Folder
```
./dataset (직접 구성해야함)
    /cityscapes
        ... ...
./model_loader
    __init__.py
    cityscapes.py
./splits
    /cityscapes_benchmark
        /train_files.txt
        /val_files.txt
        /test_files.txt
    /cityscaeps_watson
        /train_files.txt
        /val_files.txt
        /test_files.txt
```
### Example
train_filename = read_lines("./splits/city_watson_full/train_files.txt")
dataloader     = CityscapesMonoDataset(
    "./dataset/cityscapes", train_filename, True, [0, -1, 1], "train", ".png", 256, 512, 4)
### cityscapes splits
``` 
1. cityscapes_benchmark  
모든 촬영 장소마다의 촬영 이미지를 [위치, 이미지 이름] 형태로 저장해놓은 것  
2. cityscapes_watson  
[ManyDepth (CVPR 2021)](https://arxiv.org/abs/2104.14540)에서 사용한 splits  
```
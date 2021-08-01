# CityscapesDataset
시티스케이프 모노큘라 시퀀스를 위한 데이터로더
# Requirements
```
Pytorch
albumentations == 0.5.2
```
# Folder
```
./dataset (직접 구성해야함)
    /cityscapes
        ...
        ...
./model_dataloader
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
    /cityscaeps_landau
        /train_files.txt
        /val_files.txt
        /test_files.txt
```
- splits 설명  
1. cityscapes_benchmark  
모든 촬영 장소마다의 촬영 이미지를 [위치, 이미지 이름] 형태로 저장해놓은 것  
2. cityscapes_watson  
[ManyDepth (CVPR 2021)](https://arxiv.org/abs/2104.14540)에서 사용한 splits  
3. cityscpaes_landau  
프레임 아이디 [-4, -3, -2, -1, 0]까지 시퀀스로 묶을 수 있도록,  
프레임 청크 (30 프레임) 마다 양 끝 4개의 프레임은 제거한 splits  
# Usage
```
train_filename = read_lines("./splits/cityscapes_landau/train_files.txt")
dataloader     = CityscapesMonoDataset(
                    "./dataset/cityscapes", train_filename, True, [0, -1, 1], "train", ".png", 1)
```

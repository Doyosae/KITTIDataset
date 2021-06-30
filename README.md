# Requirements
```
torch
Pillow
skimage
albumentations == 0.5.2
```
# Preparing KITTI Data
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
### KITTI folder
```
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
    ㄴ2011_09_26/2011_09_26_drive_0002_sync
    ㄴ2011_09_28
    ㄴ2011_09_29
    ㄴ2011_09_30
    ㄴ2011_10_03
```
# Usage (Example)
```
splits   = ["kitti_benchmark", "kitti_landau", "kitti_eigen_full", "kitti_eigen_zhou"]
datapath = "./dataset/kitti"
filepath = "./splits" + "/" + splits[1] + "/" + "{}_files.txt"

batch_size = 8
scale      = 1
frame_ids  = [0, -2, -1, 1, 2]
key_frame  = 2
train_filename = readlines(filepath.format("train")
train_dataset  = KITTIMonoDataset(
                    datapath, train_filename, is_training = True, frame_ids = frame_ids, ext = True, scale_factor = scale))
train_loader   = DataLoader(train_dataset, batch_size, True, num_workers = 4, pin_memory = True, drop_last = True)
```
# Support
모노큘라 태스크를 풀 경우, 입력 프레임 시퀀스에서 키 프레임은 중앙에 있는 경우가 대부분  
경우에 따라 키 프레임이 오른쪽 끝 (시간 방향으로 가장 마지막)에 있을 수 있다.  
기존 KITTI 데이터에서 splits 타입으로 알려진건 kitti_benchmark, kitti_eigen, kitti_eigen_zhou  
1. frame ids: [-1, 0, 1]
    - [x] kitti_benchmark
    - [x] kitti_eigen
    - [x] kitti_eigen_zhou
2. frame ids: [-2, -1, 0]
    - [x] kitti_benchmark
    - [] kitti_eigen
    - [] kitti_eigen_zhou
3. frame ids: [-2, -1, 0, 1, 2]
    - [x] kitti_benchmark
    - [] kitti_eigen
    - [] kitti_eigen_zhou
4. frame_ids: [-4, -3, -2, -1, 0]
    - [] kitti_benchmark
    - [] kitti_eigen
    - [] kitti_eigen_zhou
직접 만든 kitti_landau splits은 KITTI 데이터의 날짜별 시퀀스 마다 양 끝 4장의 목록은 싣지 않음  
그리고 여러 논문들의 KITTI 레거시 데이터 로더와 호환되는 양식으로 작성  
4. 위의 모든 경우에 대해서
    - [x] kitti_landau
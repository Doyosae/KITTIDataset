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
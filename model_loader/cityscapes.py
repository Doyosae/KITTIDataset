import os
import cv2
import random
import numpy as np
from PIL import Image
from natsort import natsorted

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import skimage.transform
import albumentations
from albumentations import Resize
if albumentations.__version__ == "0.5.2":
    from albumentations.pytorch.transforms import ToTensor
else:
    from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.transforms import HorizontalFlip
from albumentations.augmentations.transforms import ColorJitter
from model_utility import *



class CityscapesMonoDataset(Dataset):
    def __init__(self, datapath, filename, is_training, frame_ids, mode, ext, height, width, scale = 4):
        super(CityscapesMonoDataset, self).__init__()
        """
        Args:
            datapath:    "./dataset/cityscapes"
            filename:    splits file of KITTI
            is_training: True or False
            frame_ids:   relative position list of key frame
            mode:        "train" or "val" or "test"
            ext:         ".jpg" or ".png"
            height:      height of image
            width:       width of image
            scale:       pyramid scale of image

        albumentation Resize interpolation option
        0 : cv2.INTER_NEAREST, 
        1 : cv2.INTER_LINEAR, 
        2 : cv2.INTER_CUBIC, 
        3 : cv2.INTER_AREA, 
        4 : cv2.INTER_LANCZOS4. Default: cv2.INTER_AREA.
        """
        if height % 32 != 0 or width % 32 != 0:
            raise "(H, W)는 32의 나눗셉 나머지가 0일 것, Cityscapes 권장 사이즈는 (512, 1024) or (256, 512)"

        self.datapath     = datapath
        self.filename     = filename
        self.is_training  = is_training
        self.frame_ids    = frame_ids        
        self.mode         = mode
        self.ext          = ext
        self.scale        = scale
        self.inter        = cv2.INTER_AREA # Image.ANTIALIAS와 동등한가?
        self.side_map = {
            "l": "_leftImg8bit", 
            "r": "_rightImg8bit"}
        self.cam_path = {
            "l": "leftImg8bit_sequence_trainvaltest/leftImg8bit_sequence",
            "r": "rightImg8bit_sequence_trainvaltest/rightImg8bit_sequence"}
        self.K = np.array([[1.104, 0, 0.535, 0],
                           [0, 2.212, 0.501, 0],
                           [0,    0,      1, 0],
                           [0,    0,      0, 1]], dtype = np.float32)

        self.resize        = {}
        self.height        = height
        self.width         = width
        self.scales        = list(range(scale))
        self.origin_scale  = (1024, 2048)
        self.scale_list    = [(self.height//(2**i), self.width//(2**i)) for i in self.scales]

        for scale, (height, width) in enumerate(self.scale_list):
            self.resize[scale] = Resize(
                height = int(height), width  = int(width), interpolation = self.inter)
        # depth_resize는 interpolation = 0으로 설정
        self.depth_resize   = Resize(
            height = self.origin_scale[0], width = self.origin_scale[1], interpolation = 0)

        self.augment_key    = "image"
        self.brightness     = (0.8, 1.2)
        self.contrast       = (0.8, 1.2)
        self.saturation     = (0.8, 1.2)
        self.hue            = (-0.1, 0.1)
        self.HorizontalFlip = HorizontalFlip(p = 1.0)
        self.ColorJitter    = ColorJitter(
            brightness = self.brightness, contrast = self.contrast, saturation = self.saturation, hue = self.hue, p = 1.0)
        if albumentations.__version__ == "0.5.2":
            self.image2tensor = ToTensor()
        else:
            self.image2tensor = ToTensorV2()
            
        print(">>>  CITYSCAEPS scaling table")
        print(">>>  Interpolation     :  {0}".format(self.inter))
        print(">>>  Is training???    :  {0}".format(self.is_training))
        print(">>>  Resolution List   :  {0}".format(self.scale_list))



    def flip_image(self, numpy_image):
        numpy_image = self.HorizontalFlip(image = numpy_image)
        return numpy_image[self.augment_key]

    def resize_image(self, scale, numpy_image):
        numpy_image = self.resize[scale](image = numpy_image)
        return numpy_image[self.augment_key]

    def recolor_image(self, numpy_image):
        numpy_image = self.ColorJitter(image = numpy_image)
        return numpy_image[self.augment_key]

    def numpy2tensor(self, numpy_image):
        if albumentations.__version__ == "0.5.2":
            tensor_image = self.image2tensor(image = numpy_image)
        else:
            tensor_image = self.image2tensor(image = numpy_image)
            tensor_image[self.augment_key] = tensor_image[self.augment_key] / 255.0
        return tensor_image[self.augment_key]


    def get_image_path(self, folder_name, key_frame, side):
        image_name = "{0}{1}{2}".format(key_frame, self.side_map[side], self.ext)
        image_path = os.path.join(self.datapath, self.cam_path[side], self.mode, folder_name, image_name)
        return image_path

    # def get_camera_path(self, folder_name, key_frame):
    #     camera_name = "{0}{1}{2}".format(key_frame, "_camera", ".json")
    #     camera_path = os.path.join(self.datapath, self.cam, self.mode, folder_name, camera_name)
    #     return camera_path

    def load_image(self, image_path, do_flip): # 이미지를 로드, 나중에 PIL로 고치기
        with open(image_path, 'rb') as f:
            with Image.open(f) as img:
                image_instance = img.convert('RGB')
                numpy_image    = np.array(image_instance)

                if do_flip == True:
                    numpy_image = self.flip_image(numpy_image)
                return numpy_image

    def preprocessing_image(self, input_data, folder_name, key_frame, do_flip, side):
        """
        key_frame는 키프레임 (시퀀스의 중앙에 있을수도, 맨 뒤에 있을수도 있음)
        frame_ids가 중요한데 key_frame (키 프레임) 기준으로 상대적인 위치를 나타냄
        ex)
        key_frame = 123, frame_ids = [-1, 0, 1]
        for index in frame_ids:
            outputs = load_image(index + key_frame)
        """
        for frame_id in self.frame_ids:
            relative_frame_id = "{:06d}".format(int(key_frame) + frame_id)
            image_path  = self.get_image_path(folder_name, self.location + "_" + self.frame_num + "_" + relative_frame_id, side)
            image_array = self.load_image(image_path, do_flip)
            input_data.update({("color", frame_id, scale): self.resize_image(scale, image_array) for scale in self.scales})
        return input_data

    def preprocessing_intrinsic(self, input_data):
        """
        1. 원본 intrinsic을 사용할 경우, "스케일링 크기 / 원본 크기" 비율을 곱해서 intrinsic을 줄여줌
        2. monodepth2의 intrinsic을 사용할 경우, 스케일링 크기만 곱해서 intrinsic을 늘려줌
        """
        for scale in self.scales:
            K_copy       = self.K.copy()
            K_copy[0, :] = K_copy[0, :] * self.scale_list[scale][1]
            K_copy[1, :] = K_copy[1, :] * self.scale_list[scale][0]
            inv_K        = np.linalg.pinv(K_copy)

            input_data[("K", scale)]     = torch.from_numpy(K_copy)
            input_data[("inv_K", scale)] = torch.from_numpy(inv_K)
        return input_data


    def __getitem__(self, index):
        """
        returns 
            ("color", <frame_id>, <scale>)             for raw color images,
            ("color_aug", <frame_id>, <scale>)         for aug color images,
            ("depth", 0) and ("K", 0) and ("inv_K", 0) for depth, intrinsic of key frame
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        do_flip     = self.is_training and random.random() > 0.5
        do_auge     = self.is_training and random.random() > 0.5
        
        batch_line  = self.filename[index].split() # 폴더이름, 키프레임 인덱스, 카메라
        folder_name = batch_line[0]
        key_frame   = batch_line[1]
        side        = batch_line[2]
        self.location, self.frame_num, self.frame_index = key_frame.split("_")
    
        input_data = {}
        input_data = self.preprocessing_image(input_data, folder_name, self.frame_index, do_flip, side) 
        if do_auge:
            for frame_id in self.frame_ids:
                input_data.update({("color_aug", frame_id, scale):
                    self.recolor_image(input_data[("color", frame_id, scale)]) for scale in self.scales})
        else:
            for frame_id in self.frame_ids:
                input_data.update({("color_aug", frame_id, scale):
                    input_data[("color", frame_id, scale)] for scale in self.scales})
        input_data.update({key: self.numpy2tensor(input_data[key]) for key in input_data})

        input_data = self.preprocessing_intrinsic(input_data)
        return input_data

    def __len__(self):
        return len(self.filename)



class GetCityScapes(object):
    def __init__(self, datapath: str, mode: str, cut: list):
        """
        시티스케이프 시퀀스용 데이터는 좌, 우 카메라마다 31곳의 위치와 train, val, test 합해서 총 150,002장의 사진을 제공
        한 번 촬영은 30장, 즉 1초 단위로 되어있음 (30FPS 촬영), 그래서 30장으로 끊어주면 편함
        카메라 파라미터는 모든 촬영물에서 동일한 것으로 보임

        Args:
            datapath: "./dataset/cityscapes
            mode: "train" or "val" or "test"
            cut: 양 끝을 자를 프레임 수, 예를 들어서 양 끝을 4프레임씩 자르면 [4: len(list) - 4]
        """
        self.datapath = datapath
        self.mode     = mode
        self.cut      = cut
        self.side_map = {
            "l": "_leftImg8bit", 
            "r": "_rightImg8bit"}
        self.cam_path = {
            "l": "leftImg8bit_sequence_trainvaltest/leftImg8bit_sequence",
            "r": "rightImg8bit_sequence_trainvaltest/rightImg8bit_sequence"}


    def side_cut(self, filename, l: int, r: int):
        """
        이 클래스가 존재하는 이유
        프레임 인덱스가 키 프레임 기준으로 좌, 우 몇 까지 consecutive frame을 쓸 지 모름
        만약 키 프레임 인덱스가 0이면 왼쪽 consecutive frame은 음수가 되기 때문에 사용할 수 없음
        그래서 consecutive frame 범위를 두기 위해 맨 끝 프레임은 잘라내는 목적
        """
        num_chunk = len(filename) // 30
        print("Number divided by 30       :  {}".format(num_chunk))
        filename = [filename[30*index: 30*(index+1)] for index in range(num_chunk)]
        
        modified_filename = []
        for chunk in filename:
            modified_filename += chunk[l: 30-r]
        
        print("Modified Filename Length   :  {}".format(len(modified_filename)))

        random.shuffle(modified_filename)
        return modified_filename


    def search(self):
        all_filename = []

        # ex) "./dataset/cityscapes/leftImg8bit_sequence_trainvaltest/leftImg8bit_sequence/train"
        location_path = os.path.join(self.datapath, self.cam_path["l"], self.mode) # 왼쪽, 오른쪽 카메라 파일들이 모두 똑같아서 왼쪽 경로를 탐색으로 활용
        for location in os.listdir(location_path):
            
            # ex) "./dataset/cityscapes/leftImg8bit_sequence_trainvaltest/leftImg8bit_sequence/train/aachen"
            file_path = os.path.join(location_path, location)
            for filename in os.listdir(file_path):
                line = filename.split(self.side_map["l"]) # aachen_000000_000000_leftImg8bit -> [aachen_000000_000000, aachen]

                # 촬영 좡소마다 사진 파일들을 "위치 사진 파일 이름" 형태의 무자열로 모두 append
                all_filename.append("".join([location, " ", line[0], " ", "l"])) # [aachen, aachen_000000_000000, "l"]
                all_filename.append("".join([location, " ", line[0], " ", "r"])) # [aachem, aachem_000000_000000, "r"]
        
        all_filename = natsorted(all_filename) # 모든 파일 이름 정렬
        print("Filename Full Length       :  {}".format(len(all_filename)))

        if self.cut == [0, 0]:
            print("Cut filename X")
            return all_filename
        elif self.cut != [0, 0]:
            print("Cut filename O")
            modified_filename = self.side_cut(all_filename, self.cut[0], self.cut[1])
            return modified_filename
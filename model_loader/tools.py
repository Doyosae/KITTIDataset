import os
import random
from natsort import natsorted



class GetKITTI(object):
    def __init__(self, datapath: str, mode: list, cut: list):
        """
        KITTI 데이터 폴더의 구조
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
        
        벨로다인 포인트 파일을 기준으로 키프레임 선택 (키프레임은 GT가 존재해야함)
        따라서 벨로다인 파일의 확장자를 제거한 프레임 인덱스로 키프레임 형식으로 지정
        해당 프레임 인덱스 이름에 yyyy_mm_dd 형태와 side_map을 섞어서 부여
        (레거시 kitti_splits 규약을 따름)

        ex)
        2011_10_03/2011_10_03_drive_0034_sync 0000000190 r
        2011_10_03/2011_10_03_drive_0034_sync 0000000307 l
        2011_09_26/2011_09_26_drive_0117_sync 0000000134 r
        2011_09_30/2011_09_30_drive_0028_sync 0000000534 l

        Args:
            datapath: "./dataset/kitti"
            model: "train" or "val" 데이터 폴더 리스트
            side: "l" or "r"
        """
        self.datapath  = datapath
        self.mode      = mode
        self.cut       = cut
        self.side_map  = {"2": 2, "3": 3, "l": 2, "r": 3}
        self.l_path    = "image_02/data"
        self.r_path    = "image_03/data"
        self.velo_path = "velodyne_points/data"


    def search(self):
        all_filename = []
        for yyyy_mm_dd in self.mode: # 각 날짜 별 폴더마다 순회
            velodyne_path = os.path.join(self.datapath, yyyy_mm_dd, self.velo_path)
            # left_path  = os.path.join(self.datapath, yyyy_mm_dd, self.l_path)
            # right_path = os.path.join(self.datapath, yyyy_mm_dd, self.r_path)

            # 왼쪽 카메라 파일 (image_02) 와 오른쪽 카메라 파일 (image_03)을 순회
            left_filename  = []
            right_filename = []
            for filename in os.listdir(velodyne_path): # 벨로다인 파일 이름을 순회
                # 0000000011.bin, ... ... 0000000035.bin 파일을 0000000011, 0000000035으로 쪼갬
                frame_index = filename.split(".bin")[0]

                # 쪼갠 frame_index은 image_02, image_03 폴더에 이미지 파일로 매칭됨 (스테레오 이미지)
                left_filename.append("".join([yyyy_mm_dd, " ", frame_index, " ", "l"]))
                right_filename.append("".join([yyyy_mm_dd, " ", frame_index, " ", "r"]))
            
            # 상대적인 프레임 아이디를 위해 양 끝 N개는 잘라줌
            modified_left_filename  = self.side_cut(left_filename, self.cut)
            modified_right_filename = self.side_cut(right_filename, self.cut)
            all_filename += modified_left_filename
            all_filename += modified_right_filename

        print("전체 길이  :  {}".format(len(all_filename)))
        random.shuffle(all_filename)
        return all_filename


    def side_cut(self, filename, cut):
        """
        이 클래스가 존재하는 이유
        프레임 인덱스가 키 프레임 기준으로 좌, 우 몇 까지 consecutive frame을 쓸 지 모름
        만약 키 프레임 인덱스가 0이면 왼쪽 consecutive frame은 음수가 되기 때문에 사용할 수 없음
        그래서 consecutive frame 범위를 두기 위해 맨 끝 프레임은 잘라내는 목적
        """        
        modified_filename = []
        modified_filename = filename[cut[0]: len(filename)-cut[1]]
        return modified_filename



class GetCityscapes(object):
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
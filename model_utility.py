import os
import time
import json
from tqdm import tqdm
import numpy as np
from scipy import misc
from collections import Counter

import torch
import torch.nn as nn
import matplotlib.pyplot as plt



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
KITTI 데이터셋을 위한 함수 모듈
def readlines  스플릿 파일을 읽어들이는 함수
def savelines  스플릿 파일을 저장하는 함수
def removefile 레거시 splits에서 n개 프레임 이상을 쓰기 위해 프레임 인덱스가 n 이하인 파일은 제거하는 함수
def read_cam2cam # 카메라 캘리브레이션 파일을 읽는 함수
def read_velo2cam # 벨로다인 캘리브레이션 파일을 읽는 함수
def read_velodyne_points # 포인트 클라우드를 로드하는 함수
def sub2ind
def Point2Depth # 실제로 쓰는 함수
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def readlines(datapath):
    # Read all the lines in a text file and return as a list
    with open(datapath, 'r') as f:
        lines = f.read().splitlines()
    return lines


def savelines(filename, datapath: str):
    f = open(datapath, "w")
    for data in filename:
        f.write(data + "\n") 
    f.close()


def removelines(datapath, filename, frame_ids):
    """
    for KITTI
    Args:
        filename:  ['2011_09_26/2011_09_26_drive_0057_sync 311 l', 
                    '2011_09_26/2011_09_26_drive_0035_sync 130 r]
        frame_ids: [-3, -2, -1, 0, 1, 2]
    """
    modified_key = []
    side_map     = {"2": 2, "3": 3, "l": 2, "r": 3}
    for index, data in enumerate(filename):
        line = data.split()
        name = line[0]      # 2011_09_26/2011_09_26_drive_0035_sync or
        key  = int(line[1]) # 311 or 130
        side = line[2]      # l or r
        length = len(os.listdir(datapath + "/" + name + "/" + "image_0{}/data".format(side_map[side])))
        
        if key in list(range(-frame_ids[0], length - frame_ids[-1] - 1)): # 포함되어 있다면
            modified_key.append(data)                                     # 수정된 키 리스트에 추가
        else:
            pass
    return modified_key


def read_cam2cam(path):
    data = {}
    with open(path, "r") as f:
        for line in f.readlines():
            key, value = line.split(":", 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass   

    P_rect_02 = np.reshape(data['P_rect_02'], (3, 4))
    P_rect_03 = np.reshape(data['P_rect_03'], (3, 4))
    intrinsic_left =  P_rect_02[:3, :3]
    intrinsic_right = P_rect_03[:3, :3]
    
    identity_l  = np.eye(4)
    identity_r  = np.eye(4)
    identity_l[:3, :3] = intrinsic_left
    identity_r[:3, :3] = intrinsic_right
    identity_l = identity_l.astype(np.float32)
    identity_r = identity_r.astype(np.float32)
    return identity_l, identity_r


def read_velo2cam(path):
    """
    벨로다인 캘리브레이션 파일을 읽어들이는 함수
    Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass
    return data


def read_velodyne_points(filename):
    """
    Load 3D point cloud from KITTI file format
    (adapted from https://github.com/hunse/kitti)
    """
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points


def sub2ind(matrixSize, rowSub, colSub):
    """
    Convert row, col matrix subscripts to linear indices
    """
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1


def Point2Depth(calib_path, point_path, cam = 2, vel_depth = False):
    """
    캘리브레이션 경로와 벨로다인 파일 경로를 읽어서 뎁스 맵을 만드는 함수
    Args:
        calib_path: ./dataset/kitti-master/2011_09_26
        point_path: ./dataset/kitti-master/2011_09_26/2011_09_26_drive_0022_sync/velodyne_points/data/0000000473.bin

    returns:
        GT depth image (np.max: 80.0, np.min: 0.1)
        shape: [375, 1242]
    """
    # 1. load calibration files
    cam2cam  = read_velo2cam(os.path.join(calib_path + "/" + "calib_cam_to_cam.txt"))
    velo2cam = read_velo2cam(os.path.join(calib_path + "/" + "calib_velo_to_cam.txt"))
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # 2. 이미지 모양을 획득 (375, 1242)
    im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)
    
    # 3.
    # 3차원 포인트 점을 카메라 좌표계로 변환하고 다시 K를 곱해서 이미지로 사영시키는 수식
    # 먼저 4x4 항등행렬을 선언하고 여기서 3x3 부분은 회전 행렬을 붙인다. (R_rect_00)
    # 그리고 모션 벡터를 cam2cam의 P_rect_0 성분을 불러와서 둘을 np.dot한다.
    # 마지막으로 velo2cam 매트릭스를 np.dot하면 벨로다인 포인트 -> 이미지로 사영하는 매트릭스를 만듬
    R_cam2rect = np.eye(4)                                  # 4x4 항등행렬
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3) # 회전 운동
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3, 4)     # 모션 벡터
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # 4.
    # 벨로다인 포인트 클라우드를 불러오고, x, y, z, 1의 homogenous 좌표계로 만듬
    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = read_velodyne_points(point_path)
    velo = velo[velo[:, 0] >= 0, :]

    # 5. 벨로다인 포인트 homogenous 값을 카메라의 이미지 좌표에 사영하는 계산과정 이미지 = 사영행렬 * 3차원 벨로다인 포인트
    velo_pts_im        = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis] # shape is (포인트 갯수, x, y, 1 값)

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    # 1. velo_path_im.shape는 3개 (x, y, 1) 성분이 61021개 있다. 여기의 x, y 좌표에서 1씩 빼준 것을 다시 velo_pts_im[:, 0] and [:, 1]에 대입
    # 2. 그리고 x 좌표가 0 이상이고 y 좌표가 0 이상인 값만 유효한 인덱스로 취급한다.
    # 3. 그리고 val_ind 이면서 동시에 velo_pts_im 좌표의 위치가 이미지의 크기보다 작은 것만 다시 val_inds로 할당 (그래야만 이미지에 좌표가 잘 맺히므로)
    # 4. 마지막으로 그 유효한 좌표의 위치, 즉 True만 velo_pts_im로 취급
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds          = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds          = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im       = velo_pts_im[val_inds, :]

    depth = np.zeros((im_shape[:2])) # 이미지로 사영, 375, 1245 사이즈의 zero map을 만듬
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # 마지막, 중복된 값을 제거
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts   = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()

    depth[depth < 0] = 0
    return depth



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
CITYSCAPES 데이터셋을 위한 함수
def removeline_city
def 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def read_lines(datapath):
    file = open(datapath, "r")
    lines = file.readlines()
    lines = [line.rstrip(" \n") for line in lines]
    return lines


def save_lines(filename, datapath: str):
    f = open(datapath, "w")
    for data in filename:
        f.write(data + "\n") 
    f.close()
    


# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# SINTEL 데이터셋을 위한 함수
# def flow_read
# def depth_read
# def cam_raed
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# TAG_FLOAT = 202021.25
# TAG_CHAR  = 'PIEH'
# def read_flow(filename):
#     """
#     Read optical flow from file, return (U,V) tuple. 
#     Original code by Deqing Sun, adapted from Daniel Scharstein.
#     """
#     f = open(filename,'rb')
#     check = np.fromfile(f,dtype=np.float32,count=1)[0]
#     assert check == TAG_FLOAT, ' flow_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
#     width = np.fromfile(f,dtype=np.int32,count=1)[0]
#     height = np.fromfile(f,dtype=np.int32,count=1)[0]
#     size = width*height
#     assert width > 0 and height > 0 and size > 1 and size < 100000000, ' flow_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
#     tmp = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width*2))
#     u = tmp[:,np.arange(width)*2]
#     v = tmp[:,np.arange(width)*2 + 1]
#     return u,v


# def read_depth(filename):
#     """ 
#     Read depth data from file, return as numpy array.
#     """
#     f = open(filename,'rb')
#     check = np.fromfile(f,dtype=np.float32,count=1)[0]
#     assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
#     width = np.fromfile(f,dtype=np.int32,count=1)[0]
#     height = np.fromfile(f,dtype=np.int32,count=1)[0]
#     size = width*height
#     assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
#     depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
#     return depth


# def read_cam(filename):
#     """
#     Read camera data, return (M,N) tuple.
#     M is the intrinsic matrix, N is the extrinsic matrix, so that

#     x = M*N*X,
#     with x being a point in homogeneous image pixel coordinates, X being a
#     point in homogeneous world coordinates.
#     """
#     f     = open(filename,'rb')
#     check = np.fromfile(f, dtype = np.float32, count = 1)[0]
#     assert check == TAG_FLOAT, ' cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
#     intrinsic = np.fromfile(f,dtype='float64',count=9).reshape((3,3))
#     extrinsic = np.fromfile(f,dtype='float64',count=12).reshape((3,4))
#     return intrinsic, extrinsic



# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# FLYINGTHINGS3D 데이터셋을 위한 함수
# def read
# def write
# def readPFM
# ...
# ...
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# def read(file):
#     if file.endswith('.float3'): return readFloat(file)
#     elif file.endswith('.flo'): return readFlow(file)
#     elif file.endswith('.ppm'): return readImage(file)
#     elif file.endswith('.pgm'): return readImage(file)
#     elif file.endswith('.png'): return readImage(file)
#     elif file.endswith('.jpg'): return readImage(file)
#     elif file.endswith('.pfm'): return readPFM(file)[0]
#     else: raise Exception('don\'t know how to read %s' % file)


# def write(file, data):
#     if file.endswith('.float3'): return writeFloat(file, data)
#     elif file.endswith('.flo'): return writeFlow(file, data)
#     elif file.endswith('.ppm'): return writeImage(file, data)
#     elif file.endswith('.pgm'): return writeImage(file, data)
#     elif file.endswith('.png'): return writeImage(file, data)
#     elif file.endswith('.jpg'): return writeImage(file, data)
#     elif file.endswith('.pfm'): return writePFM(file, data)
#     else: raise Exception('don\'t know how to write %s' % file)


# def readPFM(file):
#     file = open(file, 'rb')
#     color = None
#     width = None
#     height = None
#     scale = None
#     endian = None

#     header = file.readline().rstrip()
#     if header.decode("ascii") == 'PF':
#         color = True
#     elif header.decode("ascii") == 'Pf':
#         color = False
#     else:
#         raise Exception('Not a PFM file.')

#     dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
#     if dim_match:
#         width, height = list(map(int, dim_match.groups()))
#     else:
#         raise Exception('Malformed PFM header.')

#     scale = float(file.readline().decode("ascii").rstrip())
#     if scale < 0: # little-endian
#         endian = '<'
#         scale = -scale
#     else:
#         endian = '>' # big-endian

#     data = np.fromfile(file, endian + 'f')
#     shape = (height, width, 3) if color else (height, width)

#     data = np.reshape(data, shape)
#     data = np.flipud(data)
#     return data, scale


# def writePFM(file, image, scale=1):
#     file = open(file, 'wb')
#     color = None

#     if image.dtype.name != 'float32':
#         raise Exception('Image dtype must be float32.')

#     image = np.flipud(image)
#     if len(image.shape) == 3 and image.shape[2] == 3: # color image
#         color = True
#     elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
#         color = False
#     else:
#         raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

#     file.write('PF\n' if color else 'Pf\n'.encode())
#     file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))
#     endian = image.dtype.byteorder
#     if endian == '<' or endian == '=' and sys.byteorder == 'little':
#         scale = -scale

#     file.write('%f\n'.encode() % scale)
#     image.tofile(file)


# def readFlow(name):
#     if name.endswith('.pfm') or name.endswith('.PFM'):
#         return readPFM(name)[0][:,:,0:2]
#     f = open(name, 'rb')

#     header = f.read(4)
#     if header.decode("utf-8") != 'PIEH':
#         raise Exception('Flow file header does not contain PIEH')

#     width = np.fromfile(f, np.int32, 1).squeeze()
#     height = np.fromfile(f, np.int32, 1).squeeze()
#     flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))
#     return flow.astype(np.float32)


# def readImage(name):
#     if name.endswith('.pfm') or name.endswith('.PFM'):
#         data = readPFM(name)[0]
#         if len(data.shape)==3:
#             return data[:,:,0:3]
#         else:
#             return data
#     return misc.imread(name)


# def writeImage(name, data):
#     if name.endswith('.pfm') or name.endswith('.PFM'):
#         return writePFM(name, data, 1)
#     return misc.imsave(name, data)


# def writeFlow(name, flow):
#     f = open(name, 'wb')
#     f.write('PIEH'.encode('utf-8'))
#     np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
#     flow = flow.astype(np.float32)
#     flow.tofile(f)


# def readFloat(name):
#     f = open(name, 'rb')
#     if(f.readline().decode("utf-8"))  != 'float\n':
#         raise Exception('float file %s did not contain <float> keyword' % name)

#     dim = int(f.readline())
#     dims = []
#     count = 1
#     for i in range(0, dim):
#         d = int(f.readline())
#         dims.append(d)
#         count *= d

#     dims = list(reversed(dims))
#     data = np.fromfile(f, np.float32, count).reshape(dims)
#     if dim > 2:
#         data = np.transpose(data, (2, 1, 0))
#         data = np.transpose(data, (1, 0, 2))
#     return data


# def writeFloat(name, data):
#     f = open(name, 'wb')
#     dim=len(data.shape)
#     if dim>3:
#         raise Exception('bad float file dimension: %d' % dim)

#     f.write(('float\n').encode('ascii'))
#     f.write(('%d\n' % dim).encode('ascii'))

#     if dim == 1:
#         f.write(('%d\n' % data.shape[0]).encode('ascii'))
#     else:
#         f.write(('%d\n' % data.shape[1]).encode('ascii'))
#         f.write(('%d\n' % data.shape[0]).encode('ascii'))
#         for i in range(2, dim):
#             f.write(('%d\n' % data.shape[i]).encode('ascii'))

#     data = data.astype(np.float32)
#     if dim==2:
#         data.tofile(f)

#     else:
#         np.transpose(data, (2, 0, 1)).tofile(f)



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
기타 모듈
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class tools(object):
    def __init__(self):
        pass

    @staticmethod
    def tensor2numpy(tensor): # 토치 텐서를 넘파이로
        return tensor.numpy()


    @staticmethod
    def numpy2tensor(numpy): # 넘파이를 토치 텐서로
        return torch.from_numpy(numpy)


    @staticmethod
    def parameter_number(model): # 모델의 파라미터를 계산하는 함수
        raise NotImplementedError


    @staticmethod
    def sample_dataset(dataloader, end): # 모델 데이터로터에서 배치 샘플 하나를 추출
        test = []
        start = time.time()
        for index, data in tqdm(enumerate(dataloader)):
            test.append(data)
            if index == end:
                break
            if index == "all":
                pass
        print("batch sampling time:  ", time.time() - start)
        return test


    @staticmethod
    def show_image(image, option = "torch", size = (10, 4), cmap = "magma", show_disp = True):
        """
        토치나 텐서플로우 형태의 이미지를 받아서 이미지를 띄우는 함수
        Args: tensor type
                Pytorch:    [B, N, H, W]
                Tensorflow: [B, H, W, C]
        """
        plt.rcParams["figure.figsize"] = size

        if option == "torch":
            if image.shape[0] == 3 or len(image.shape) == 3:
                image = np.transpose(image, (1, 2, 0))
            else:
                image = np.squeeze(image, axis = 0)
                image = np.transpose(image, (1, 2, 0))

        elif option == "tensorflow": # N H W C
            if len(image.shape) == 3:
                pass
            else:
                image = np.squeeze(image, axis = 3)

        """
        uint8 -> float32로 바꾸면 엄청 큰 정수 단위의 float32가 됨
        따리서 255.로 나누어 주는 것이 중요
        그리고 cv2.imread로 불러온 이미지를 plt.imshow로 띄울때는 cv2.COLOR_BGR2RGB
        """
        if show_disp:
            plt.imshow(image, cmap = cmap, vmax = np.percentile(image, 95))
            plt.axis('off')
            plt.show()
        else:
            plt.imshow(image, cmap = cmap)
            plt.axis('off')
            plt.show()


    @staticmethod
    def show_graph(data, xlabel, ylabel, title, color, marker, linestyle):
        """
        data:   np.array
        xlabel: str
        ylabel: str
        title:  str
        color:  str
        marker: "o" or 
        """
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams['lines.linewidth'] = 1.5
        plt.rcParams['axes.grid'] = True 
        plt.rc('font', size = 10)        # 기본 폰트 크기
        plt.rc('axes', labelsize = 15)   # x,y축 label 폰트 크기
        plt.rc('figure', titlesize = 15) # figure title 폰트 크기

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.plot(data, c = color, marker = marker, linestyle = linestyle)
        plt.show()
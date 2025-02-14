import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy.fft import fft2, fftshift
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr, spearmanr
import cv2
import torch
import torch.nn.functional as F
from glob import glob
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import math
from scipy import stats
from matplotlib.ticker import FuncFormatter
from scipy.signal import savgol_filter
# from fft_radius_power import image_2freq_power
from matplotlib import rcParams
from collections import namedtuple
from itertools import combinations

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

# trainId2label = { label.trainId : label for label in reversed(labels) }
id2label = { label.id      : label.trainId for label in labels           }
print(id2label)

# 定义一个函数，用于将图像的 ID 映射到训练时使用的 ID
def fromid2label(img):
    img_clone = img.copy()
    for k, v in id2label.items():
        img_clone[img == k] = v
    return img_clone

# 定义一个函数，用于查找指定文件夹下的所有文件（以指定扩展名结尾）
def _find_npy_files(folder_path, endwith='.npy'):
    npy_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(endwith):
                npy_files.append(os.path.join(root, file))
    return npy_files

# 定义一个函数，用于查找指定文件夹下的所有文件（以指定扩展名结尾）
def find_npy_files(folder_path, endwith='.npy'):
    return glob(folder_path + f'/*{endwith}')

# 定义一个类，用于获取语义分割任务中的边缘信息
class GetSemanticEdge(torch.nn.Module):
    def __init__(self, max_pool_k=5, loss=torch.nn.MSELoss()):
        super().__init__()
        self.max_pool_k = max_pool_k  # 最大池化核的大小
        # 定义高斯权重的卷积核，用于对图像进行平滑处理
        gaussian_weight = [
            [1/8, 1/4, 1/8], 
            [1/4, 1/2, 1/4], 
            [1/8, 1/4, 1/8], 
        ]
        self.gaussian_weight = torch.Tensor(gaussian_weight)[None, None]  # 将卷积核转换为张量
        # 定义边缘检测的卷积核，用于提取图像的边缘特征
        edge_weight = [
            [-1, -1, -1], 
            [-1, 8, -1], 
            [-1, -1, -1], 
        ]
        self.edge_weight = torch.Tensor(edge_weight)[None, None]  # 将卷积核转换为张量
        # 将卷积核的权重设置为不可训练参数
        self.gaussian_weight.requires_grad = False
        self.edge_weight.requires_grad = False
        self.loss = loss  # 定义损失函数

    def forward(self, seg_label, ignore=255):
        seg_label = torch.tensor(seg_label)[None, None]  # 转换为张量
        b, c, h, w = seg_label.shape  # 获取张量的形状信息
        seg_label = seg_label.clone()  # 复制输入张量，避免修改原数据
        seg_label[seg_label == 255] = 0  # 将无效类别设置为 0
        # 将卷积核转换为与当前设备一致的张量
        self.gaussian_weight = self.gaussian_weight.to(seg_label.device)
        self.edge_weight = self.edge_weight.to(seg_label.device)
        # 使用具边界填充的操作（形状为 (1)），用于图像的边界填充
        seg_label = nn.ReplicationPad2d(padding=1)(seg_label.float())
        # 使用卷积操作提取边缘特征
        semantic_edge = F.conv2d(seg_label, weight=self.edge_weight.repeat(c, 1, 1, 1), padding=0, groups=c)
        # 使用最大池化操作来突出边缘区域
        semantic_edge = nn.ReplicationPad2d(padding=self.max_pool_k//2)(semantic_edge)
        semantic_edge = F.max_pool2d(semantic_edge, kernel_size=self.max_pool_k, stride=1, padding=0)
        return semantic_edge.squeeze().numpy() > 0  # 返回边缘区域的二值图像

# 定义一个函数，用于进行特征相似度分析（带有边缘检测）
def feat_sim_analysis_with_edge(feat_path, edge_k=5, channel_clip=False, gt_size=(256, 512), debug=True, inter_mode='bilinear', num=11):
    """
    channel_clip: 选择是否仅使用特征的某个阶段（只适用于特征是多阶段拼接的情况）
    """
    print(feat_path)  # 打印特征路径
    print('=' * 30)  # 打印分隔符
    print(f'edge_k={edge_k}, channel_clip={channel_clip}, gt_size={gt_size}, inter_mode={inter_mode}, num={num}')
    print('=' * 30)
    npy_files_list_1 = find_npy_files(feat_path)  # 查找特征文件路径
    npy_files_list_1.sort()  # 对特征文件路径进行排序
    name_list = ['frankfurt_000001_060135_leftImg8bit']  # 定义一个图像名称列表
    # 获取 Cityscapes 数据集中的图像路径
    images = find_npy_files('/home/ubuntu/2TB/dataset/cityscapes/leftImg8bit/val/*/', '.jpeg')
    images.sort()  # 对图像路径进行排序
    # 获取 Cityscapes 数据集中的地面真实标签路径
    gts = glob('/home/ubuntu/2TB/dataset/cityscapes/gtFine/val/*/*labelIds.png')
    gts.sort()  # 对标签路径进行排序

    # 初始化字典，用于存储不同类别的相似度等信息
    sim_dict = {i: [] for i in range(19)}  # 存储类别的内部相似度
    diff_sim_dict = {i: [] for i in range(19)}  # 存储类别的差异相似度
    error_sim_dict = {i: [] for i in range(19)}  # 存储类别的错误相似度
    error_sim_all_dict = {i: [] for i in range(19)}  # 存储类别的总错误相似度
    inter_sim_dict = {i: [] for i in range(19)}  # 存储类别的交叉相似度

    for idx, (gt_path, feat_path) in tqdm(enumerate(zip(gts, npy_files_list_1))):
        if idx >= num:  # 如果处理的索引超过指定数量，停止循环
            break
        # 加载地面真实标签图像
        cls = cv2.imread(gt_path, 0)
        # 将标签图像调整为指定大小
        cls = cv2.resize(cls, (gt_size[1], gt_size[0]), interpolation=cv2.INTER_NEAREST)
        # 将标签图像的 ID 映射为训练时的 ID
        cls = fromid2label(cls)
        # 如果需要使用边缘检测
        if edge_k > 0:
            # 使用拉普拉斯算子计算图像的梯度
            gt_edges_1pix = cv2.Laplacian(cls.astype(np.uint8), cv2.CV_64F)
            gt_edges_1pix[gt_edges_1pix != 0] = 1  # 将非零值设为 1
            # 定义一个膨胀核，用于增强边缘信息
            kernel = np.ones((edge_k, edge_k), np.uint8)
            # 使用膨胀操作突出边缘
            gt_edges = cv2.dilate(gt_edges_1pix, kernel, iterations=1)
            # 忽略无效区域（ID 为 255 的像素）
            ignore = np.zeros(cls.shape)
            ignore[cls == 255] = 1
            # 使用膨胀操作扩展无效区域
            ignore_3pix = cv2.dilate(ignore, np.ones((3, 3), np.uint8), iterations=1)
            # 清除边缘区域中的无效像素
            gt_edges_1pix[ignore_3pix == 1] = 0
            # 再次忽略无效区域
            ignore = cv2.dilate(ignore, kernel, iterations=1)
            gt_edges[ignore == 1] = 0
            # 创建边缘类别图像
            edge_cls = cls.copy()
            edge_cls[gt_edges == 0] = 255
            cls[gt_edges == 1] = 255
        else:
            edge_cls = cls.copy()

        # 将类别和边缘类别转换为张量
        cls = torch.LongTensor(cls)
        edge_cls = torch.LongTensor(edge_cls)
        # 加载特征文件
        feat_npy = np.load(feat_path)
        # 如果需要剪裁通道，仅保留特定通道
        if channel_clip:
            feat_tensor = torch.FloatTensor(feat_npy)[None, :256]
        else:
            # 将特征数据转换为张量
            feat_tensor = torch.FloatTensor(feat_npy)[None, ]
        # 使用插值操作调整特征张量的大小
        feat_tensor = F.interpolate(feat_tensor, size=gt_size, mode=inter_mode)

        # 初始化字典，用于存储各个类别的特征中心
        c_tensor_dict = {i: [] for i in range(19)}
        # 遍历类别的唯一值
        for c in cls.unique():
            c = int(c)
            if c == 255:  # 跳过无效类别
                continue
            if debug:  # 如果处于调试模式，打印类别信息
                print('class:', c)
            # 获取属于该类别的特征
            c_tensor = feat_tensor[:, :, cls == c]
            # 计算该类别特征的中心点
            c_tensor_dict[c] = c_tensor.mean(dim=-1, keepdims=True)
            # 获取边缘区域的特征
            c_edge_tensor = feat_tensor[:, :, edge_cls == c]
            if c_edge_tensor.size(-1) <= 0:  # 如果边缘区域的特征为空，跳过
                continue
            if debug:  # 如果处于调试模式，打印特征形状信息
                print(c_edge_tensor.shape)
                print('c_tensor', c_tensor.shape)
            # 计算边缘区域的特征与类别中心之间的相似度
            c_sim = F.cosine_similarity(c_edge_tensor, c_tensor.mean(dim=-1, keepdims=True), dim=1)
            if debug:  # 如果处于调试模式，打印相似度信息
                print('c_sim:', c_sim.shape)
                print(c_sim.mean())
            # 将相似度信息存储到对应的类别
            sim_dict[c].append(c_sim.flatten())

        # 计算不同类别之间的相似度
        for c1, c2 in combinations([i for i in range(19) if len(c_tensor_dict[i]) > 0], 2):
            distance = F.cosine_similarity(c_tensor_dict[c1], c_tensor_dict[c2], dim=1)
            inter_sim_dict[c1].append(distance.flatten())

        # 计算边缘区域的特征与不同类别之间的相似度误差
        for c in cls.unique():
            c = int(c)
            if c == 255:  # 跳过无效类别
                continue
            if debug:  # 如果处于调试模式，打印类别信息
                print('class:', c)
            c_edge_tensor = feat_tensor[:, :, edge_cls == c]
            intra_c_sim = F.cosine_similarity(c_edge_tensor, c_tensor_dict[c], dim=1)
            diff_c_sim = (intra_c_sim > intra_c_sim).int()
            # 遍历所有其他类别，计算相似度误差
            for c2 in cls.unique():
                c2 = int(c2)
                if c2 == 255:  # 跳过无效类别
                    continue
                if c2 == c:  # 跳过相同类别
                    continue
                c2_tensor = c_tensor_dict[c2]
                c2_sim = F.cosine_similarity(c_edge_tensor, c2_tensor, dim=1)
                diff_c_sim += (c2_sim > intra_c_sim).int()
            # 计算正确像素的数量
            _correct_pixel = int((diff_c_sim == 0).sum())
            _all_pixel = len(diff_c_sim.flatten())
            # 存储错误相似度和总相似度
            error_sim_dict[c].append(_all_pixel - _correct_pixel)
            error_sim_all_dict[c].append(_all_pixel)

    # 计算类别的平均相似度和标准差
    all_cls = []
    all_std = []
    for k, v in sim_dict.items():
        if len(v) > 1:
            cat_v = torch.cat(v, dim=0)
            avg = cat_v.mean()
            std = cat_v.std()
            all_cls.append(avg)
            all_std.append(std)
            print(k, avg.item(), std.item())
    # 输出类别的平均相似度和平均标准差
    if len(all_cls) > 1:
        print('intra class 平均相似度: avg:', sum(all_cls).item()/len(all_cls))
    if len(all_std) > 1:
        print('all avg:', sum(all_std).item()/len(all_std))

    # 计算不同类别之间的交叉相似度
    all_cls = []
    all_std = []
    for k, v in inter_sim_dict.items():
        if len(v) > 1:
            cat_v = torch.cat(v, dim=0)
            avg = cat_v.mean()
            std = cat_v.std()
            all_cls.append(avg)
            all_std.append(std)
            print(k, avg.item(), std.item())
    # 输出交叉相似度的平均值和平均标准差
    if len(all_cls) > 1:
        print('inter class 平均相似度: avg:', sum(all_cls).item()/len(all_cls))
    if len(all_std) > 1:
        print('all avg:', sum(all_std).item()/len(all_std))

    # 计算错误率和正确率
    all_cls_acc = []
    accurate_pixel_count = 0
    pixel_count = 0
    for k, v in error_sim_all_dict.items():
        if len(v) > 1:
            error_pixel = sum(error_sim_dict[k])
            all_pixel = sum(error_sim_all_dict[k])
            pixel_count += all_pixel
            accurate_pixel_count += (all_pixel - error_pixel)
            error_rate = error_pixel / all_pixel
            right_rate = 1.0 - error_rate
            all_cls_acc.append(right_rate)
            print(f'{k} error_rate', error_rate, 'right_rate', right_rate)
    # 输出平均误差率和平均正确率
    print('pixel average', sum(all_cls_acc) / len(all_cls_acc))
    print('category average', accurate_pixel_count / pixel_count)
    return None

# 定义一个函数，用于进行特征融合分析
def feat_fusion_analysis(
        feat_path_list=[],
                  edge_k=5, channel_clip=False, gt_size=(256, 512), debug=True, inter_mode='bilinear', num=11):
    """
    channel_clip: 选择是否仅使用特征的某个阶段（只适用于特征是多阶段拼接的情况）
    """
    # 打印特征路径
    print(feat_path_list)
    print('=' * 30)
    print(f'edge_k={edge_k}, channel_clip={channel_clip}, gt_size={gt_size}, inter_mode={inter_mode}, num={num}')
    print('=' * 30)
    # 查找特征文件路径
    npy_files_list = [find_npy_files(feat_path) for feat_path in feat_path_list]
    for i in npy_files_list:
        print(len(i))
    for i in npy_files_list:
        i.sort()
    # 获取 Cityscapes 数据集中的图像路径
    images = find_npy_files('/home/ubuntu/2TB/dataset/cityscapes/leftImg8bit/val/*/', '.jpeg')
    images.sort()
    # 获取 Cityscapes 数据集中的地面真实标签路径
    gts = glob('/home/ubuntu/2TB/dataset/cityscapes/gtFine/val/*/*labelIds.png')
    gts.sort()

    # 初始化字典，用于存储不同类别的相似度等信息
    sim_dict = {i: [] for i in range(19)}  # 存储类别的内部相似度
    diff_sim_dict = {i: [] for i in range(19)}  # 存储类别的差异相似度
    error_sim_dict = {i: [] for i in range(19)}  # 存储类别的错误相似度
    error_sim_all_dict = {i: [] for i in range(19)}  # 存储类别的总错误相似度
    inter_sim_dict = {i: [] for i in range(19)}  # 存储类别的交叉相似度

    for idx, (gt_path, ) in tqdm(enumerate(zip(gts))):
        if idx >= num:  # 如果处理的索引超过指定数量，停止循环
            break
        # 加载特征文件
        feat_npys = [
            np.load(feat_path) for feat_path in [npy_files_list[i][idx] for i in range(len(npy_files_list))]
        ]
        # 将特征数据转换为张量并调整大小
        feat_tensor = torch.cat([
            F.interpolate(torch.FloatTensor(feat_npy)[None, ], size=gt_size, mode=inter_mode) for feat_npy in feat_npys
        ], dim=1)
        # 加载地面真实标签图像
        cls = cv2.imread(gt_path, 0)
        # 将标签图像调整为指定大小
        cls = cv2.resize(cls, (gt_size[1], gt_size[0]), interpolation=cv2.INTER_NEAREST)
        # 将标签图像的 ID 映射为训练时的 ID
        cls = fromid2label(cls)
        # 如果需要使用边缘检测
        if edge_k > 0:
            # 使用拉普拉斯算子计算图像的梯度
            gt_edges_1pix = cv2.Laplacian(cls.astype(np.uint8), cv2.CV_64F)
            gt_edges_1pix[gt_edges_1pix != 0] = 1  # 将非零值设为 1
            # 定义一个膨胀核，用于增强边缘信息
            kernel = np.ones((edge_k, edge_k), np.uint8)
            # 使用膨胀操作突出边缘
            gt_edges = cv2.dilate(gt_edges_1pix, kernel, iterations=1)
            # 忽略无效区域（ID 为 255 的像素）
            ignore = np.zeros(cls.shape)
            ignore[cls == 255] = 1
            # 使用膨胀操作扩展无效区域
            ignore_3pix = cv2.dilate(ignore, np.ones((3, 3), np.uint8), iterations=1)
            # 清除边缘区域中的无效像素
            gt_edges_1pix[ignore_3pix == 1] = 0
            # 再次忽略无效区域
            ignore = cv2.dilate(ignore, kernel, iterations=1)
            gt_edges[ignore == 1] = 0
            # 创建边缘类别图像
            edge_cls = cls.copy()
            edge_cls[gt_edges == 0] = 255
            cls[gt_edges == 1] = 255
        else:
            edge_cls = cls.copy()

        # 将类别和边缘类别转换为张量
        cls = torch.LongTensor(cls)
        edge_cls = torch.LongTensor(edge_cls)

        # 初始化字典，用于存储各个类别的特征中心
        c_tensor_dict = {i: [] for i in range(19)}
        # 遍历类别的唯一值
        for c in cls.unique():
            c = int(c)
            if c == 255:  # 跳过无效类别
                continue
            if debug:  # 如果处于调试模式，打印类别信息
                print('class:', c)
            # 获取属于该类别的特征
            c_tensor = feat_tensor[:, :, cls == c]
            # 计算该类别特征的中心点
            c_tensor_dict[c] = c_tensor.mean(dim=-1, keepdims=True)
            # 获取边缘区域的特征
            c_edge_tensor = feat_tensor[:, :, edge_cls == c]
            if c_edge_tensor.size(-1) <= 0:  # 如果边缘区域的特征为空，跳过
                continue
            if debug:  # 如果处于调试模式，打印特征形状信息
                print(c_edge_tensor.shape)
                print('c_tensor', c_tensor.shape)
            # 计算边缘区域的特征与类别中心之间的相似度
            c_sim = F.cosine_similarity(c_edge_tensor, c_tensor.mean(dim=-1, keepdims=True), dim=1)
            if debug:  # 如果处于调试模式，打印相似度信息
                print('c_sim:', c_sim.shape)
                print(c_sim.mean())
            # 将相似度信息存储到对应的类别
            sim_dict[c].append(c_sim.flatten())

        # 计算不同类别之间的相似度
        for c1, c2 in combinations([i for i in range(19) if len(c_tensor_dict[i]) > 0], 2):
            distance = F.cosine_similarity(c_tensor_dict[c1], c_tensor_dict[c2], dim=1)
            inter_sim_dict[c1].append(distance.flatten())

        # 计算边缘区域的特征与不同类别之间的相似度误差
        for c in cls.unique():
            c = int(c)
            if c == 255:  # 跳过无效类别
                continue
            if debug:  # 如果处于调试模式，打印类别信息
                print('class:', c)
            c_edge_tensor = feat_tensor[:, :, edge_cls == c]
            intra_c_sim = F.cosine_similarity(c_edge_tensor, c_tensor_dict[c], dim=1)
            diff_c_sim = (intra_c_sim > intra_c_sim).int()
            # 遍历所有其他类别，计算相似度误差
            for c2 in cls.unique():
                c2 = int(c2)
                if c2 == 255:  # 跳过无效类别
                    continue
                if c2 == c:  # 跳过相同类别
                    continue
                c2_tensor = c_tensor_dict[c2]
                c2_sim = F.cosine_similarity(c_edge_tensor, c2_tensor, dim=1)
                diff_c_sim += (c2_sim > intra_c_sim).int()
            # 计算正确像素的数量
            _correct_pixel = int((diff_c_sim == 0).sum())
            _all_pixel = len(diff_c_sim.flatten())
            # 存储错误相似度和总相似度
            error_sim_dict[c].append(_all_pixel - _correct_pixel)
            error_sim_all_dict[c].append(_all_pixel)

    # 计算类别的平均相似度和标准差
    all_cls = []
    all_std = []
    for k, v in sim_dict.items():
        if len(v) > 1:
            cat_v = torch.cat(v, dim=0)
            avg = cat_v.mean()
            std = cat_v.std()
            all_cls.append(avg)
            all_std.append(std)
            print(k, avg.item(), std.item())
    # 输出类别的平均相似度和平均标准差
    if len(all_cls) > 1:
        print('intra class 平均相似度: avg:', sum(all_cls).item()/len(all_cls))
    if len(all_std) > 1:
        print('all avg:', sum(all_std).item()/len(all_std))

    # 计算不同类别之间的交叉相似度
    all_cls = []
    all_std = []
    for k, v in inter_sim_dict.items():
        if len(v) > 1:
            cat_v = torch.cat(v, dim=0)
            avg = cat_v.mean()
            std = cat_v.std()
            all_cls.append(avg)
            all_std.append(std)
            print(k, avg.item(), std.item())
    # 输出交叉相似度的平均值和平均标准差
    if len(all_cls) > 1:
        print('inter class 平均相似度: avg:', sum(all_cls).item()/len(all_cls))
    if len(all_std) > 1:
        print('all avg:', sum(all_std).item()/len(all_std))

    # 计算错误率和正确率
    all_cls_acc = []
    accurate_pixel_count = 0
    pixel_count = 0
    for k, v in error_sim_all_dict.items():
        if len(v) > 1:
            error_pixel = sum(error_sim_dict[k])
            all_pixel = sum(error_sim_all_dict[k])
            pixel_count += all_pixel
            accurate_pixel_count += (all_pixel - error_pixel)
            error_rate = error_pixel / all_pixel
            right_rate = 1.0 - error_rate
            all_cls_acc.append(right_rate)
            print(f'{k} error_rate', error_rate, 'right_rate', right_rate)
    # 输出平均误差率和平均正确率
    print('pixel average', sum(all_cls_acc) / len(all_cls_acc))
    print('category average', accurate_pixel_count / pixel_count)
    return None

# 主函数入口
if __name__ == '__main__':
    # 调用特征相似度分析函数
    feat_sim_analysis_with_edge('/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/visualization/TPAMI2024/SegFormer-CITY/out_freqfusions.0_2_lr/', edge_k=0, channel_clip=True, gt_size=(256, 512), debug=False, inter_mode='nearest', num=50)
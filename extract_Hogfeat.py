from skimage.feature import hog
from utils import rgb2gray
import os
import imageio
from os import listdir
import pandas as pd

# trainsetpath = os.path.join('E:\Ximenzi\Baseline\WeldFault\data\Train', str(0))  # 路径拼接
# file = listdir(trainsetpath)
# print(len(file))

# def creatHogvec(zero):
for j in range(4):
    trainsetpath = os.path.join('E:\Ximenzi\Baseline\WeldFault\data\Train', str(j))  # 路径拼接
    file = listdir(trainsetpath)
    for i in range(len(file)):
        img = imageio.imread(os.path.join(trainsetpath, file[i]))
        gray = rgb2gray(img) / 255.0
        vector = hog(gray, orientations=9, block_norm='L1', pixels_per_cell=[135, 120], cells_per_block=[2, 2],
                 visualize=False, transform_sqrt=True, feature_vector=True)
        # indices_nonzero = np.nonzero(vector != 0)  # 检索非0元素的位置
        # Hognonvec = np.transpose(vector[indices_nonzero]).tolist()  # 提取vector中的非0元素并将array转换为list
        Hogvec = vector.tolist()
        Hogvec = [round(x, 4) for x in Hogvec]  # 对list中的每个元素保留小数点后4位
        Hogvec.insert(0, file[i].split(".")[0])  # 任意位置追加元素
        Hogvec.append(j)  # 末尾追加元素
        data = pd.DataFrame(Hogvec)  # 将list转换为DataFrame格式
        train = pd.DataFrame(data.values.T)
        train.to_csv('train.csv', float_format='%.3f', header=None, index=False, mode='a')  # 去除表头和左侧索引，mode='a'：追加写入数据不会清空之前的数据。



# train = pd.read_csv("./train.csv")
# b = train.iloc[0, 1:-1]
# print(b[100])

# img = imageio.imread(os.path.join('./', '1.jpg'))

# plt.imshow(img, cmap=plt.cm.gray)
# plt.show()

# # 裁剪图像
# cropped = img[214:506, 279:1535]
# cv2.imwrite("./cv_cut_thor.jpg", cropped)

# print(gray.shape)
# vector, hog_image = hog(gray, orientations=9, block_norm='L1', pixels_per_cell=[13, 40], cells_per_block=[2, 2], visualize=True, transform_sqrt=True,
#                         feature_vector=True)

# plt.imshow(hog_image, cmap=plt.cm.gray)
# plt.show()
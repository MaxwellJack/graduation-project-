# -*- coding: utf-8 -*-
# @Author  : Huangcc

import os
import numpy as np
from skimage import feature as skif
from skimage import io, transform
import random
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVR,SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split,cross_val_score
import numpy as np
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.model_selection import train_test_split
import graduation_project.new.read_all as read_all
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.svm import SVC

# # 全局变量
# IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'images')  # images的绝对路径
# POS_IMAGE_DIR = os.path.join(IMAGES_DIR, 'positive')
# NEG_IMAGE_DIR = os.path.join(IMAGES_DIR, 'negative')
# RESIZE_POS_IMAGE_DIR = os.path.join(IMAGES_DIR, 'resize_positive')
# RESIZE_NEG_IMAGE_DIR = os.path.join(IMAGES_DIR, 'resize_negative')
# IMG_TYPE = 'png'  # 图片类型
# IMG_WIDTH = 256
# IMG_HEIGHT = 256
#
#
# def resize_image(file_in, file_out, width, height):
#     img = io.imread(file_in)
#     out = transform.resize(img, (width, height),
#                            mode='reflect')  # mode {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}
#     io.imsave(file_out, out)
#
#
# 全局变量
IMAGES_DIR = r"C:\Users\Administrator\Desktop"  # images的绝对路径
# n4= os.path.join(IMAGES_DIR, 'No4')
n12 = os.path.join(IMAGES_DIR, 'No12')
# n4_one=os.path.join(n4,"ceshiyangben")
# n4_two=os.path.join(n4,"xunlianyangben")
n12_one=os.path.join(n12,"ceshiyangben")
n12_two=os.path.join(n12,"xunlianyangben")
target_name=["jingxiangcidian","kuaizhuangcidian","weixiangcidian","zhenchang"]
two_list=[]
for name in target_name:
    # two_list.append(os.path.join(n4_one, name))
    # two_list.append(os.path.join(n4_two, name))
    two_list.append(os.path.join(n12_one, name))
    two_list.append(os.path.join(n12_two, name))
all_list=[]
all_y=[]
for i in two_list:
    l = list(map(lambda x: os.path.join(i, x), os.listdir(i)))
    all_list=all_list+l
for i in all_list:
    if "jingxiangcidian" in i :
        all_y.append(100)
    if "kuaizhuangcidian" in i :
        all_y.append(101)
    if "weixiangcidian" in i :
        all_y.append(110)
    if "zhenchang" in i :
        all_y.append(111)

def get_hog_data(images_data, hist_size=256, pixels_per_cell=(8, 8), cells_per_block=(3, 3)):
    n_images = images_data.shape[0]
    hist = np.zeros((n_images, hist_size))
    for i in np.arange(n_images):
        # 使用HOG方法提取图像的纹理特征.
        hog = skif.hog(images_data[i], pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)
        # 统计图像的直方图
        # max_bins = int(hog.max() + 1)
        # hist size:256
        # hist[i], _ = np.histogram(hog, normed=True, bins=max_bins, range=(0, max_bins))
        hist[i] = hog

    return hist


def load_images(images_list, width, height):
    data = np.zeros((len(images_list), width, height))  # 创建多维数组存放图片
    for index, image in enumerate(images_list):
        image_data = io.imread(image, as_grey=True)
        data[index, :, :] = image_data  # 读取图片存进numpy数组
    return data
data=load_images(all_list,128,128)
data=get_hog_data(data, hist_size=15876, pixels_per_cell=(8, 8), cells_per_block=(3, 3))
knc = KNeighborsClassifier(n_neighbors=10)
mnb=GaussianNB()
dtc = DecisionTreeClassifier()
svc=SVC(kernel='poly',degree=2,gamma=1,coef0=0)
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                         algorithm="SAMME",
                         n_estimators=5, learning_rate=0.8)
scores1 = cross_val_score(knc, data, all_y, cv=5, scoring='accuracy')#knn
scores2 = cross_val_score(mnb, data, all_y, cv=5, scoring='accuracy')#beiyesi
scores3 = cross_val_score(dtc, data, all_y, cv=5, scoring='accuracy')#juceshu
scores4= cross_val_score(svc, data, all_y, cv=5, scoring='accuracy')#svm
scores5= cross_val_score(mlp, data, all_y, cv=5, scoring='accuracy')#jichengxuexi
scores6= cross_val_score(bdt, data, all_y, cv=5, scoring='accuracy')#bp
print(scores1.mean(),scores2.mean(),scores3.mean(),scores4.mean(),scores5.mean(),scores6.mean())
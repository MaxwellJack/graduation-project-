from skimage.feature import hog
import os
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC,SVR
import numpy as np
from skimage import feature as skif
from skimage import io, transform
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
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

# def get_hog_data(images_data, hist_size=256, pixels_per_cell=(8, 8), cells_per_block=(3, 3)):
#     n_images = images_data.shape[0]
#     hist = np.zeros((n_images, hist_size))
#     for i in np.arange(n_images):
#         # 使用HOG方法提取图像的纹理特征.
#         hog = skif.hog(images_data[i], pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)
#         # 统计图像的直方图
#         # max_bins = int(hog.max() + 1)
#         # hist size:256
#         # hist[i], _ = np.histogram(hog, normed=True, bins=max_bins, range=(0, max_bins))
#         hist[i] = hog
#
#     return hist


def load_images(images_list, width, height):
    data = np.zeros((len(images_list), width, height))  # 创建多维数组存放图片
    for index, image in enumerate(images_list):
        image_data = io.imread(image, as_grey=True)
        data[index, :, :] = image_data  # 读取图片存进numpy数组
    return data
# data=load_images(all_list,128,128)
# X_train, X_test, y_train, y_test = train_test_split(data, all_y, test_size=0.3)
# train_feature= hog(X_train, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3,3))
# test_feature= hog(X_test, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3,3))
s1=[]
s2=[]
s3=[]
s4=[]
s5=[]
s6=[]
for i in [4, 8, 16, 32]:
    for j in [2, 3, 4, 6]:
        if i <= j:
            pass
        else:
            print(i, j)
            data = load_images(all_list, 128, 128)
            feature=[]
            for k in range(len(all_list)):
                per=hog(data[k],orientations=9, pixels_per_cell=(i, i), cells_per_block=(j,j))
                feature.append(per)
            f=np.array(feature)
            X_train, X_test, y_train, y_test = train_test_split(f, all_y, test_size=0.3)
            svr_rbf=SVC(kernel='poly',degree=2,gamma=1,coef0=0)
            # svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
            # 训练和测试
            score = OneVsRestClassifier(svr_rbf, n_jobs=-1).fit(X_train, y_train).score(X_test, y_test)  # n_jobs是cpu数量, -1代表所有
            print("支持向量机")
            print(score)
            s4.append(round(score, 2))
            # #knn
            # knc = KNeighborsClassifier(n_neighbors=10)
            # knc.fit(X_train,y_train)
            # predict = knc.predict(X_test)
            # print("knn")
            # print("accuracy_score: %.4lf" % accuracy_score(predict,y_test))
            # s1.append(round(accuracy_score(predict,y_test),2))
            # # print("Classification report for classifier %s:\n%s\n" % (knc, classification_report(y_test, predict)))
            #朴素贝叶斯
            # mnb = MultinomialNB()
            # mnb=GaussianNB()
            # mnb.fit(X_train,y_train)
            # predict = mnb.predict(X_test)
            # print ("贝叶斯")
            # print("accuracy_score: %.4lf" % accuracy_score(predict,y_test))
            # s2.append(round(accuracy_score(predict, y_test), 2))
            #决策树
            # dtc = DecisionTreeClassifier()
            # dtc.fit(X_train,y_train)
            # predict = dtc.predict(X_test)
            # print("决策树")
            # print("accuracy_score: %.4lf" % accuracy_score(predict,y_test))
            # s3.append(round(accuracy_score(predict, y_test), 2))
            # 集成学习
            from sklearn.ensemble import AdaBoostClassifier
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.datasets import make_gaussian_quantiles

            bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                                     algorithm="SAMME",
                                     n_estimators=10, learning_rate=0.8)
            bdt.fit(X_train,y_train)
            predict = bdt.predict(X_test)
            print("集成学习")
            print("accuracy_score: %.4lf" % accuracy_score(predict, y_test))
            s6.append(round(accuracy_score(predict, y_test), 2))
            # bp网络
            # 构建模型，2个隐藏层，第一个隐藏层有100个神经元，第2隐藏层50个神经元，训练500周期
            from sklearn.neural_network import MLPClassifier

            mlp = MLPClassifier(hidden_layer_sizes=(1000, 700), max_iter=4000)
            mlp.fit(X_train,y_train)
            predict = mlp.predict(X_test)
            print("bp网络")
            print("accuracy_score: %.4lf" % accuracy_score(predict, y_test))
            s5.append(round(accuracy_score(predict, y_test), 2))
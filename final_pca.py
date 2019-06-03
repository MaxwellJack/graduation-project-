from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.model_selection import train_test_split
import graduation_project.new.read_all as read_all
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

testpath12 = r"C:\Users\Administrator\Desktop\No4\ceshiyangben"
trainpath12 = r"C:\Users\Administrator\Desktop\No4\xunlianyangben"
dic12=read_all.read_dic(testpath12,trainpath12)
dic=dic12

# testpath4 = r"C:\Users\Administrator\Desktop\No4\ceshiyangben"
# trainpath4 = r"C:\Users\Administrator\Desktop\No4\xunlianyangben"
# dic4=read_all.read_dic(testpath4,trainpath4)
# dic = {}
# for key in dic4:
#     if dic12.get(key):
#         dic[key] = dic4[key] + dic12[key]
#     else:
#         dic[key] = dic4[key]
# for key in dic12:
#     if dic4.get(key):
#         pass
#     else:
#         dic[key] = dic12[key]
datax=[]
datax=dic["zhenchang"]+dic["jingxiangcidian"]+dic["kuaizhuangcidian"]+dic["weixiangcidian"]
datay=[]
datay=len(dic["zhenchang"])*[111]+len(dic["jingxiangcidian"])*[110]+len(dic["kuaizhuangcidian"])*[101]+len(dic["weixiangcidian"])*[100]

#pca
knn=[0]
beiyesi=[0]
zcxlj=[0]
tree=[0]
jicxx=[0]
bp=[0]
# d={}
# for i in range(len(datax)):
#     pca0 = PCA(n_components=100)
#     data0 = pca0.fit_transform(datax[i].reshape(1, -1)[0])
#     d[i]=data0
for num in [50,100,150,200,300,350,400,500]:
    pca0 = PCA(n_components=num )
    print("pca等于=",num)
    d = pca0.fit_transform(datax)
    X_train, X_test, y_train, y_test = train_test_split(d, datay, test_size=0.3)


    #SVM
    # from sklearn import svm
    # clf = svm.SVC()
    # clf.fit(X=X_train, y=y_train,sample_weight=None)  # 训练模型。参数sample_weight为每个样本设置权重。应对非均衡问题
    # result = clf.predict(X_test)  # 使用模型预测值
    # print(metrics.classification_report(y_test, result,target_names=target_name))
    # print(metrics.confusion_matrix(y_test, result))

    from sklearn.svm import LinearSVC

    # clf = LinearSVC(penalty='l2',C=1,loss='hinge') # 创建线性可分svm模型，参数均使用默认值
    # clf.fit(X_train, y_train)  # 训练模型
    # result = clf.predict(X_test)  # 使用模型预测值
    # target_name=["jingxiangcidian","kuaizhuangcidian","weixiangcidian","zhenchang"]
    # print("svm")
    # print("accuracy_score: %.4lf" % accuracy_score(result,y_test))
    # print(metrics.classification_report(y_test, result,target_names=target_name))
    # print(metrics.confusion_matrix(y_test, result))


    # #knn
    # knc = KNeighborsClassifier(n_neighbors=1, algorithm='auto')
    # knc.fit(X_train,y_train)
    # predict = knc.predict(X_test)
    # print("knn")
    # print("accuracy_score: %.4lf" % accuracy_score(predict,y_test))
    # knn.append(round(accuracy_score(predict,y_test),2))
    # # print("Classification report for classifier %s:\n%s\n" % (knc, classification_report(y_test, predict)))

    # #朴素贝叶斯
    # # mnb = MultinomialNB()
    # mnb=GaussianNB()
    # mnb.fit(X_train,y_train)
    # predict = mnb.predict(X_test)
    # print ("贝叶斯")
    # print("accuracy_score: %.4lf" % accuracy_score(predict,y_test))
    # beiyesi.append(round(accuracy_score(predict,y_test),2))


    # #决策树
    # dtc=DecisionTreeClassifier(criterion='entropy')
    # dtc.fit(X_train,y_train)
    # predict = dtc.predict(X_test)
    # print("决策树")
    # print("accuracy_score: %.4lf" % accuracy_score(predict,y_test))
    # tree.append(round(accuracy_score(predict,y_test),2))
    # # print("Classification report for classifier %s:\n%s\n" % (dtc, classification_report(y_test, predict)))

    #支持向量机
    svc=SVC(kernel='poly',degree=2,gamma=1,coef0=0)
    svc.fit(X_train,y_train)
    predict = svc.predict(X_test)
    print("svc")
    print("accuracy_score: %.4lf" % accuracy_score(predict,y_test))
    zcxlj.append(round(accuracy_score(predict,y_test),2))

    # 集成学习
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=30, min_samples_split=20, min_samples_leaf=5),
                             algorithm="SAMME.R",
                             n_estimators=500, learning_rate=0.8)
    # bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
    #                          algorithm="SAMME",
    #                          n_estimators=5, learning_rate=0.8)
    bdt.fit(X_train,y_train)
    predict=bdt.predict(X_test)
    print("集成学习")
    print("accuracy_score: %.4lf" % accuracy_score(predict, y_test))
    jicxx.append(round(accuracy_score(predict,y_test),2))
    #bp网络
    # 构建模型，2个隐藏层，第一个隐藏层有100个神经元，第2隐藏层50个神经元，训练500周期
    from sklearn.neural_network import MLPClassifier

    mlp = MLPClassifier(hidden_layer_sizes=(2500, 1000), max_iter=700)
    mlp.fit(X_train,y_train)
    predict=mlp.predict(X_test)
    print("bp网络")
    print("accuracy_score: %.4lf" % accuracy_score(predict, y_test))
    bp.append(round(accuracy_score(predict,y_test),2))

    # 测试集准确率的评估
    # predictions = mlp.predict(x_test)

from matplotlib import pyplot
import matplotlib.pyplot as plt
from pylab import *                                 #支持中文
import numpy as np
mpl.rcParams['font.sans-serif'] = ['SimHei']
# names = range(8, 21)
# names = [str(x) for x in list(names)]

x=[0,50,100,150,200,300,350,400,500]
# knn=[0,0.73, 0.67, 0.63, 0.62, 0.62, 0.57, 0.61, 0.6]
# beiyesi=[0, 0.79, 0.77, 0.81, 0.81, 0.79, 0.76, 0.76, 0.77]
# zcxlj=[0,0.89, 0.85, 0.85, 0.83, 0.83, 0.81, 0.8, 0.81]
# tree=[ 0,0.69, 0.68, 0.63, 0.6, 0.63, 0.58, 0.63, 0.64]
# y_train = [0.840, 0.839, 0.834, 0.832, 0.824, 0.831, 0.823, 0.817, 0.814, 0.812, 0.812, 0.807, 0.805]
# y_test = [0.838, 0.840, 0.840, 0.834, 0.828, 0.814, 0.812, 0.822, 0.818, 0.815, 0.807, 0.801, 0.796]
# plt.plot(x, y, 'ro-')
# plt.plot(x, y1, 'bo-')
# pl.xlim(-1, 11)  # 限定横轴的范围
# pl.ylim(-1, 110)  # 限定纵轴的范围


# plt.plot(x, knn, marker='o', mec='r', mfc='w', label='KNN')
# plt.plot(x, beiyesi, marker='*', ms=10, label='贝叶斯')
plt.plot(x, zcxlj, marker='.', ms=10, label='支持向量机')
# plt.plot(x, tree, marker='^', ms=10, label='决策树')
plt.plot(x, jicxx, marker='^', ms=10, label='集成学习')
plt.plot(x, bp, marker='.', ms=10, label='BP网络')
plt.legend()  # 让图例生效

plt.xticks(np.linspace(0,550,11,endpoint=False))
plt.margins(0)
plt.subplots_adjust(bottom=0.10)
plt.xlabel('pca降维')  # X轴标签
plt.ylabel("准确率")  # Y轴标签
pyplot.yticks(np.linspace(0,1,10,endpoint=True))
plt.show()






# -*- encoding: utf-8 -*-
from sklearn.datasets import load_digits
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
digits=load_digits()
from sklearn.model_selection import train_test_split
# fig=plt.figure(figsize=(6,6))
# for i in range(64):
#     ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
#     ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
#     # 用目标值标记图像
#     ax.text(0, 7, str(digits.target[i]))
# plt.savefig('handwriten.png')
X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, test_size=0.25)
#随机森林
rf=RandomForestClassifier()
rf.fit(X_train,Y_train)
rfpre=rf.predict(X_test)
rfrep=metrics.classification_report(rfpre, Y_test)
print(rfrep)
svcModel=svm.LinearSVC()

svcModel.fit(X_train,Y_train)
pre=svcModel.predict(X_test)
rep=metrics.classification_report(pre, Y_test)
print(rep)
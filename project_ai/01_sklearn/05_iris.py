import numpy
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
# 데이터 준비

iris=load_iris()

# x값
# print(iris)
# print(iris.data)
# print(iris.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
##sepal:꽃밫침 #petal 꽃잎


# y값
# print(iris.target)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2]

# print(iris.target_names)
# ['setosa', 'versicolor', 'virginica']



x=iris.data
y=iris.target

'''
df=pd.DataFrame(x)
df.columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df['category']=pd.DataFrame(iris.target_names[y].reshape(-1))
# print(df)

groups = df.groupby("category")
fig, ax = plt.subplots()
for name, group in groups:
    ax.scatter(group.sepal_width,group.petal_width, marker='.', label=name)
ax.legend()


plt.xlabel("sepal_width")
plt.ylabel("petal_width")
plt.show()
'''



#데이터 분할

train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=0.3, random_state=1)

train_y.shape
print(train_y.shape)
# x값 (150,4) // y값 (150,)
# train x값 (105,4) y값 (105,)
# text  x값 (45,4) y값 (45,)

#모델준비

logistic=LogisticRegression()


# 학습
logistic.fit(train_x,train_y)



#예측및 평가.
pred=logistic.predict(test_x)
print(test_x)
print(pred)


# 예측및 평가

plt.plot(test_x,test_y, "b.")
plt.plot(test_x,pred,"r.")

plt.xlim(10,140)
plt.ylim(100,220)
plt.grid()
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.linear_model import LogisticRegression

# 데이터 준비

# 공부,과외-> 합격/불합격
#[1,0]/[0] fail
#[8,0]/[1] pass
# a시간 공부하고, b시간 과외받은 학생은?

x=[
    [1,0],
    [2,0],
    [5,1],
    [2,3],
    [3,3],
    [8,1],
    [10,0]
]
# print(x)
y=[
    [0],
    [0],
    [0],
    [0],
    [1],
    [1],
    [1]
]

# print(y)


# 데이터 분할

train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=0.3, random_state=1)
#
# train_y=train_y.values.reshape(1,-1)
# test_y=test_y.values.reshape(1,-1)




# 모델 준비

logistic=LogisticRegression()






# 학습

logistic.fit(train_x,np.ravel(train_y))





# 예측및 평가
pred=logistic.predict(test_x)
print(test_x)
print(pred)


pred_pass=logistic.predict(([[4,0]]))
print("{}시간 공부,{}시간 과외:{}".format(4,0,"pass" if pred_pass==1 else "fait"))
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# def celsius_to_fahrenheit(x):
#     return x*1.8+32
#
#
# data_c=np.array(range(0,100))
# data_f=celsius_to_fahrenheit(data_c)


#데이터 준비

def celsius_to_fahrenheit(x):
    return x*1.8+32
data_c =np.array(range(0,100))
data_f=celsius_to_fahrenheit(data_c)
# print(data_c)
# print(data_f)






#데이터 분할
x=data_c
y=data_f

train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=0.3, random_state=1)
# train_x=train_x.values.reshape(-1,1)
# test_x=test_x.values.reshape(-1,1)




#모델 준비
linear=LinearRegression()





#학습

linear.fit(train_x.reshape(-1,1),train_y)



#예측 및 평가

predict=linear.predict(test_x.reshape(-1,1))
pred_predict = linear.predict([[40]])
print("온도 예측:", pred_predict)


accuracy=linear.score(test_x.reshape(-1,1),predict)
print("accuracy:", accuracy)
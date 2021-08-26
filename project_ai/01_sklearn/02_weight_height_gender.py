import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 데이터준비

df=pd.read_csv("02_weight_height.csv", encoding="euc-kr")
df=df[["학교명","학년","성별",'키',"몸무게"]]
df.dropna(inplace=True)

df['grade']=list(map(lambda x: 0 if x[-4:]=='초등학교' else(6 if x[-3:]=="중학교" else 9), df["학교명"]))+df["학년"]
df.drop(['학교명','학년'], axis='columns', inplace=True)


df.columns=["gender",'height','weight','grade']

df['gender']=df['gender'].map(lambda x: 0 if x=="남" else 1)
# print(df)

x=df[["weight","gender"]]
y=df["height"]

print(x)

# x와 y의 모형을 보고 동일모양으로 reshape를 정한다


# 데이터 분할

train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=0.3, random_state=1)

train_y=train_y.values.reshape(-1)
test_y=test_y.values.reshape(-1)
# 이번엔 걍 미리 .reshape 함

# reshape 공부 다시해보자..

# 모델 준비
linear=LinearRegression()




#학습

linear.fit(train_x,train_y)


# 예측및 평가
predict=linear.predict(test_x)
pred_grd=linear.predict([[80,0]])
print("키 예측:",pred_grd)


plt.plot(test_x,test_y, "b.")
plt.plot(test_x,predict,"r.")

plt.xlim(10,140)
plt.ylim(100,220)
plt.grid()
plt.show()
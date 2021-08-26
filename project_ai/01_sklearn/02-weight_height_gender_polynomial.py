import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import olynomialFeatures
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


poly=PolynomialFeatures()
x=poly.fit().transform(x)
#[1,a,b,a^2,ab,b^2]>>다항함수>>걍 이차함수..

# 데이터 분할







# 모델 준비





# 합습






# 에측및 평가
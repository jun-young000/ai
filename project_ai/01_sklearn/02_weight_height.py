#pip install pandas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# print(df)
# 공공데이터 교육부 학생건강검사 결과분석 rawdata 서울 2015
# 데이터준비

pd.set_option('display.width',300)
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',30)
df=pd.read_csv("weight_height.csv", encoding="euc-kr")

# print(df)

df=df[["학교명","학년","성별",'키',"몸무게"]]
df.dropna(inplace=True)  # 결측치를 제거한 상태를 함수 사용시에 바로 적용함.이러면 따로 데이터 프레임을 지정해서 저장하지 않고도 바로 적용된다.

# print(df)

# print(df)

# 학년값 1,2,3,4,5,6,1,2,3,1,2,3->1,2,3,4,5,6,7,8,9,10,11,12
# 초등학교 0 중학교 6 고등학교 9

df['grade']=list(map(lambda x: 0 if x[-4:]=='초등학교' else(6 if x[-3:]=="중학교" else 9), df["학교명"]))+df["학년"]
# x[-4:].>>>>> xxxx초등학교이니깐 뒷자리 4부터..끝까지라는 소리 헷갈렷다.
# 람다는 elif 는 안되요~!!

# 위 람다식을 반복문의로 만들어 보자!!
# def year(x):
#     for i in df['학교명']:
#         if str(i).find('중학교'):
#             df[(df['학년'] == "1")] = '7'
#             df[(df['학년'] == "2")] = '8'
#             df[(df['학년'] == "3")] = '9'
#         elif str(i).find('고등학교'):
#             df[(df['학년'] == "1")] = '10'
#             df[(df['학년'] == "2")] = '11'
#             df[(df['학년'] == "3")] = '12'
#         else:
##### 마무리 못함
#
#
# print(df)

df.drop(['학교명','학년'], axis='columns', inplace=True)
df.columns=["gender",'height','weight','grade']
# DROP (axis=0) 각 열의 모든행에 대해서 동작한다는 소리
# 각열의 행을 지운다라고 생각하면되겠지~!!
# print(df)

# df['gender'] 의 값을, 남>0 여>1 로 변환


#1번#
df['gender']=df['gender'].map(lambda x: 0 if x=="남" else 1)

#2번# df['gender']=list(map(lambda x: 0 if x=="남" else 1,df['gender']))


#3번# df['gender']=[0 if i=="남" else 1 for i in df['gender']]

#4번 # df['gender']= df['gender'].map({"남":0,"여":1})>>>>흐으음...


#5번# df.loc[df['gender']=='남','gender']='0'
# df.loc[df['gender']=='여','gender']='1'
# df.loc[condition, column_label] = new_value
# condition: 이 매개 변수는 조건을 참으로 만드는 값을 반환합니다.
# column_label: 업데이트 할 대상 열을 지정하는 데 사용되는이 매개 변수
# 더 쉽게 말하면 매개 변수를 통해 값을 결정한 후 new_value로 업데이트합니다.


is_boy=df['gender']==0

boy_df, girl_df = df[is_boy], df[~is_boy]
print(boy_df)
print(girl_df)


# 데이터 분할
x=boy_df['weight']
y=boy_df['height']

# train / test set 분리
train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=0.3, random_state=1)
train_x=train_x.values.reshape(-1,1)
test_x=test_x.values.reshape(-1,1)


# 모델 준비
linear=LinearRegression()

# 학습

linear.fit(train_x,train_y)


# 예측및 평가

predict=linear.predict(test_x)
# print(test_x)
# print(predict)
# print(test_y)


# 그래프
plt.plot(test_x,test_y, "b")
plt.plot(test_x,predict,"r")

plt.xlim(10,140)
plt.ylim(100,220)
plt.grid()
# plt.show()

# 몸무게 80킬로의 키를 에측시에

pred_grd = linear.predict([[80]])
print("남학생 예측키:", pred_grd)
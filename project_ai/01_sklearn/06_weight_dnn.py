import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor






# 1.데이터준비
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
x=poly.fit(x).transform(x)

# 2.데이터 분할
train_x, train_y, test_x , test_y =train_test_split(x,y , test_size=0.3, random_state=1)




# 3. 모델 준비
# hyperparameter tuning
MLP=MLPRegressor(hidden_layer_sizes=(300), random_state=1)
#MLP=MLPRegressor(idden_layer_sizes=(300,) 300으로 하면 오히려 학습효과 떨어짐.오버히팅 문제걸림.

# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor
# 하나씩 연습해보자.먼뜻인지 몰라도..

# 4. 학습
MLP.fit(train_x,train_y)


# 5. 예측 및 평가

#
predict=MLP.predict(test_x)

print("키 예측:",predict)

accracy=MLP.score(test_x,test_y)
print(accracy)
# plt.plot(test_x,test_y, "b.")
# plt.plot(test_x,predict,"r.")
#
# plt.xlim(10,140)
# plt.ylim(100,220)
# plt.grid()
# plt.show()
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

'''
서포트 벡터 머신(SVM, Support Vector Machine)이란 주어진 데이터가
어느 카테고리에 속할지 판단하는 이진 선형 분류 모델
무조컨 마진을 최대하하는게 목적이아니라 데이터를 정확하게 분류후 마진을 최대화한다고 생각해보자.

SVM이란?

SVM은 분류에 사용되는 지도학습 머신러닝 모델이다.
SVM은 서포트 벡터(support vectors)를 사용해서 결정 경계(Decision Boundary)를 정의하고, 
분류되지 않은 점을 해당 결정 경계와 비교해서 분류하게 된다.
기존의 퍼셉트론은 가장 단순하고 빠른 분류 모형이지만 결정경계가 유일하게 존재하지 않는다.
서포트 벡터 머신(SVM)은 퍼셉트론 기반의 모형에 가장 안정적인 결정 경계를 찾기 위해 제한 조건을 추가한 
모형이라고 볼 수 있다.
서포트 벡터 : 클래스 사이 경계에 가깝게 위치한 데이터 포인트 (결정 경계와 이들 사이의 거리가 
SVC 모델의 dual_coef_에 저장된다.)

커널 기법

데이터셋에 비선형 특성을 추가하면 선형 모델을 더 강력하게 만들 수 있음
하지만, 어떤 특성을 추가해야 할지 알 수 없고, 특성을 많이 추가하면 연산 비용이 커진다.
커널 기법 : 새로운 특성을 만들지 않고 고차원 분류기를 학습시킬 수 있도록 한다. 
주어진 데이터를 고차원의 특징 공간으로 사상해, 원래의 차원에선 포이지 않던 선형(초평면)이
데이터를 분류할 수 있도록 한다. 어렵다면 2차원을 3차원으로 바꾼다 이리생각해보자.
!!
 
고차원 공간 맵핑 방법 : 가우시안 커널, RBF (Radial Basis Function) 커널

# 초평면~!!!!!!!!!!!!!!!!!!!!!!!!!

정리해보자
Support Vector Machine (SVM)
- Margin이 최대화가 되는 결정 경계(초평면)를 정의

- Hard Margin SVM : 이상치(Outlier)를 허용하지 않음 (overfitting)
- Soft Margin SVM : 이상치를 어느정도 허용 (underfitting)

- Kernel Trick : 차원을 추가하여 분류


'''

#데이터준비

iris=load_iris()
x=iris.data
y=iris.target


#데이터분할

train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=0.3, random_state=1)



#모델준비

svm=SVC(kernel="linear")


#학습

svm.fit(train_x,train_y)



#예측및 결과

pred=svm.predict(test_x)
# print(test_x)
# print(pred)


for i in range(len(test_x)):
    print(test_x[i],"-> 예측:", iris.target_names[pred[i]],"\t 실제:",iris.target_names[test_y[i]])

accuracy=svm.score(test_x,test_y)
print(accuracy)
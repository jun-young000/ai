from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
'''
K-최근접 이웃(K-Nearest Neighbor)은 머신러닝에서 사용되는 분류(Classification) 알고리즘이다.
유사한 특성을 가진 데이터는 유사한 범주에 속하는 경향이 있다는 가정하에 사용한다.

고민할점
1. 정규화(Normalization)
2. K 개수 선택
k가 너무 작을 때 : Overfitting
k가 너무 클 때 : Underfitting

KNN 알고리즘은 n_neighbors의 갯수가 적을때는 결정 경계가 매우 복잡해지면, 반대로 n_neighbors의 갯수가 많아지면 결정 경계가 단순해집니다.

n_neighbors 적어지면 -> model의 결정경계가 복잡 -> overfitting
n_neighbors 많아지면 -> model의 결정경계가 단순 -> underfitting

장점

쉬운 모델, 쉬운 알고리즘과 이해 (입문자가 샘플데이터를 활용할 때 좋음)
튜닝할 hyperparameter 스트레스가 없음
초기 시도해보기 좋은 시작점이 되는 모델
단점

샘플 데이터가 늘어나면 예측시간도 늘어나기 때문에 매우 느려짐
pre-processing을 잘하지 않으면 좋은 성능을 기대하기 어려움
feature가 많은(수 백개 이상) 데이터셋에서는 좋은 성능을 기대하기 어려움
feature의 값이 대부분 0인 데이터셋과는 매우 안좋은 성능을 냄
'''


#데이터준비

iris=load_iris()

x=iris.data
y=iris.target


#데이터분할

train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=0.3, random_state=1)



#모델준비

KNN=KNeighborsClassifier()


#학습

KNN.fit(train_x,train_y)



#예측및 결과

pred=KNN.predict(test_x)
# print(test_x)
# print(pred)


for i in range(len(test_x)):
    print(test_x[i],"-> 예측:", iris.target_names[pred[i]],"\t 실제:",iris.target_names[test_y[i]])

accuracy=KNN.score(test_x,test_y)
print(accuracy)
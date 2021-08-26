from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

'''
결정트리

Decision Tree는 Random Forest Ensemble 알고리즘의 기본이 되는 알고리즘이며,
 Tree 기반 알고리즘입니다. 의사결정나무 혹은 결정트리로 불리우는 이 알고리즘은 
 머신러닝의 학습 결과에 대하여 시각화를 통한 직관적인 이해가 가능하다는 것이 큰 장점입니다. 
 더불어, Random Forest Ensemble 알고리즘은 바로 이 Decision Tree 알고리즘의 
 앙상블 (Ensemble) 알고리즘인데, Random Forest 앙상블 알고리즘이 사용성은 쉬우면서 
 성능까지 뛰어나 캐글 (Kaggle.com)과 같은 데이터 분석 대회에서 Baseline 알고리즘으로 
 많이 활용되고 있습니다.


분할과 가지치기 과정을 반복하면서 모델을 생성한다.
결정트리에는 분류와 회귀 모두에 사용할 수 있다.
여러개의 모델을 함께 사용하는 앙상블 모델이 존재한다. (RandomForest, GradientBoosting, XGBoost)
각 특성이 개별 처리되기 때문에 데이터 스케일에 영향을 받지 않아 특성의 정규화나 표준화가 필요 없다.
시계열 데이터와 같이 범위 밖의 포인트는 예측 할 수 없다.
과대적합되는 경향이 있다. 이는 본문에 소개할 가지치기 기법을 사용해도 크게 개선되지 않는다.

결정트리 or 의사결정나무 (Decision Tree)
결정트리를 가장 단수하게 표현하자면, Tree 구조를 가진 알고리즘입니다.

의사결정나무는 데이터를 분석하여 데이터 사이에서 패턴을 예측 가능한 규칙들의 조합으로 나타내며,
이 과정을 시각화 해 본다면 마치 스무고개 놀이와 비슷합니다.

'''

#데이터준비

iris=load_iris()
x=iris.data
y=iris.target


#데이터분할

train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=0.3, random_state=1)



#모델준비

D_TREE=DecisionTreeClassifier()


#학습

D_TREE.fit(train_x,train_y)



#예측및 결과

pred=D_TREE.predict(test_x)
# print(test_x)
# print(pred)


for i in range(len(test_x)):
    print(test_x[i],"-> 예측:", iris.target_names[pred[i]],"\t 실제:",iris.target_names[test_y[i]])

accuracy=D_TREE.score(test_x,test_y)
print(accuracy)
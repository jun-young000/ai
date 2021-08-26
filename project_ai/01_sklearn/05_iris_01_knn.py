from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
'''
K-Nearest Neighbors (KNN)

가장 고전적이고 직관적인 머신러닝 분류 알고리즘
K-최근접 이웃(K-Nearest Neighbor)은 머신러닝에서 사용되는 분류(Classification) 알고리즘이다.
유사한 특성을 가진 데이터는 유사한 범주에 속하는 경향이 있다는 가정하에 사용한다.
기하학적 거리 분류기
가장 가깝게 위치하는 멤버로 분류하는 방식


가장 중요한 hyperparameter인 K값은 근처에 참고(reference)할 이웃의 숫자

- 입력값에서 가장 가까운 k개의 데이터를 비교
- k개 중 가장 많은 class로 분류
- 일반적으로 k는 홀수 (짝수인 경우, 동점이 되어 분류 불가 가능성)
'''
'''
KNeighborsClassifier(n_neighbors, weights, algorithm, leaf_size, p, metric, metric_params, n_jobs)
n_neighbors : 이웃의 수 (default : 5)
weights : 예측에 사용된 가중 함수 (uniform, distance) (default : uniform / uniform : 가중치를 동등하게 설정, distance : 가중치를 거리에 반비례하도록 설정)
algorithm : 가까운 이웃을 계산하는데 사용되는 알고리즘 (auto, ball_tree, kd_tree, brute)
leaf_size : BallTree 또는 KDTree에 전달 된 리프 크기
p : (1 : minkowski_distance, 2: manhattan_distance 및 euclidean_distance)
metric : 트리에 사용하는 거리 메트릭스
metric_params : 메트릭 함수에 대한 추가 키워드 인수
n_jobs : 이웃 검색을 위해 실행할 병렬 작업 수

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
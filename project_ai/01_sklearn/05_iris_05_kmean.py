import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

'''
군집화!!-비지도 학습-문제는 있고 답은 없다.
'''
'''
K - 평균 (K - means)

구현이 쉽고, 다른 군집 알고리즘에 비해 효율이 좋아 인기가 많은 알고리즘이다.학계와 산업현장 모두에서 활용된다.
프로토 타입 기반 군집(각 클러스터가 하나의 프로토타입으로 표현됨)에 속한다.
프로토 타입 : 연속적 특성에서는 비슷한 데이터 포인트의 centroid(평균) / 범주형 특성에서는 medopid(가장 자주 등장하는 포인트)
몇개의 군집을 설정할 것인지 정해야한다는 다소 주관적인 사람의 판단이 개입된다. (K값 결정)

K - 평균의 과정

1) 데이터 포인터에서 랜덤하게 k개의 센트로이드를 초기 클러스터 중심으로 선택한다.
* 오늘의 영단어 centroid:중심
2) 각 데이터를 가장 가까운 센트로이드에 할당한다.
3) 할당된 샘플들의 중심으로 센트로이드를 이동시킨다.
4) 클러스터 할당이 변하지 않거나, 사용자가 지정한 허용오차나 최대 반복횟수에 도착할 때 까지 두번째와 
세번째 과정을 반복한다.
'''
'''
유사도 측정 : 임의의 차원 공간의 두 데이터 포인트 x와 y 사이의 유클리디안 거리,
유클리디안 거리 제곱 지표를 기반하여 '최적화 문제'로 k평균 알고리즘을 설명할 수 있다. 
즉, k-means도 최적화 문제라고 할 수 있다.
클러스터 내 제곱 오차합(SSE = 센트로이드와 클러스터 내 데이터의 거리 제곱합)를 계산하여 
이를 반복적으로 최소화시킨다. 각 데이터를 가장 가까운 센트로이드(클러스터)에 할당하면,
센트로이드는 이동되고 SSE는 다시 계산된다. 이런 과정에서 SSE가 일정 수준 내로 들어온다면 
클러스터가 변화하지 않는다는 것이고, 최적화가 완료되었음을 의미한다.
각 데이터들의 변동폭이 크다면, '왜곡'이 일어날 가능성이 높다. 이러한 왜곡을 줄이기 위해서는
거리 산출시 불필요한 항목간의 특성을 제거하고 단위를 일치시키는 '표준화' 과정을 진행하면 
좀 더 좋은 결과를 가져올 수 있다.

km = KMeans(n_clusters = 3, init = 'random', n_init = 10, max_iter = 300, tol = 1e-04, random_state = 0)
     KMeans(클러스터개수, 초기 중심좌표설정방법, 초기설정 시 가장 작은 SSE값 찾는 횟수, 데이터 추가 후 센트로이드 이동 횟수, SSE 허용 오차값, 랜덤 설정값)
초기 센트로이드 위치를 랜덤하게 설정하기 때문에, 잘못 설정된다면 클러스터 성능이 매우 불안정해진다.
이를 위해 초기 센트로이드를 더 똑똑하게 할당할 수 있는 방법 등장 (K-means++)
또한 K-평균 방법은 데이터 당 클러스터가 중복되지 않고, 계층적이지 않다. 그리고 클러스터 당 하나 이상의 데이터는 있다는 '가정'도 존재한다.
이러한 특징과 가정 극복이 어렵다. 때문에 더더욱 초기 센트로이드를 잘 설정해야 한다.
'''
#데이터준비

iris=load_iris()
x=iris.data
y=iris.target

# k_means 은 답이 없으니 y값이 필요없으나 여기서는 쓰던습관대로 쓰지만 실제 할때는 y값 안쓴다
# 즉 아말은 밑에 train_test_split설정시 train_y, test_y 도 안쓴다는 소리

#데이터분할

train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=0.3, random_state=1)



#모델준비

mean= KMeans(n_clusters=3)


#학습

mean.fit(train_x)



#예측및 결과

pred=mean.predict(test_x)

'''
# 만약 실제 x값을 넣는다면.!! ex=[5.0,3.1,0.2,1.0]
pred=mean.predict(test_x)
predic=mean.prefict(np.array([5.0,3.1,0.2,1.0], dtype=np.flote64).reshape(1,-1))
print(predic)
'''

df=pd.DataFrame(test_x)
df.columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df["category"]=pd.DataFrame(pred)


centers=pd.DataFrame(mean.cluster_centers_)
centers.columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
centers_x=centers["sepal_length"]
centers_y=centers["sepal_width"]
#
#
plt.scatter(df["sepal_length"],df['sepal_width'],c=df["category"])
plt.scatter(centers_x,centers_y,s=100,c="r",marker="*")
plt.show()



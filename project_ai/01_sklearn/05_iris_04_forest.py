from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

'''
Random Forest
- 여러 개의 Decision Tree를 연결 (앙상블)
- Bagging(Bootstrap AGGregatING) : 임의의 부분집합(forest)
 > 각 decision tree의 결과를 더해서 분류

앙상블 알고리즘은 방법론 적인 측면에서 Voting, Bagging, Boosting 알고리즘등으fh,
앙상블의 앙상블 알고리즘인 Stacking 그리고 Weighted Blending 등의 기법. 
앙상블 알고리즘은 단일 알고리즘 대비 성능이 좋기 때문에, 캐글(Kaggle.com)과 같은 
데이터 분석 대회에서도 상위권에 항상 꼽히는 알고리즘 들입니다.

머신러닝 앙상블이란 여러개의 머신러닝 모델을 이용해 최적의 답을 찾아내는 기법이다.
여러 모델을 이용하여 데이터를 학습하고, 모든 모델의 예측결과를 평균하여 예측하는 

앙상블 기법의 종류 (Ensemble : 여러 개의 모델을 사용하여 결과 도출)

보팅 (Voting): 투표를 통해 결과 도출
배깅 (Bagging): 샘플 중복 생성을 통해 결과 도출.
(쉽게 말하면 같은 모델을 여러 개 병렬!!!로 실행하여 선형 결합)
부스팅 (Boosting): 이전 오차를 보완하면서 가중치 부여
(쉽게 말하면 가벼운 모델을 순차적으로 학습하여 결과 도출)

스태킹 (Stacking): 여러 모델을 기반으로 예측된 결과를 통해 meta 모델이 다시 한번 예측
https://teddylee777.github.io/scikit-learn/scikit-learn-ensemble
'''
#데이터준비

iris=load_iris()
x=iris.data
y=iris.target


#데이터분할

train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=0.3, random_state=1)



#모델준비

FOREST=RandomForestClassifier()


#학습

FOREST.fit(train_x,train_y)



#예측및 결과

pred=FOREST.predict(test_x)
# print(test_x)
# print(pred)


for i in range(len(test_x)):
    print(test_x[i],"-> 예측:", iris.target_names[pred[i]],"\t 실제:",iris.target_names[test_y[i]])

accuracy=FOREST.score(test_x,test_y)
print(accuracy)
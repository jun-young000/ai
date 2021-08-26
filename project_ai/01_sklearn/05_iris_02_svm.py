from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

'''
서포트 벡터 머신(SVM, Support Vector Machine)이란 주어진 데이터가
어느 카테고리에 속할지 판단하는 이진 선형 분류 모델
무조컨 마진을 최대하하는게 목적이아니라 데이터를 분류후 마진을 최대화한다고 생각해보자.
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
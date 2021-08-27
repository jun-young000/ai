import tensorflow.compat.v1 as tf

# tf 1.x 선형 회귀코드

# 1.데이터 준비
# 2.데이터 분할


x= tf.placeholder(tf.float32)
y= tf.placeholder(tf.float32)
"""
대이터를 나중에 던져줄 거기 때문에 미리 공간만 확보해 놓는다고 생각하면 편하다.
scalar 값을 던져 주기 때문에 shape은 [None] 몇개가 들어와도 상관 없기 때문에.쓰던 안쓰던.
X = tf.placeholder(tf.float32, shape = [None])
Y = tf.placeholder(tf.float32, shape = [None])
 쓰면 요런 느낌.
"""


#3,모델??준비

# 1.가설 설정
# H=Weight * X + bias
w=tf.Variable(tf.random_normal([1]), name='weight')
b=tf.Variable(tf.random_normal([1]), name='bias')
H = w * x + b   #H=hypothesis  # # 가설(모델) 제작 : 아마도 1차원의 분포를 따를 것이다.
'''
W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')
가중치와 바이어스를 설정
variable를 만드는데 값을 random_normal 정규화 분포의 값으로 무작위로 제작, 모양은 스칼라 이기 때문에 1
'''
'''
계산 그래프 준비 : 
tf는 계산그래프를 구축하는 것에서 시작해, 만들어진 계산그래프를 다시 돌리는 과정이 필요.
고로, 변수 Variable을 사용하여 연산자를 이용한 계산그래프 생성.
hypothesis는, 실제 연산 결과라기보다는, 식 자체를 가지고 있는 상황.
함수와 같음.
'''

#loss funtion(cost function)
loss=tf.reduce_mean(tf.square(H-y))

'''
# 예측에서 실제 y값을 빼면 = 오차
오차를 제곱해서 평균을 구하면 2차원의 함수가 제작된다. 
제곱하는 이유는 절대값을 사용할 수도 있지만 음수를 방지하기 위해 사용하는 것 같다.
reduce_mean() : 평균 구하는 메소드
square() : 제곱을 구하는 메소드
오차(=편차) 제곱의 평균 어디서 많이 들어봤지!!!
'''

# optimizer
# 경사 하강법( Gradient descent): loss funtion이 최소가 되도록!!
# learning rate
optimizer=tf.train.GradientDescentOptimizer(0.01)
train=optimizer.minimize(loss)
'''
tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(loss)
경사 하강법을 통하여 cost를 최소화 시킨다. 하강하는 범위 = learning_rate
약간 다르긴 한데 하강하는 범위를 설정가능함
'''
'''
손실함수는 오차제곱법에, 학습 optimizer는, 경사하강법. 학습률은 0.01
학습은, optimizer에서, loss값을 줄이는 방향으로 실행.
(이제껏 구축한 Variable과 같은 텐서플로 그래프가 아직은 계산되지 않는다는 것을 유의)
'''


# Session
sess=tf.Session()

# 변수 초기화
sess.run(tf.global_variables_initializer())
'''
Session 객체를 생성.  tf.~는 계산그래프로 분석머신의 구조를 만들었으면,
이 sess 객체로 실행을 시켜야 함.
tf.global_variables_initializer()을 통하여 변수를 초기화 시킨다.
runs은 텐서플로 객체를 실행해줌.
run()메소드를 사용하여 그래프를 실행한다.
'''

#4.학습
# 학습 횟수:epochs

epochs=5000

for step in range(epochs):
    tep, loss_val, w_val , b_val =sess.run([train, loss, w, b], feed_dict={x:[1,2,3,4,], y:[3,5,7,9]})
    if step% 500==0:
        print("w:{} \t b: {} \t loss:{}".format( w_val , b_val, loss_val))

'''
미리 정해진 그래프로, 변수들(가중치와 편향)을 초기화 시키고,
에폭을 5000로 하여 미리 만든 학습 그래프를 실행시킴.
대략 500번에 한번씩 학습 상황을 로그함.

'''

# 5.예측및 평가
print(sess.run(H, feed_dict={x:[10,11,12,13]}))

'''
- 간단한 선형회귀.
이 버전의 가장 큰 특징은,
계산그래프와 session을 분리한다는 것.
'''
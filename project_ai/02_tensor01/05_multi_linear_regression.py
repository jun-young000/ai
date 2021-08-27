import tensorflow.compat.v1 as tf



# 1.데이터 준비
# 2.데이터 분할
# 쪽지 시험 3번 점수 / 실제 평가 결과
train_x=[
    [73,80,75],
    [93,88,93],
    [89,91,90],
    [96,89,100],
    [73,66,70]
]
train_y=[
    [80],
    [91],
    [88],
    [94],
    [61]
]

# 형태: 변수의 종류(쪽지시험 3번)은 변하지 않음!!


x= tf.placeholder(shape = [None,3],dtype=tf.float32)
y= tf.placeholder(shape = [None,1],dtype=tf.float32)

'''
몇명이 시험볼지 모르니 none 단 쪽지시험은 3번 볼테니 3 즉 shape=[None,3]이 됨
'''


# 3,준비
#가설 설정

w=tf.Variable(tf.random_normal([3,1]), name='weight')
# 행렬곱이라서 >3*1 이라서 random_normal([3,1]) 이라고 함  # 3행 1열!!
b=tf.Variable(tf.random_normal([1]), name='bias')

H = tf.matmul(x,w)+ b

#loss funtion(cost function)

loss=tf.reduce_mean(tf.square(H-y))

# optimizer
# Gradient descent
learning_rate = 0.00004
optimizer=tf.train.GradientDescentOptimizer(learning_rate)
train=optimizer.minimize(loss)

'''
learning_rate = 0.00005 로 해버리면 발산~~즉 무한데가 되버림...
다른말로 간격이 너무 크다는 소리이다.

learning_rate 와  epochs 를 조정해서  loss의 최저값을 찾도록 해야한다.

'''

# Session

sess=tf.Session()
sess.run(tf.global_variables_initializer())

# 4.학습

epochs=100000

for step in range(epochs):
    _, loss_val, w_val , b_val =sess.run([train, loss, w, b], feed_dict={x: train_x, y:train_y})
    if step% 500==0:
        print("w:{} \t b: {} \t loss:{}".format( w_val , b_val, loss_val))
#  #  _, 처음값 의미 없어서 !!! 언더바 처리함.


# 5.예측및 평가


print(sess.run(H, feed_dict={x:[[100,80,87]]}))
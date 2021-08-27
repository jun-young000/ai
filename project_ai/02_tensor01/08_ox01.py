import tensorflow.compat.v1 as tf


# 1.데이터 준비
# 2.데이터 분할

train_x=[
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]

train_y=[
    [0],
    [1],
    [1],
    [0]
]

x= tf.placeholder(shape = [None,2],dtype=tf.float32)
y= tf.placeholder(shape = [None,1],dtype=tf.float32)


# 3.준비
#가설 설정
w=tf.Variable(tf.random_normal([2,1]), name='weight')
b=tf.Variable(tf.random_normal([1]), name='bias')

#loss funtion(cost function)

logit= tf.matmul(x,w)+ b
H=tf.sigmoid(logit)

"""
# 합불이니 이진탐색알고리즘인 sigmoid 사용
"""

loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=y))
"""
x값이 하나가 아니니깐 cross를 쓴다.
"""

#optimizer(Gradient descent)
train=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
"""
optimizer=tf.train.GradientDescentOptimizer(0.01)
train=optimizer.minimize(loss)
"""
#Session

sess=tf.Session()
sess.run(tf.global_variables_initializer())


# 4.학습

epochs=30000

for step in range(epochs):
    _, loss_val =sess.run([train,loss], feed_dict={x: train_x, y:train_y})
    if step% 300==0:
        print("loss:{}".format( loss_val))


# 5.예측및 평가

predict=tf.cast(H>0.5,dtype=tf.float32 ) # sigmoid 특성상 y값이 0.5 이상이면 1 아니면 0임
correct=tf.cast(tf.equal(predict,y),dtype=tf.float32) #예측 돌려보면 예측값 낮음
accuracy=tf.reduce_mean(correct)  #정확성

print("accuracy:", sess.run(accuracy, feed_dict={x:train_x, y:train_y}))
# print("예측:", sess.run(H,feed_dict={x:[[1,1]]}))
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
#가설 설정(딮런링시작)
#입력충
w1=tf.Variable(tf.random_normal([2,10]), name='weight1')
b1=tf.Variable(tf.random_normal([10]), name='bias1')
layer1=tf.sigmoid(tf.matmul(x,w1)+b1)
#히든층
w2=tf.Variable(tf.random_normal([10,20]), name='weight2')
b2=tf.Variable(tf.random_normal([20]), name='bias2')
layer2=tf.sigmoid(tf.matmul(layer1,w2)+b2)  #이전 값을 받아와야함

#히든층

w3=tf.Variable(tf.random_normal([20,10]), name='weight3')
b3=tf.Variable(tf.random_normal([10]), name='bias3')
layer3=tf.sigmoid(tf.matmul(layer2,w3)+b3)


#출력층
w4=tf.Variable(tf.random_normal([10,1]), name='weight4')  # 왜 1이나면 0or1이 나와야 하니깐.
b4=tf.Variable(tf.random_normal([1]), name='bias4')
logits=tf.matmul(layer3,w4)+b4
H=tf.sigmoid(logits)


#loss funtion(cost function)

loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))

#optimizer(Gradient descent)

train=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)


#Session
sess=tf.Session()
sess.run(tf.global_variables_initializer())



# 4.학습

epochs=10000

for step in range(epochs):
    _, loss_val =sess.run([train,loss], feed_dict={x: train_x, y:train_y})
    if step% 1000==0:
        print("loss:{}".format(loss_val))

# 5.예측및 평가

print("예측:", sess.run(H, feed_dict={x:train_x}))

predict=tf.cast(H>0.5,dtype=tf.float32 ) # sigmoid 특성상 y값이 0.5 이상이면 1 아니면 0임
correct=tf.cast(tf.equal(predict,y),dtype=tf.float32) #예측 돌려보면 예측값 낮음
accuracy=tf.reduce_mean(correct)  #정확성

print("accuracy:", sess.run(accuracy, feed_dict={x:train_x, y:train_y}))

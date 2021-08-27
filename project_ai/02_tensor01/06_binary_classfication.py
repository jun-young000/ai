import tensorflow.compat.v1 as tf

#logistic regression


# 1.데이터 준비
# 2.데이터 분할
# [1,0]: 1시간 공부/0시간 과외-> [0]: fail
# [8,1]: 8시간 공부/1시간 과외-> [1]: pass

train_x=[
    [1,0],
    [2,0],
    [5,1],
    [2,3],
    [3,3],
    [8,1],
    [10,0]
]

train_y=[
    [0],
    [0],
    [0],
    [0],
    [1],
    [1],
    [1]
]

x= tf.placeholder(shape = [None,2],dtype=tf.float32)
y= tf.placeholder(shape = [None,1],dtype=tf.float32)

# 3.준비

#가설 설정
w=tf.Variable(tf.random_normal([2,1]), name='weight')
b=tf.Variable(tf.random_normal([1]), name='bias')   # [1,1]로 하면 안된다~!!조심하자...
                                                    # [1]는 "b"가 스칼라(상수) 이기때문에.!!
                                                    # 좀더 정확하게 한다면 bias가 하나이기 때문이다.!!
logit= tf.matmul(x,w)+ b
# activation funtion: sigmoid 로직 사용(0또는1)->>(합아니면 불)
H=tf.sigmoid(logit)

#loss funtion(cost function)

loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=y))
# sigmoid_cross_entropy_with_logits p를 몰라서 q로 파악한다?????수학적개념->다시 정리해야됨...이해불가.

#optimizer(Gradient descent)
learning_rate = 0.1
optimizer=tf.train.GradientDescentOptimizer(learning_rate)
train=optimizer.minimize(loss)

#Session

sess=tf.Session()
sess.run(tf.global_variables_initializer())

# global_variables_initializer()  소괄호좀 까먹지말자..


# 4.학습

epochs=100000

for step in range(epochs):
    _, loss_val, w_val , b_val =sess.run([train, loss, w, b], feed_dict={x: train_x, y:train_y})
    if step% 500==0:
        print("w:{} \t b: {} \t loss:{}".format( w_val , b_val, loss_val))
#   언더바라는 _, 를 붙이는 이유  처음값 의미 없어서 !!!


# 5.예측및 평가
# sigmoid 그래프 특성상 y값이 0.5 이상이면 1 아니면 0임
predict=tf.cast(H>0.5,dtype=tf.float32 )

equl=tf.cast(tf.equal(predict,y),dtype=tf.float32)
accuracy=tf.reduce_mean(equl)

# 지금 데이터의 크기가 작아서 학습데이터를 넣어줬지만 .사실 test_set(test_x,text_y) 필요!!


print(sess.run(accuracy, feed_dict={x:train_x, y:train_y}))

print("예측:", sess.run(H,feed_dict={x:[[4,2]]}))

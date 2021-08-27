import tensorflow.compat.v1 as tf


# 1.데이터 준비
# 2.데이터 분할

# x=쪽지시험 4번(10점 만점)/y= 상:0 중:1 하:2 으로 해서 하나준다.
train_x=[
    [10,7,8,5],
    [8,8,9,4],
    [7,8,2,3],
    [6,3,9,3],
    [7,5,7,4],
    [3,5,6,2],
    [2,4,3,1]
]
train_y=[
    [1,0,0],
    [1,0,0],
    [0,1,0],
    [0,1,0],
    [0,1,0],
    [0,0,1],
    [0,0,1]
]

# 위에값은 4개인데 아래값은 왜 3개이지??
# one-hot-recoding
# 상중하의 카테고리를 만들었다고 생각하기
# 각각의 카테고리의 전체 비율의 합이 1이 되도록...카테고리를 나눈다..좀더 강의 복습해서 정리해볼것...
x= tf.placeholder(shape = [None,4],dtype=tf.float32)
y= tf.placeholder(shape = [None,3],dtype=tf.float32)


# 3.준비
#가설 설정
w=tf.Variable(tf.random_normal([4,3]), name='weight')
"""
x는 4개 y는 3개이다...
tf.matmul(x,w)+b 를 봐도 matmul(x,w)이당~!!!

행렬의 의미를 생각한다면
(1,4) x (a,b) = (1,3)
a= 4, b= 3

"""
b=tf.Variable(tf.random_normal([3]), name='bias')
'''
[3] 인이유는 값이 3개이당~!! [1,0,0]면 3개이듯이 bias는 상수이다..
'''

logits=tf.matmul(x,w)+b
H=tf.nn.softmax(logits)
#softmax를 쓰는 이유 y전체값합이 1이어서


#loss funtion(cost function)
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))



#optimizer(Gradient descent)
learning_rate = 0.01
train=optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

#Session


sess=tf.Session()
sess.run(tf.global_variables_initializer())


# 4.학습


epochs=3000

for step in range(epochs):
    _, loss_val =sess.run([train, loss], feed_dict={x: train_x, y:train_y})
    if step% 300==0:
        print("loss:{}".format(loss_val))

# 5.예측및 평가
print("예측:", sess.run(H,feed_dict={x:[[9,4,5,1]]}))

predict=tf.argmax(H,1)
correct=tf.equal(predict,tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct,dtype=tf.float32))
print("정확도:",sess.run(accuracy, feed_dict={x:train_x, y:train_y}))


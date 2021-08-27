import tensorflow.compat.v1 as tf



#placeholder:그래프를 실행하는 시점에 데이터를 입력받아서 실행


node1= tf.placeholder(dtype =tf.float32)
node2= tf.placeholder(dtype =tf.float32)
'''
# 대이터를 나중에 던져줄 거기 때문에 미리 공간만 확보해 놓는다고 생각하면 편하다.
scalar 값을 던져 주기 때문에 shape은 [None] 몇개가 들어와도 상관 없기 때문에
'''

node3=node1+node2


sess=tf.Session()

print(sess.run(node3, feed_dict={node1:[10,20,30], node2:[40,50,60]}))
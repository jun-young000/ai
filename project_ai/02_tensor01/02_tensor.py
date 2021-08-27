import tensorflow.compat.v1 as tf


#tf.float32 랑 np.float32 랑 같다!(tensorflow 내부적으로는 numpy 사용중
node1= tf.constant(10, dtype =tf.float32)
node2= tf.constant(10, dtype =tf.float32)


#그래프
# node1과 node가 먼저 실행되야 한다.!(자동으로 node1과 node2 실행해줌)
node3=node1+node2


sess=tf.Session()
print(sess.run(node3))
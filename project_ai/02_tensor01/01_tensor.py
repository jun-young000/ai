
import tensorflow.compat.v1 as tf
print(tf.__version__)
#텐서플로우 버젼확인

'''
Tensorflow
-tensor :데이터 저장 객체
-variable :weight,bias
-Operation: H=w*x=b (수식/노드)--->그래프
-Session  : 실행환경(학습)
'''
#상수노드

node=tf.constant(100)

# Session : 그래프 실행(runner)
sess=tf.Session()

#노드 (그래프) 실행

print(sess.run(node))
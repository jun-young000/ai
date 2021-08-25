import matplotlib.pyplot as plt




x=[1,2,3,4,5]
# result=[]
# def f(x):
#     for num in x:
#         result.append(num*2)
#     print(result)
# print(x)
# ----------------------별로
#
# def f(x):
#     return[i*2 for i in x]
# print(f(x))
#________________________ 조금 굿

# def f(x):
#     for i in x:
#         return x*2
#     map(f,x)
# print(x)
# ------------------------ 굿
# def f(x):
#     return list(map(lambda i:2*i,x))
# print(f(x))
# plt.plot(x,f(x))
# plt.show()
# #--------------------------베리 굿
# # 데이터 분석에서는 람다를 많이 쓰니 숙지할것.사용자 정의이름이 없는 함수가가 람다.
#

# map(함수,입력리스트)
#
# def 함수이름(매개변수):                    lambda 매개변수:결과
#     return 결과
#
#
# list(map(lambda 매개변수:조건식,입력리스트)
#-----------------------

x=[1,2,3,4,5]
def f(x):
     return[i*2 for i in x]
y=f(x)
print(y)
plt.plot(x,y)
plt.show()
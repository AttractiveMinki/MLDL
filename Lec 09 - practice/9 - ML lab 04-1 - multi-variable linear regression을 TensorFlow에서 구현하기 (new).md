https://www.youtube.com/watch?v=fZUV3xjoZSM&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=9



Multi-variable linear regression tensorflow에서 구현



지난 시간까지 한 실습은 x, y 하나 있는 toy project



Multi-variable 사용 가능 -> 실제 데이터에 적용 가능해진다.





### Hypothesis using matrix

H(x1, x2, x3) = x1w1 + x2w2 + x3w3



x 많을수록 더 잘 예측 가능

출석, 퀴즈 점수, 다른 이전의 과목 점수..

x 굉장히 늘어날 것임. 늘어날수록 y를 예측하는 힘이 쎄진다.

prediction power가 쎄진다.



Hypothesis 자연스럽게 세 개로.. -> 학습해야 할 예시가 세 개다. 쭉 늘어써주면 됨, bias 생략

이걸 코드로 써보자.



https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-04-1-multi_variable_linear_regression.py

![9-1](9 - ML lab 04-1 - multi-variable linear regression을 TensorFlow에서 구현하기 (new).assets/9-1.PNG)

```python
# Lab 4 Multi-variable linear regression
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]

y_data = [152., 185., 180., 196., 142.]

# placeholders for a tensor that will be always fed.
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b
```



x, y 데이터 나열해준다.

placeholder 만들어준다.

weight는 각각 하나의 값 -> [1] 일케 드간다. tf.random_normal([1])

bias도 마찬가지

hypothesis는 식 그대로 곱하기를 한 다음에 더한다

맨 뒤에 bias 추가



```python
# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize. Need a very small learning rate for this data set
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                   feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

```



cost는 hypothesis와 y 차이의 제곱

optimizer 사용

숫자 0.01, 0.001... 복잡해지쥬?

1e-5처럼 쓸 수 있다. Learning Rate 주었따.

똑같은 방법으로 sess 만들고, initializer 하고, 학습을 시킨다.

train 돌리게 됨. 같이 cost와 hypothesis를 기록했다가 출력.

x1, x2, x3을 feed_dict로 주게 된다.

다 똑같고, data부분, weight부분만 확장되었다(늘어났다)고 보면 됨.



처음에 cost 상당히 높음

많이 할수록 cost 낮아짐.

prediction -> 목표한 y값과 유사하게 수렴한다고 볼 수 있다.



그림 굉장히 복잡해보이는 이유 중 하나

placeholder, Variable 주는거 복잡해보이쥬? [같은 수식 반복..]

x data 100개가 된다면.. -> decode같이 정말 복잡해질 것.

=> 이런 방법은 사용하지 않는다.





### Matrix

위 방법을 대체한 것이 우리가 좋아하는 Matrix

![9-2](9 - ML lab 04-1 - multi-variable linear regression을 TensorFlow에서 구현하기 (new).assets/9-2.PNG)



x 갯수에 맞게 옆으로 쭉 나열하고, 갯수에 맞게 weight 써주면, 곱해라 라고 하면

다 곱해서.. H(W) = XW



이것을 구현



https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-04-2-multi_variable_matmul_linear_regression.py

```python
# Lab 4 Multi-variable linear regression
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]


# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b
```



Matrix기 때문에 위와 같이 넣어줄 수 있다.

[73., 80., 75.] 요거 하나가 x1, x2, x3의 값..

하나의 element, 하나의 instance, 하나의 x data가 된다.

여기에 해당하는 y는 [152. ]가 된다.



만약 x가 많아졌다 -> 75. 뒤에 데이터 추가되면 된다.



placeholder도 만들어야겠죠? 만들 때 주의사항 -> shape



x1, x2, x3 3개 드감 -> X 맨 뒤에 3이 드감 shape=[None, 3]

None 이 부분 -> data가 몇 개냐? -> 지금은 5개 정해져있지만 필요에 따라 추가될 수 있다.

대략 x data n개 가질 것이다 -> tensorflow에선 None으로 한다. 니가 원하는만큼 쓸 수 있다.

각 element가 가지는 값은 3개다 -> 3 써줌



y는 값 하나로 되어있쥬? final_exam 점수 하나자너 -> 1

n개 가능 -> None

hypothesis 그대로 가져감



```python
# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
```



cost, optimizer, train 똑같다.

학습한 모든 방법 똑같다.



위의 툴을 보면.. 이전에 굉장히 복잡하게 나열 -> 상당히 단순하게 간단해짐.

데이터는 좀 복잡하게 가져올 수 있겠지만, 나머지는 굉장히 단순해짐.

Matrix의 굉장한 장점.

결과 이전과 똑같이 나온다.



Matrix 사용 -> Variable 갯수 상관없이 원하는 갯수만큼 쉽게 표현할 수 있다.



다음시간 - 굉장히 많은 데이터 다루게 됨..

일일이 파일에 소스코드 적어주기 힘드니까

데이터를 파일에다가 적어주고, 그것을 로딩하는 방법에 대해 얘기해보자.


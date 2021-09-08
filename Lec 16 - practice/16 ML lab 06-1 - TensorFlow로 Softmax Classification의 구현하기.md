https://www.youtube.com/watch?v=VRnubDzIy3A&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=16



### Lab 6-1 SoftMax Classifier





### Softmax function

![16-1](16 ML lab 06-1 - TensorFlow로 Softmax Classification의 구현하기.assets/16-1.PNG)

softmax function

여러 개의 class를 return할 때 유용하다.



binary classification -> 0이냐 1이냐 이렇게만 예측

두 개보다 여러 개를 예측하는 경우가 많다.

n개의 예측할 것들이 있을 때 softmax classifier를 사용하면 굉장히 유용하다.



시작: XW = Y

주어진 X값에 학습할 Weight를 곱해서 값을 만드는 것을 시작한다.

2.0, 1.0, 0.1 이렇게 나오는 값들은 score에 불과하다.

이것을 softmax라고 불리는 function을 통과시키면, 이것이 확률로 나오게 된다.

A, B, C label이 있다면

A가 될 확률이 0.7, B가 0.2, C가 0.1 이런 식으로 확률로 표현이 가능해진다.

모든 class의 label 확률을 모두 더하게 되면 반드시 1이 될 것입니다.( Σ = 1 )

이런 식으로 멋지게 표현 가능.



이것을 tensorflow로 어떻게 구현할 것인가?

매우 간단

![16-2](16 ML lab 06-1 - TensorFlow로 Softmax Classification의 구현하기.assets/16-2.PNG)

기본적으로 tensorflow로 수식을 그대로 써주면 된다.

XW=Y tensorflow로 어떻게 쓰면 될까요?

```python
tf.matmul(X, W) + b
```

매트릭스곱하기(X, W) + bias



softmax function 복잡하게 생겼지만 예쁘게 생김

실제로 구현하셔도 됩니다. 조금 복잡하니깐 tensorflow library 제공

```python
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
```

tf.nn.softmax라고 써주면 된다.

안쪽에 값들, Logit 넣어주면 softmax가 나오게 된다.

이것이 우리의 최종 hypothesis가 된다.

어떤 label이 될 것인가에 대한 확률로 나오게 된다.

이런 것 tensorflow 이용해서 쉽게 구현 가능하다.





### Cost function: cross entropy

![16-3](16 ML lab 06-1 - TensorFlow로 Softmax Classification의 구현하기.assets/16-3.PNG)

softmax function이란 아름다운 function 봤다.

여기에 걸맞는 Loss function이 필요하다.



tensorflow를 통해 쉽게 얘기할 수 있다.

https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-06-1-softmax_classifier.py

```python
# Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
```



loss function 수업시간에 얘기한 것처럼

`Ylog(Y햇)` [Y햇은 예측한 값,hypothesis]

위에 것을 우리가 cross-entropy라고 얘기했다.

이것을 tensorflow로 구현하는 방법 간단.

위에꺼를 하고 평균을 낸다.

`Y * tf.log(hypothesis)` 한다.

이 곱 자체가 여러 개의 matrix 값이 되겠죠?

합을 한 다음에 평균을 내면 된다.



cost가 주어졌으면 cost를 minimize해야 되겠죠?

많이 다루었던 경사면 타고 내려가기를 통해서..

(오른쪽 초록색 글자 참고)

cost함수 Lost를 미분한 값에다가 앞에 Learning Rate를 곱해서 weight에 -를 해주면 되죠?

이것이 기본적인 방법.



tensorflow에서 사용하는 것은 한 줄로 GradientDescentOptimizer를 이용한다.

Learning Rate는 0.1이고, 무엇을 minimize하는지 알려준다.



전체 구현 한 페이지로..

1.X version

```python
# Lab 6 Softmax Classifier
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

X = tf.placeholder("float", [None, 4])
Y = tf.placeholder("float", [None, 3])
nb_classes = 3

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
            _, cost_val = sess.run([optimizer, cost], feed_dict={X: x_data, Y: y_data})

            if step % 200 == 0:
                print(step, cost_val)
```

X, Y 눈여겨보자.

x_data 한 데이터에 4개의 element가 있다.

y_data 이전에 우리가 binary classification에서 숫자 하나만 주어졌죠? 0이나 1

여기 -> 여러 개의 class가 있음 -> 한 자리로 표시가 불가능

어떻게 표시할까.. 여기에선

0, 1, 2 세 가지 class가 있겠죠?

어떻게 표시할까? 여러 가지 방법..

그 중에서 'ONE-HOT' Encoding을 사용한다.

ONE-HOT 재밌는 표현. 하나만 뜨겁다, 하나만 HOT하다.

세 개의 label을 가진다.

-> 세 자리를 만들어준다.

그 중에 하나만 HOT하게 한다. (1이란 얘기)

하나만 1로 만들어준다.

y_data ONE-HOT encoding으로 표시함.

label을 이렇게 표시.



X, Y에 대해 placeholder

shape을 조심해야 한다.

x_data 4개의 element

[None, 4]

None: data가 몇 개가 있나요? 상관없음

Y는 [None, 3]

ONE-HOT으로 표시한 것 3가지

ONE-HOT으로 표시할 때에는 Y의 갯수가 label의 갯수이다.



nb_classes = 3

클래스의 갯수



W, b 정해준다. shape 주의

[4, nb_classes]

4: 입력되는 값, X의 값이 4개

nb_classes: 나가는 값, Y의 값

여기서는 number의 class.. nb_classes = 3

b는 출력 값과 같다 -> [nb_classes] = 3

이렇게 Weight과 bias를 만들 때 Shape를 주의하셔야 합니다.



나머지는 똑같다.

```python
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
```

softmax를 거쳐서 matrix X, W를 곱한 다음에 bias를 더한다.



cost 멋지게 계산한 후에 GradientDescent로 학습을 시킨다.

학습하는 방법

세션을 열고 초기화

loop를 돌면서 optimizer 실행

x data, y data를 던져준다.

학습이 일어나고 중간중간에 출력 가능





### Test & one-hot encoding

그럼 이제 출력을 해볼까요? 간단하게 하나씩 보자..

![16-4](16 ML lab 06-1 - TensorFlow로 Softmax Classification의 구현하기.assets/16-4.PNG)

hypothesis를 이렇게 주었다..

```python
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
```



오른쪽 그림이 보여준다..

softmax를 통해서.. 확률로 주어졌다.

이렇게 주어졌을 때 이것을 이용해서 새로운 값을 던져주고, 예측해봐!

이걸 하고싶다.



학습이 끝난 다음에, hypothesis에 x를 던져준다.

그 다음에 예측을 하게 함.



a란 값은 0.7, 0.2, 0.1 이런 식으로 나올 것이다.

softmax를 통과된 확률 값.

출력해보면 a -> 3개의 값이 나온다.

복잡해보이지만.. 대략 중앙은 0.9, 양 옆은 매우 작은 값

전체 합은 1이 될 것이다.

중앙이 가장 최강자쥬?

0번 째, 1번 째, 2번 째 중 1번 째가 최강자

이것을 자동으로 해주는 것이

```python
arg_max(a, 1)
```

tensorflow에 있는 arg_max

이런 형식(softmax)의 array를 주면..

몇 번째에 있는 argument가 가장 높은지 물어봄 -> 1을 돌려주게 된다.

돌아온 softmax 값 -> 우리가 원하는 label로.. arg_max를 사용한다.



![16-5](16 ML lab 06-1 - TensorFlow로 Softmax Classification의 구현하기.assets/16-5.PNG)

질문할 때 하나씩 질문하면 조금 심심하니까, 여러 개를 질문할 수도 있죠?

친절한 tensorflow 다 답을 알려준다.

hypothesis 아래 형식으로 확률로 답을 알려준다.

각각 첫 번째, 두 번째, 세 번째 질문에 대한 답.



우리가 알고 싶은 것: label을 구체적으로 말해보라

바로 arg_max를 쓴다.

arg_max(all, 1) 

최강자를 찾아주세요

답이 [1, 0, 2] 이렇게 나온다.



softmax로 구현도 하고, arg_max를 사용해서 나온 값을 실제 label로 바꾸어보았다.



다음시간)

Lab 6-2

Fancy Softmax Classfier

cross_entroty, one_hot, reshape



cross_entroty, one_hot, reshape 사용해서 좀 더 Fancy한 Softmax Classifier를 만들어보겠다.


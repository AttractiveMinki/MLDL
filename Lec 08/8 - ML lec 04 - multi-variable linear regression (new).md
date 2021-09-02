https://www.youtube.com/watch?v=kPxpJY6fRkY&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=8



### Lecture 4

Multivariable linear regression



하나의 변수가 아닌 multivariable linear regression





### Recap

Linear regression 하기 위해 필요한 세 가지

- Hypothesis
- Cost function
- Gradient descent algorithm



1. 가설이 무엇인지, 어떻게 예측할 것인지
   1. H(..)
2. Cost를 어떻게 계산할 것인지, 우리가 잘했는지 못했는지 예측하는 방법
   1. cost, loss = ()^2
3. Cost를 최적화하는 알고리즘

![8-1](8 - ML lec 04 - multi-variable linear regression (new).assets/8-1.PNG)



H(x) = Wx + b

W: Weight

b: bias

W, b 두 개의 값을 학습하겠다.



cost는 무엇인가? cost 함수로 만듦.

W, b의 함수가 되겠쥬? W, b에 따라 Cost가 정해진다.

우리가 예측한 값(H(x))에 실제 값 y.. (true value)

예측 값 - 실제 값 차이를 제곱해서 각각 더한다.

cost 함수가 주어지면 대부분 밥그릇 형태로 주어진다..

cost 최저가 되는 값을 찾는다.

W 어떤 점에서 시작하면 경사면을 따라점점 내려간다.

Gradient descent algorithm을 통해서, W, b 값을 찾는다.





### Predicting exam score: regression using one input (x)

하나의 input variable

one-variable

one-feature

| x (hours) | y (score) |
| :-------: | :-------: |
|    10     |    90     |
|     9     |    80     |
|     3     |    50     |
|     2     |    60     |
|    11     |    40     |





### Predicting exam score: regression using three inputs (x1, x2, x3)

하나가 아니라 여러 개의 input. 여기선 세 개



multi-variable / feature

| x1 (quiz 1) | x2 (quiz 2) | x3 (midterm 1) | Y (final) |
| :---------: | :---------: | :------------: | :-------: |
|     73      |     80      |       75       |    152    |
|     93      |     88      |       93       |    185    |
|     89      |     91      |       90       |    180    |
|     96      |     98      |      100       |    196    |
|     73      |     66      |       70       |    142    |

Test Scores for General Psychology



퀴즈 3개 점수로 최종 점수 예측

어떻게 하면 될까요?





### Hypothesis

이전) H(x) = Wx + b

H(x1, x2, x3) = w1x1 + w2x2 + w3x3 + b





### Cost function

예측한 값 - 실제 값

![8-2](8 - ML lec 04 - multi-variable linear regression (new).assets/8-2.PNG)



cost 똑같고, Hypothesis만 여러 개로 벌려준다.





### Multi-variable

더 많은 경우도 마찬가지

![8-3](8 - ML lec 04 - multi-variable linear regression (new).assets/8-3.PNG)





### Matrix

w1x1 + w2x2 + w3x3 + ... + wnxn

좀 불편한 감이 있는 듯. 엄청 길어질텐데..

잘 처리하는 방법이 없을까? -> Matrix 이용.

복잡한 느낌 들 수 있으나 전혀 그렇지 않다.





### Matrix multiplication

Matrix 곱셈만 사용할꺼다.

![8-4](8 - ML lec 04 - multi-variable linear regression (new).assets/8-4.PNG)

기억나시쥬?

기억 안나도 괜찮다. 가볍게 복습해오십쇼





### Hypothesis using matrix

![8-5](8 - ML lec 04 - multi-variable linear regression (new).assets/8-5.PNG)



w1x1 + w2x2 + w3x3 어떻게 Matrix로 표현?

계산..

실제로 원하는 값은 오른쪽

왼쪽으로 편하게 표현 가능.



오른쪽에 x1w1.. x, w의 순서가 바뀌어 있다.

바뀌어 있어도 같은 식이쥬?



matrix 사용시 보통 X를 앞에 쓴다.

X = (x1 x2 x3), W = (w1)

​								  w2

​								  w3

matrix란 뜻으로 X, W 대문자로 쓴다.

H(X) = XW



그럼 이제 실제 데이터에 적용해보자.

간단하게 하기 위해 bias 생략

![8-6](8 - ML lec 04 - multi-variable linear regression (new).assets/8-6.PNG)



근데 생각해보면 데이터 수 5개.

더 많을 수도 있겠죠? 100개, 200개...

이 하나를 instance라고 부른다.

많은 인스턴스들이 있다..

각각을 연산할 때 하나씩 넣어서 (Matrix)곱하고, 그 다음 x instance를 넣어서 곱하고... 여러 번 반복해서 계산하면 되죠? 그런데 효율성은 좀 떨어지겠죠?



Matrix의 굉장히 놀라운 성질

![8-7](8 - ML lec 04 - multi-variable linear regression (new).assets/8-7.PNG)



instance의 수대로 matrix를 줘버린다.

w는 이전과 똑같다. 변함없음

Matrix 곱을 함.

마치 이전 매트릭스 식에서 각각 한 번 계산한 것과 같이

쭉 넣기만 하면 연산 값이 한 번에 나온다.

Matrix 또다른 장점

Instance 많을 때 각각의 instance 계산할 필요 없이 긴 Matrix에 넣고  W를 매트릭스곱 -> 원하는 값이 나온다.



![8-8](8 - ML lec 04 - multi-variable linear regression (new).assets/8-8.PNG)



[5, 3] 5 by 3

3 어디서 왔나? x1, x2, x3

입력으로 들어오는 feature가 3개

5 -> data sample, instance가 5였다.



Weight 어떻게 Matrix를 해?

마찬가지쥬? [3, 1]

연산을 하게 되면 [5, 1]로 나온다.

곱하기를 할 때 [5, 3] [3, 1]에서 앞 행렬의 뒤 수(3)와 뒤 행렬의 앞 수(3)가 같아야 한다.

같다면 곱하기를 하면 두 개를 지운 [5, 1] 결과가 나온다.



![8-9](8 - ML lec 04 - multi-variable linear regression (new).assets/8-9.PNG)



종종 이런 형태의 Matrix 곱 많이 하게 될 것이다.

보통 [5, 3] 주어진다.

3: x variable의 갯수

5: 데이터를 몇 개나 넣을 것인가? instance의 갯수



출력 값도 보통 주어진다. [5, 1]

1: Linear regression의 경우 y 하나만 나오쥬?

5: 마찬가지로 instance의 값. 몇 개나 넣냐



대개의 경우 설계해야 할 것은 W에 해당되는 weight의 크기를 결정해야 한다.

어떻게 결정할 수 있을까? 쉽다.

[5, 3]의 3을 가져오고, [5, 1]의 1을 가져온다.

[3, 1]



instance의 개수 무시.

y variable의 개수가 얼마인가? 3

몇 개를 y의 결과로 낼 것인가? 1

[3, 1]

쉽게 w의 크기 결정 가능



![8-10](8 - ML lec 04 - multi-variable linear regression (new).assets/8-10.PNG)



예제 데이터 5개만 넣음

실제로는 데이터 개수에 따라 가변하는 값 -> n개라고 표현 가능.

실제 구현에서는, numpy에서는 -1로 표시. tensorflow에선 None이라고 표시

None개 -> n개, 여러 개, 원하는 만큼 들어올 수 있다고 보시면 된다.





### Hypothesis using matrix (n output)

출력 꼭 하나일 필요는 없쥬?

![image-20210831225703704](8 - ML lec 04 - multi-variable linear regression (new).assets/image-20210831225703704.png)

다음 번에 나오는 여러 개 형태의 Machine Learning을 다룰 것..

이 때에도 Matrix 쓰면 상당히 편하다.

n개의 입력, 3개의 x feat가 있는 경우의 데이터를 받아들임

우리가 2개의 출력, 3개, 4개의 출력을 원할 때

W 크기는 얼마가 될까요?



앞의 행렬에서 입력 3개, 뒤의 행렬에서 출력 2개

[3, 2] W를 사용하면 원하는 출력을 얻을 수 있다.

Matrix를 사용하면 multi-variable의 경우에도 쉽게 처리할 수 있고, instance가 많은 경우에도 n으로 쉽게 처리할 수 있고, 출력이 여러 개인 경우에도 쉽게 처리할 수 있는 굉장한 장점이 있다.





### WX vs XW

- Lecture (theory):
  - H(x) = Wx + b
- Implementation (TensorFlow)
  - H(X) = XW



정리

Matrix 많이 사용하게 될 거다.

x 하나만 주어지지 않고 여러 개 주어지니까. 굉장히 많이 사용하게 될 것.

Lecture, theory에서는 H(x) = Wx + b 형태로 많이 쓰인다.

실제로 구현할 때엔 X를 앞에 쓰면, 별도의 처리 없이 Matrix 뿐만으로도 계산 가능



구현에선 XW

수학적인 의미는 Wx + b와 같다.



다음 시간

정말 중요한 Logistic Regression (Classification)에 대해 얘기해보겠다.


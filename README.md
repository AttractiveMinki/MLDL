https://www.youtube.com/playlist?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm



모두를 위한 딥러닝 강좌 시즌 1 

by 김성훈 교수님





- ### 목표

Machine Learning, Deep Learning 및 tensorflow에 대한 전반적인 이해

ML 관련 프로젝트 진행 시 기반이 될 지식 학습





- ### 학습 계기

AI 관련 특화 프로젝트

모 학우의 github를 보고 따라하고 싶은 마음이 생겼다.





- ### 사전에 알아두면 좋을 내용

강의: 2016년 3월 경 시작

현재 2021년 8월

당시 대세: tensorflow

요즘 대세: tensorflow, pytorch



당시 tensorflow: 1.x 버전

요즘 tensorflow: 2.x 버전



#### 1.x 버전과 2.x 버전의 차이가 있다.

ex) 

1.x버전: sess = tf.Session(), tf.placeholder(tf.float32)같은 명령어 사용

2.x버전: Session, placeholder 사용 불가



바뀐 점이 많다.

모든 코드를 이해하고 넘어가기보단, 개념 위주로 수강하자!





### 추가로 도움을 받은 사이트 모음

딥 러닝을 이용한 자연어 처리 입문

https://wikidocs.net/book/2155





### 목차

1 - Lec 00 - Machine, Deep learning 수업의 개요와 일정

2 - ML lec 01 - 기본적인 Machine Learning 의 용어와 개념 설명

3 - ML lab 01 - TensorFlow의 설치및 기본적인 operations (new)

4 - ML lec 02 - Linear Regression의 Hypothesis 와 cost 설명

5 - ML lab 02 - TensorFlow로 간단한 linear regression을 구현 (new)

6 - ML lec 03 - Linear Regression의 cost 최소화 알고리즘의 원리 설명

7 - ML lab 03 - Linear Regression 의 cost 최소화의 TensorFlow 구현 (new)

8 - ML lec 04 - multi-variable linear regression (new)

9 - ML lab 04-1 - multi-variable linear regression을 TensorFlow에서 구현하기 (new)

10 - ML lab 04-2 - TensorFlow로 파일에서 데이타 읽어오기 (new)

11 - ML lec 5-1 - Logistic Classification의 가설 함수 정의

12 ML lec 5-2 Logistic Regression의 cost 함수 설명

13 ML lab 05 - TensorFlow로 Logistic Classification의 구현하기 (new)

14 ML lec 6-1 - Softmax Regression - 기본 개념 소개

15 ML lec 6-2 - Softmax classifier 의 cost함수

16 ML lab 06-1 - TensorFlow로 Softmax Classification의 구현하기

17 ML lab 06-2- TensorFlow로 Fancy Softmax Classification의 구현하기

18 - lec 07-1 - 학습 rate, Overfitting, 그리고 일반화 (Regularization)

19 lec 07-2 - Training, Testing 데이타 셋

20 ML lab 07-1 - training, test dataset, learning rate, normalization

21 ML lab 07-2 - Meet MNIST Dataset

22 lec 08-1 - 딥러닝의 기본 개념, 시작과 XOR 문제

23 lec 08-2 - 딥러닝의 기본 개념2, Back-propagation 과 2006,2007 '딥'의 출현

24 ML lab 08 - Tensor Manipulation

25 lec9-1 - XOR 문제 딥러닝으로 풀기

26 lec9-x - 특별편 - 10분안에 미분 정리하기 (lec9-2 이전에 보세요)

27 lec9-2 - 딥넷트웍 학습 시키기 (backpropagation)

28 ML lab 09-1 - Neural Net for XOR

29 ML lab 09-2 - Tensorboard (Neural Net for XOR)

30 lec10-1 - Sigmoid 보다 ReLU가 더 좋아

31 lec10-2 - Weight 초기화 잘해보자

32 lec10-3 - Dropout 과 앙상블

33 lec10-4 - 레고처럼 넷트웍 모듈을 마음껏 쌓아 보자

34 ML lab10 - NN, ReLu, Xavier, Dropout, and Adam

35 lec11-1 ConvNet의 Conv 레이어 만들기

36 lec11-2 - ConvNet Max pooling 과 Full Network

37 - lec11-3 ConvNet의 활용예

38 - ML lab11-1 - TensorFlow CNN Basics

39 ML lab11-2 - MNIST 99% with CNN

40 ML lab11-3 - CNN Class, Layers, Ensemble

41 lec12 - NN의 꽃 RNN 이야기

42 ML lab12-1 - RNN - Basics

43 ML lab12-2 - RNN - Hi Hello Training

44 ML lab12-3 - Long Sequence RNN

45 ML lab12-4 - Stacked RNN + Softmax Layer

46 ML lab12-5 - Dynamic RNN

47 ML lab12-6 - RNN with Time Series Data

48 lab13 - TensorFlow를 AWS에서 GPU와 함께 돌려보자

49 lab14 - AWS에서 저렴하게 Spot Instance를 터미네이션 걱정없이 사용하기

50 Google Cloud ML with Examples 1 (KOREAN)



-----

이하 0강 내용 요약



재미있는 머신러닝, 딥러닝 여러분들과 얘기해보고 싶어서 강좌를 만들었다.





### 이세돌, 알파고 바둑 대결

바둑 -> 경우의 수가 너무 많아 명석하고 예지력, 직관력 있는 사람만이 잘 한다고 믿음.

이세돌 선수 아쉽게도 패배

많은 사람들이 인공지능이란 무엇인가? 인간만이 할 수 있는 직관적인 의사결정 인공지능도 할 수 있나? 에 대한 고민 하게 된 대결





### Dr.Andrew Ng(2015년): 

ML(머신 러닝)을 잘 이해하고 사용하는 것이 super power를 갖는 것

많은 사람들이 super power를 갖게 하는 것이 목표!





### 강의 타겟

- 머신 러닝이 궁금한 사람

- 수학적인 지식이나 CS 배경지식이 없거나 약한 사람도 이해할 수 있는 수준

  y = Wx + b 이정도 수학 이해하면 따라올 수 있고 ML 이해 가능

- ML 블랙박스처럼 안에 뭐들었는지 몰라도 입력 출력을 보고 원하는 일을 할 수 있다.

  좀만 더 이해하면 블랙박스 효율적으로 사용 가능

- 이론적인 건 아는데 Tensorflow로 구현해보고 싶다.





### 비디오 목표

- ML에 대한 기본적인 이해
  - Linear regression, Logistic regression
  - 굉장히 쉬우면서도 powerful한 알고리즘
  - 이거 이해시 DL 쉽게 이해 가능
- DL(딥 러닝)을 이해하기 위한 기본적인 알고리즘
  - Neural networks, Convolutional Neural Network, Recurrent Neural Network
- 기본적인 이해를 바탕으로 문제를 풀고 싶다.
  - Tensorflow and Python





### 수업

- 10분 정도의 비디오(지하철, 이동 중이나 짬을 내어서도 충분히 보면서 내용 이해)
- Tensorflow 프로그래밍 하는 것 보고 어떻게 돌아가는구나 감을 잡을 수 있다.





### 참고자료

- Andrew Ng's ML class

- Convolutional Neural Networks for Visual Recognition

- Tensorflow





### 스케쥴

- Machine learning basic concepts
- Linear regression
- Logistic regression (classification)
- Multivariable (Vector) linear / logistic regression
- Neural networks
- Deep learning
  - CNN
  - RNN
  - Bidirectional Neural networks





다음 시간부터 머신 러닝의 기본적인 내용 알아보자.


https://www.youtube.com/watch?v=qPMeuL2LIqY&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=2



기본적인 용어, 개념 설명

다음 강의부터 실제적인 알고리즘



### Basic concepts

- What is ML?
- What is learning?
  - supervised
  - unsupervised
- What is regression?
- What is classification?





### Machine Learning

일종의 소프트웨어, 프로그래밍.

앱 입력 기반으로 데이터 보여줌 -> explicit programming

개발자가 이런 환경에서 일케 반응하라고 만들어놓음

어떤 프로그램에서는 explicit하기 어려운 경우가 있다.

- Limitations of explicit programming
  - Span filter: many rules
  - Automatic driving: too many rules

rule이 너무 많아 개발자가 일일이 이럴땐 이렇게 해라 하기 어렵

구글 자동주행차 프로그래밍 할 순 있지만 어렵

1959 Arthur Samuel

일일이 프로그래밍하지 말고 어떤 자료에서, 현상에서 자동적으로 배우면 어떨까? -> 머신 러닝

- Machine learning: "Field of study that gives computers the ability to learn without being explicitlly programmed" Arthur Samuel (1959)

개발자가 일일이 프로그램 어떻게 할지 정하지 않고, 프로그램이 데이터를 보고 학습하는 것





### Supervised / Unsupervised learning

학습하는 방법에 따라 나뉨



- Supervised learning
  - learning with labeled examples - training set

라벨이 정해져 있는 데이터. training set으로 학습

이미지를 주고 고양이? 강아지? 알아내는 프로그램 -> ML을 이용해 만듦

사람들이 고양이 사진을 막 주고 이건 고양이야 알려줌(label 달려있는 자료 줌) -> supervised learning



- Unsupervised learning: un-labeled data
  - Google news grouping
  - Word clustering

어떤 러닝 일일이 라벨 줄 수 없는 경우 있다.

구글 뉴스 - 자동적으로 유사한 기사 모음.

미리 라벨 어렵..

비슷한 단어 모으기.. -> label 데이터를 보고 스스로 학습.





### Supervised learning

비디오에서 주로 다룸

- Most common problem type in ML
  - Image labeling: learning from tagged images
  - Email spam filter: learning from labeled (spam or ham) email
  - Predicting exam score: learning from previous exam score and time spend

스팸 필터 - 이메일 가운데 이런건 스팸, 이런건 아님 라벨 매겨놓은거로 학습

시험 성적 예측 - 이전에 시험 친 사람들 몇 시간 준비했는데 성적 얼마다 데이터 있으면 이거로 학습 가능





### Training data set

ML이란 모델이 있고, 이미 답(Label)이 정해져 있는.. Y가 있고, X는 값의 특징(feature)

|    X    |  Y   |
| :-----: | :--: |
| 3, 6, 9 |  3   |
| 2, 5, 7 |  2   |
| 2, 3, 5 |  1   |



이런 데이터를 갖고 학습.. -> 모델이 발생.

내가 모르는 X_test의 값이 X_test = [9, 3, 6]

이걸 ML한테 물어본다 -> ML 내 생각에 이건 Y=3이다. 답을 해줌

이게 일반적인 Supervised learning의 예시

표 -> training data set





### 알파고

여러 요소 중 하나



기존) 사람들이 바둑판에서 바둑을 익히고 학습

이세돌 선수가 바둑돌을 놓음 -> 알파고가 데이터를 입력받고 다음수 출력

이것도 하나의 Supervised learning이고 사용된 데이터는 training data set





### Types of supervised learning

- Predicting final exam score based on time spent
  - regression
  - 0 ~ 100점
- Pass / non-pass based on time spent
  - binary classification
  - 패논패
- Letter grade (A, B, C, E and F) based on time spent
  - multi-label classification
  - A, B, C, E, F

Supervised learning 대게 세 가지

regression, classification, multi-label classification

##### 참고) classification: 분류, 유형, 범주





### Predicting final exam score based on time spent

supervised learning -> 학습할 data가 필요

| x (hours) | y (score) |
| :-------: | :-------: |
|    10     |    90     |
|     9     |    80     |
|     3     |    50     |
|     2     |    30     |

Regression Model

학습 데이터를 갖고 모델 train

어떤 학생 7시간 공부 -> x=7

y=? 대략 75점정도 받겠다 예측 가능





### Pass / non-pass based on time spent

| x (hours) | y (pass/fail) |
| :-------: | :-----------: |
|    10     |       P       |
|     9     |       P       |
|     3     |       F       |
|     2     |       F       |

classification, 두 개니까 binary classification





### Letter grade (A, B, C, E and F) based on time spent

| x (hours) | y (pass/fail) |
| :-------: | :-----------: |
|    10     |       A       |
|     9     |       B       |
|     3     |       D       |
|     2     |       F       |

multi-label classification



다음 비디오

Linear regression 어떻게 만들고 어떻게 학습시키는지 알아보자.


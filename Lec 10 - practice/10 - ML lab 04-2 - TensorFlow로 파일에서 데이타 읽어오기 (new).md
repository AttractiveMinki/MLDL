https://www.youtube.com/watch?v=o2q4QNnoShY&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=10



### Lab 4-2 Loading Data from File



많은 multi-variable(x data, y data)를 file에서 읽어오는 방법 소개



[교수님께서 linear_regression부터 tf 2.0을 지원하는 코드도 올려주셨다! 감사,,]

https://github.com/hunkim/DeepLearningZeroToAll/blob/master/tf2/tf2-04-3-file_input_linear_regression.py



데이터 많아지니까 소스코드에 일일이 쓰기 복잡해졌다.

text file로 정리해뒀다.

,로 나눠진 csv파일에 보통 정리를 많이 한다.



```python
import numpy as np

xy = np.loadtxt('../data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# Make sure the shape and data are OK
print(x_data, "\nx_data shape:", x_data.shape)
print(y_data, "\ny_data shape:", y_data.shape)
```



EXAM1	EXAM2	EXAM3	FINAL
73	80	75	152
93	88	93	185
89	91	90	180
96	98	100	196
73	66	70	142
53	46	55	101



어떻게 읽어오면 좋을까?

여러 방법이 있지만 가장 간단한 방법 - numpy에 있는 loadtxt를 쓰면 된다.

파일 이름, separate로 , 사용, data type 입력

단점) 전체 데이터가 같은 데이터 타입이어야 한다.



2차원 데이터 x, y slice로 나눈다.





### Slicing

Python list의 강력한 기능 중 하나.

nums = range(5)

print nums

print nums[2:4]

print nums[:-1] 마지막 한 개를 떼고 가져온다.



tensor를 다루다보니, 여러분이 알아두시면 좋다.





### Indexing, Slicing, Iterating

b = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

b[:, 1]

// array([2, 6, 10])



b[0:2, :]

// array([[1, 2, 3, 4], [5, 6, 7, 8]])



다시 위의 코드로 돌아가서,,

x_data = xy[:, 0:-1]

행 다가져오고, 열 마지막 하나 빼놓은 걸 x_data라 하겠다.

y_data = xy[:, [-1]]

전체 행을 가져온다, 마지막 열만 가져오는 걸 y_data라 한다.



데이터 읽은 뒤 shape이 맞는지, data가 맞는지 꼭 확인 후에 학습하자.





```python
import tensorflow as tf
import numpy as np

xy = np.loadtxt('../data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# Make sure the shape and data are OK
print(x_data, "\nx_data shape:", x_data.shape)
print(y_data, "\ny_data shape:", y_data.shape)

# data output
'''
[[ 73.  80.  75.]
 [ 93.  88.  93.]
 ...
 [ 76.  83.  71.]
 [ 96.  93.  95.]] 
x_data shape: (25, 3)
[[152.]
 [185.]
 ...
 [149.]
 [192.]] 
y_data shape: (25, 1)
'''

tf.model = tf.keras.Sequential()
# activation function doesn't have to be added as a separate layer. Add it as an argument of Dense() layer
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=3, activation='linear'))
# tf.model.add(tf.keras.layers.Activation('linear'))
tf.model.summary()

tf.model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(lr=1e-5))
history = tf.model.fit(x_data, y_data, epochs=2000)

# Ask my score
print("Your score will be ", tf.model.predict([[100, 70, 101]]))
print("Other scores will be ", tf.model.predict([[60, 70, 110], [90, 100, 80]]))
```



학습을 시킨 뒤 물어볼 수 있겠죠?

동시에 물어보는 것도 가능.





### Queue Runners

파일이 굉장히 크던지 메모리 한 번에 올리기 부족한 경우 있겠죠?

numpy 써서 올리면 메모리 부족하다 나옴

tf -> Queue Runners라는 멋진걸 가지고 있다.



![10-1](10 - ML lab 04-2 - TensorFlow로 파일에서 데이타 읽어오기 (new).assets/10-1.PNG)

하나의 파일이 아니라 여러 개의 파일을 읽어올 수 있겠죠?

여러 개의 파일을 Queue에 쌓는다.

이것을 Reader로 연결하고 data를 읽은 다음에 양식에 맞게 decoder 한다(comma 분리 등)

데려온 값을 Queue에 쌓는다.

학습시킬 때 전체 다가 필요한건 아니쥬?

batch만큼 데려와서 학습시킨다.

조금씩 꺼내쓰면 된다.



전체 과정은 tf가 책임진다.

메모리도 한 번에 전체 데이터를 올리지 않으니, 큰 데이터를 사용할 때 좋다.

굉장히 복잡해 보이지만..

막상 보면 간단하게 사용 가능.



![10-2](10 - ML lab 04-2 - TensorFlow로 파일에서 데이타 읽어오기 (new).assets/10-2.PNG)

세 가지 스텝으로 나눌 수 있다.



첫 스텝: 가지고 있는 파일이 몇개인가

파일의 리스트를 만들어준다.

shuffle 할껀가, file list의 이름.

기본적으로 '파일 이름'을 리스트한다 고 보면 됨



두 번째: 이 파일을 읽어올 Reader를 정의해준다.

TextineReader, Binary를 읽는 다른 reader 도 있다...

key value를 읽겠습니다..

text 파일 읽을 때 일반적으로 사용 가능



(3) 텍스트 파일 읽어옴

value를 읽어왔쥬?

value 값을 어떻게 parsing할 것인가, 어떻게 이해할 것인가

tf_decode_csv 가지고, 콤마로 분리시켜서 가져올 수도 있다.

읽어올 때 각각의 필드에 해당하는 값이 어떤 데이터 타입인지를 정해줄 수 있다.

record_defaults



이게 다쥬





### tf.train.batch

https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-04-4-tf_reader_linear_regression.py

```python
# collect batches of csv in
train_x_batch, train_y_batch = \
    tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)
```

값을 tf의 batch를 가지고 읽어오게 된다.

일종의 펌프 역할을 해서 데이터를 읽어온다.

읽어온 x 데이터와 똑같은 형태, 뒤에는 y data

이것을 배치하라.. x, y

batch_size: 한 번에 몇 개씩 가져올꺼냐



```python
sess = tf.Session()
...

# Start populating the filename queue.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    ...

coord.request_stop()
coord.join(threads)
```

coord, threads 요렇게 쓰신다고 보면 된다. queue를 관리..

복잡한 부분은 tf가 다 알아서 해준다.

sess와 coord를 넣어준다..



loop를 돌 때.. train_x_batch와 train_x_batch도 tense쥬?

session을 가지고 실행을 시킨다..

이 값을 추후에 feed-dict같은 것을 사용해서 값을 넘겨주면 된다.



좀 복잡해 보이지만 일반적으로 이렇게 하기 때문에 그대로 따라하면 된다.



```python
import tensorflow as tf

filename_queue = tf.train.string_input_producer(
    ['data-01-test-score.csv'], shuffle=False, name='filename_queue')

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

# collect batches of csv in
train_x_batch, train_y_batch = \
    tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)
```



csv 파일 하나만 쓸 꺼니까 하나만 갖고옴

reader 정의



갖고 있는 value를 csv로 decode한다. 각 field의 data type 넘겨준다.



xy 읽어준 것을 batch new 해서 펌프 빨아들이듯이 읽어온다.

노드 이름 train_x_batch, train_y_batch로 정해준다.

x가 무엇인지, y가 무엇인지 정의

한 번 펌프할 때마다 몇 개씩 가져올지도 정의



shape, random_normal 안에 x, y의 갯수 맞춰줘야 한다.

shape만 신경써주면 됨.



그 다음에 hypothesis, cost 등등 정의



```python
# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Start populating the filename queue.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

coord.request_stop()
coord.join(threads)
```

학습하는 과정 보자



sess 만든 다음

coord, threads 부분은 일반적으로 쓰는 부분이다.



(for문) session 안에서 batch를 뽑아서, 펌프질을 해서 데이터를 가져온다.

이 데이터를 필요하면 한 번 출력해봐도 됨.

학습을 시킬 때  x_batch, y_batch를 넘겨준다. 이렇게 학습 가능.





### shuffle_batch

원하신다면 batch 순서 섞기도 가능.

```python
# min_after_dequeue defines how big a buffer we will randomly sample
# 	from -- bigger means better shuffling but slower start up and mor
# 	memory used.
# capacity must be larger than min_after_dequeue and the amount Larger
# 	determines the maximum we will prefetch. Recommendation:
#     min_after_dequque + (num_threads + a small safety margin) * batch_size

min_after_dequeue = 10000
capacity = min_after_dequeue + 3 * batch_size
example_batch, label_batch = tf.train.shuffle_batch(
	[example, label], batch_size=batch_size, capacity=capacity,
	min_after_dequeue=min_after_dequeue
)
```

다른 굉장히 유용한 함수 있으니 참고하면 좋을 듯.


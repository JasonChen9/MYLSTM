# 编写自己的LSTM模型

之前我通过调包成功使用了LSTM模型来进行训练，但是由于是调包的模型因此对模型内部的结构理解的还不是非常清楚，这次我通过我自己的理解来编写自己的LSTM模型

首先我们先定义最简单的循环神经网络单元：

```python
import tensorflow as tf

# 定义简单循环神经网络单元


class RNN:
    def __init__(self, batchsize, length):
        self.batchsize = batchsize
        self.outputshape = length

    def _input_add_state(self, input, state, active_fn=tf.nn.sigmoid, name=None):
        inputshape = input.get_shape().as_list()
        with tf.variable_scope(name):
            # 初始化变量
            u = tf.get_variable(name='U', initializer=tf.random_uniform((inputshape[-1], self.outputshape)))
            w = tf.get_variable(name='W', initializer=tf.random_uniform((self.outputshape, self.outputshape)))
            b = tf.get_variable(name='B', initializer=tf.random_uniform((inputshape[0], self.outputshape)))
            # 返回最简单的循环神经网络单元
            # output = new_state = act(W * input + U * state + B)
            return active_fn(tf.matmul(input, w) + tf.matmul(state, u) + b)
```

之后我们继承RNN类来定义LSTM单元类：

```python
class LSTMCELL(RNN):
    def __init__(self, batchsize, length):
        # 继承父类
        super().__init__(batchsize, length)
        # 成员方法

    def build(self, inputs, state, candidate, reuse=False):
        with tf.variable_scope('LSTM', reuse=reuse):
            self.state = state
            self.candidate = candidate
            # 遗忘门
            forget = self._input_add_state(inputs, self.state,  name='forget')
            # 输入门
            inputgate = self._input_add_state(inputs, self.state, name='inputgate')
            # 输出门
            output = self._input_add_state(inputs, self.state, name='output')
            # 计算输出值
            self.candidate = tf.multiply(forget, self.state) + tf.multiply(inputgate,self._input_add_state(inputs, self.state,
                                                                                                     tf.nn.tanh,
                                                                                                     name='candidate'))
            # 计算状态值
            self.state = tf.multiply(output, tf.nn.tanh(self.candidate))
        # 返回输出值和状态值
        return self.candidate,self.state
```

最后继承LSTMCELL类定义LSTM模型类：

```python
class LSTM(LSTMCELL):
    def __init__(self, batchsize, length, num_steps):
        # 继承父类
        super().__init__(batchsize, length)
        # 赋值成员变量得到单元个数
        self.num_steps = num_steps

    # 定义方法
    def buildlstm(self, input_data, state, candidate, reuse=False):
        self.resue = reuse
        self.state = state
        self.candidate = candidate
        # 循环神经网络 传递输出值和状态值
        for steps in range(self.num_steps-1):
            self.candidate, self.state = self.bulid(input_data[:, [steps], :],self.state,self.candidate, reuse)

        # 返回最后输出
        return self.state
```

# 总结

通过本次小小的实验我比较清晰的了解了LSTM网络的大致轮廓和内部结构。该代码只是一个简单的框架，用来学习该模型的原理，尚未可以实际运用。
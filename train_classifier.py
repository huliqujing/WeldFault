import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 加载训练数据和测试数据
train = pd.read_csv('./train.csv')
test = pd.read_csv('./Test.csv')
x_train = train.iloc[:, 1:-1]
y_train = train.iloc[:, -1]
x_test = test.iloc[:, 1:-1]
y_test = test.iloc[:, -1]

# 转换特征值的数据类型，使之与后面数据运算时数据类型一致
x_train = tf.cast(x_train, dtype=tf.float32)  # 调用tensorflow中的cast方法，将x_train中的特征值类型转换为float32
x_test = tf.cast(x_test, dtype=tf.float32)  # 调用tensorflow中的cast方法，将x_test中的特征值类型转换为float32

# 用from_tensor_slices方法将特征值和标签值配对
# 将训练集的特征x_train和标签y_train配对，用batch方法将1880个训练数据分成32个为一组的batch,所以共分出1880/32=59个batch
train_data_batch = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
# 将测试集的特征x_test和标签y_test配对，用batch方法将467个测试数据分成32个为一组的批次,所以共分出467/32=15个batch
test_data_batch = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)  # 将测试集的特征x_test和标签y_test配对，用batch方法将467个测试数据分成32个为一组的批次

# 用truncated_normal方法构建神经网络，并用Variable方法标记可训练数据
w1 = tf.Variable(tf.random.truncated_normal([3780, 4], stddev=0.1, seed=1))  # 用truncated_normal方法,构建3780个输入特征，4个分类的神经网络结构，标准差为0.1的正态分布，随机种子为1
b1 = tf.Variable(tf.random.truncated_normal([4], stddev=0.1, seed=1))  # 用truncated_normal方法,因为b1和w1的分类维度要一样，所以是4，标准差为0.1的正态分布，随机种子为1

# 设置变量
learnRate = 0.1  # 学习率为0.1

train_loss_results = []  # 将每轮epoch的loss记录在此列表中，为后面画loss曲线时提供数据

test_accuracy = []  # 将每轮的精度accuracy记录在此列表中，为后面画精度accuracy曲线提供数据

epoch = 500  # 循环500轮

loss_all = 0  # 每轮分59个step,loss_all记录59个step生成的59个loss的和

# 训练部分
for epoch in range(epoch):  # 遍历数据集，每个epoch循环一次数据集
    for step, (x_train, y_train) in enumerate(train_data_batch):  # 遍历batch，每个step循环一次batch，故step从0循环到58
        with tf.GradientTape() as tape:  # 用上下文管理器记录梯度信息
            y = tf.matmul(x_train, w1) + b1  # 神经网络乘加运算，用tensorflow中的matmul方法将训练特征值x_train和w1参数进行矩阵相乘
            y = tf.nn.softmax(y)  # 用tensorflow中的softmax方法将神经网络乘加运算后得到的输出符合正态分布，输出和为1，可以在之后用来与独热码相减求loss
            y_one_hot = tf.one_hot(y_train, depth=4)  # 用tensorflow中的one_hot方法将训练标签y_train转换为独热码格式，因为y输出为4，所以深度为4，方便接下来计算loss的和
            loss = tf.reduce_mean(tf.square(y_one_hot - y))  # 用tensorflow中的reduce_mean方法求平均值，用tensorflow中的square方法求平方，这里用均方误差求损失函数loss
            loss_all += loss.numpy()  # 将每个step计算出的loss累加，后面可以用来求loss平均值，

        # 计算loss对各个参数的梯度
        loss_gradient = tape.gradient(loss, [w1, b1])  # 用tensorflow中的GradientTape方法中的gradient方法求loss对各个参数w1,b1的梯度gradient

        # 梯度更新
        w1.assign_sub(learnRate * loss_gradient[0])  # 用assign_sub方法进行自减，实现参数w1的自动更新,等价于w1 = w1 - learn_Rate * loss_gradient[0]
        b1.assign_sub(learnRate * loss_gradient[1])  # 用assign_sub方法进行自减，实现参数b1的自动更新,等价于b = b - learn_Rate * loss_gradient[1]

    # 每个epoch,打印loss信息
    print("epoch: {}, loss: {}".format(epoch, loss_all / 59))  # 每个epoch,打印loss信息,有59个step，所以总loss_all要除以59，求得每次step的平均loss
    train_loss_results.append(loss_all / 59)  # 用append方法将59个step的loss求平均值记录在train_loss_results中
    loss_all = 0  # loss_all归零，为下一个epoch的求loss做准备

    # 测试部分
    total_correct = 0  # total_correct为预测对的样本个数，初始化为0
    total_test_number = 0  # total_number为测试的总样本数，初始化为0

    for x_test, y_test in test_data_batch:  # 遍历测试集的特征值和标签值
        # 用更新后的参数进行预测
        y = tf.matmul(x_test, w1) + b1  # 用tensorflow中的matmul方法来进行乘加运算，再加上b1得到前向传播的结果
        y = tf.nn.softmax(y)  # 用tensorflow中的softmax方法将神经网络乘加运算后得到的前向传播的结果符合正态分布，输出和为1，可以在之后用来与独热码相减求loss
        predict = tf.argmax(y, axis=1)  # 用tensorflow中的argmax方法，返回y中最大值的索引，即预测的标签分类，axis表示按列求值
        predict = tf.cast(predict, dtype=y_test.dtype)  # 将predict的类型转换为测试集标签y_test的数据类型
        correct = tf.cast(tf.equal(predict, y_test), dtype=tf.int32)  # 用tensorflow中的equal方法判断，若分类正确，则值为1，否则为0，并用tensorflow中的cast方法将bool类型转化为int32类型
        correct = tf.reduce_sum(correct)  # 用tensorflow中的reduce_sum方法将每个batch的correct数加起来
        total_correct += int(correct)  # 将所有batch中的correct数转化为int类型，并加起来
        total_test_number += x_test.shape[0]  # 用shape方法返回测试集特征x_test的行数，也就是测试的总样本数

    accuracy = total_correct / total_test_number  # 总的准确率
    test_accuracy.append(accuracy)  # 测试集的准确率添加到列表中来，方便记录
    print("test_accuracy:", accuracy)  # 打印测试集精度准确率
    print("-------------------------------------------------")  # 为每个epoch进行分隔，方便查看

# 绘制loss曲线
plt.title('Loss Function Curve')  # 用matplotlib中的title方法标出图片标题
plt.xlabel("Epoch")  # 用matplotlib中的xlabel方法标出x轴变量名称
plt.ylabel("Loss")  # 用matplotlib中的ylabel方法标出y轴变量名称
plt.plot(train_loss_results, label="$Loss$")  # 用matplotlib中的plot方法逐点画出训练集损失值结果train_loss_results值并连线，连线的标签为Loss
plt.legend()  # 用matplotlib中的legend方法画出曲线图标
plt.show()  # 用matplotlib中的show方法画出图像

# 绘制accuracy曲线
plt.title("Accuracy Curve")  # 用matplotlib中的title方法标出图片标题
plt.xlabel("Epoch")  # 用matplotlib中的xlabel方法标出x轴变量名称
plt.ylabel("Accuracy")  # 用matplotlib中的ylabel方法标出y轴变量名称
plt.plot(test_accuracy, label="$Accuracy$")  ##用matplotlib中的plot方法逐点画出测试集精准度test_accuracy值并连线，连线的标签为Accuracy
plt.legend()  # 用matplotlib中的legend方法画出曲线图标
plt.show()  # 用matplotlib中的show方法画出图像
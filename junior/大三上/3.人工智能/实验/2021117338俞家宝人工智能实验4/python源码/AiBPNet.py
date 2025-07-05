import numpy as np


# 定义激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 定义BP神经网络类
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 随机初始化权重
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

    # 前向传播
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    # 反向传播
    def backward(self, X, y, learning_rate):
        m = X.shape[0]

        # 计算输出层的误差
        delta2 = self.a2 - y

        # 计算隐藏层的误差
        delta1 = np.dot(delta2, self.W2.T) * self.a1 * (1 - self.a1)

        # 更新权重和偏置
        dW2 = np.dot(self.a1.T, delta2) / m
        db2 = np.sum(delta2, axis=0) / m
        dW1 = np.dot(X.T, delta1) / m
        db1 = np.sum(delta1, axis=0) / m

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    # 训练模型
    def train(self, X, y, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            # 前向传播
            output = self.forward(X)

            # 反向传播
            self.backward(X, y, learning_rate)

            # 计算损失
            loss = np.mean(-y * np.log(output) - (1 - y) * np.log(1 - output))

            # 每隔一段时间输出损失
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    # 预测
    def predict(self, X):
        output = self.forward(X)
        predictions = np.round(output)
        return predictions


# 读取数据
def load_data(image_file, label_file):
    with open(label_file, 'rb') as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    with open(image_file, 'rb') as f:
        images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(len(labels), -1)
    return images, labels


# 加载训练集和测试集数据
train_images, train_labels = load_data('train-images.idx3-ubyte', 'train-labels.idx1-ubyte')
test_images, test_labels = load_data('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')

# 数据预处理
train_images = train_images / 255.0
test_images = test_images / 255.0

# 将标签转换为独热编码
num_classes = 10
train_labels = np.eye(num_classes)[train_labels]
test_labels = np.eye(num_classes)[test_labels]

# 创建并训练神经网络模型
input_size = train_images.shape[1]
hidden_size = 64
output_size = num_classes
numepochs = 1000
learning_rate = 0.1

model = NeuralNetwork(input_size, hidden_size, output_size)
model.train(train_images, train_labels, numepochs, learning_rate)

# 在测试集上进行预测
predictions = model.predict(test_images)

# 计算准确率
accuracy = np.mean(predictions == test_labels)
print("Test Accuracy:", accuracy * 100, '%')

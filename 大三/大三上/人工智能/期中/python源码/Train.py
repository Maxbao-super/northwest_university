import torch
from matplotlib import pyplot as plt
from torch import nn, optim

from AnimalIdentify import LoadData, Module

train_loader = LoadData.firstDataSet()
# model = Module.Net()
model = Module.Net()
model.load_state_dict(torch.load('model.pth'))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 100
for epoch in range(num_epochs):
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, running_loss / len(train_loader)))

print("Finished Training")

# 8. 测试模型

model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print("Accuracy on the test set: {:.2f}%".format(100 * correct / total))
    # 保存模型
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved to 'model.pth'")

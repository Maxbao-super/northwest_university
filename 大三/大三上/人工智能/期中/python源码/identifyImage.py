import torch
import torchvision.transforms as transforms
import Module

def preIdentify(image):
    # 加载模型
    model = Module.Net()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    # 图片预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 图片预处理和模型推理
    input_tensor = transform(image).unsqueeze(0)
    output = model(input_tensor)

    # 获取预测结果
    _, predicted_idx = torch.max(output, 1)
    predicted_class = predicted_idx.item()


    # 输出预测结果
    return predicted_class

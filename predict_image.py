import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

# ===== 和训练时完全一样的模型结构 =====
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ===== 加载训练好的模型 =====
model = Net()
model.load_state_dict(torch.load("mnist_cnn.pth", map_location="cpu"))
model.eval()

# ===== 图像预处理（关键）=====
transform = transforms.Compose([
    transforms.Grayscale(),        # 转灰度
    transforms.Resize((28, 28)),   # MNIST 尺寸
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ===== 读取并预测图片 =====
img = Image.open("my_digit.jpg")
img = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(img)
    pred = output.argmax(dim=1)

print("Predicted digit:", pred.item())
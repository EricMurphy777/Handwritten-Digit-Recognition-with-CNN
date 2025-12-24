import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

# ===== model definition（和训练时一模一样）=====
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
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
        return self.fc2(x)

# ===== load model =====
model = SimpleCNN()
model.load_state_dict(torch.load("mnist_cnn.pth", map_location="cpu"))
model.eval()

# ===== image preprocessing =====
img = Image.open(r"C:\Users\EricM\Desktop\AI_digit project\my_digit.jpg").convert("L")
img = img.resize((28, 28))
img = transforms.functional.invert(img)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

img_tensor = transform(img).unsqueeze(0)

# ===== inference =====
with torch.no_grad():
    output = model(img_tensor)
    probs = F.softmax(output, dim=1)
    pred = probs.argmax(dim=1)

print("Predicted digit:", pred.item())
print("Probabilities:")
for i in range(10):
    print(f"{i}: {probs[0][i].item():.2f}")
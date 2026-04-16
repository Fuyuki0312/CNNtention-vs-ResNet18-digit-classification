from model import Model_detecting_number
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split

device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters --------------------------------------------------------

torch.manual_seed(42)
BATCH_SIZE = 128
MODEL_ADDRESS = "ModelDetectingNumber.pth"

# Load data --------------------------------------------------------------

test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((90, 140)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

full_data = datasets.ImageFolder(root="numbers", transform=test_transform)

train_size = int(0.8 * len(full_data))
train_data, test_data = random_split(
    full_data,
    [train_size, len(full_data) - train_size],
    generator=torch.Generator().manual_seed(42)
)

test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True
)

# Load model --------------------------------------------------------------

God_of_Number = Model_detecting_number()
checkpoint = torch.load(f=MODEL_ADDRESS, map_location=device, weights_only=True)
God_of_Number.load_state_dict(checkpoint["model_state_dict"])
God_of_Number.to(device)

# Test --------------------------------------------------------------------

label_true, label_pred = [], []
for images, labels in test_dataloader:
    images, labels = images.to(device), labels.to(device)

    God_of_Number.eval()
    with torch.inference_mode():
        test_pred_logits = God_of_Number(images)

    test_argmax_pred = torch.argmax(test_pred_logits, dim=1)
    label_pred.extend(test_argmax_pred.cpu().detach().numpy())
    label_true.extend(labels.cpu().detach().numpy())

# Plot confusion matrix --------------------------------------------------------

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(label_true, label_pred)

plt.imshow(cm)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")

for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="white")

plt.colorbar()
plt.show()
from model import Model_detecting_number
import torch
from torchvision import transforms
from PIL import Image

# Hyperparameters --------------------------------------------

MODEL_ADDRESS = "ModelDetectingNumber.pth"

# Setups -----------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((90, 140)),
    transforms.CenterCrop((90, 140)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.286,),
                         std=(0.353,))
])

God_of_Number = Model_detecting_number()
checkpoint = torch.load(f=MODEL_ADDRESS, weights_only=True, map_location=device)
God_of_Number.load_state_dict(checkpoint["model_state_dict"])
God_of_Number.to(device)

# Inference ------------------------------------------------

God_of_Number.eval()
print('input "quit" to end the program')
user_input = str(input("Enter an image's name to test this model (make sure to have both this file and image's file in the same folder or both have same address): "))

while True:
    if user_input == "quit":
        break
    try:
        input_image = transform(Image.open(user_input)).to(device).unsqueeze(dim=0)
        with torch.inference_mode():
            model_logits = God_of_Number(input_image)
        pred_prob = torch.softmax(model_logits, dim=1)
        result = torch.argmax(pred_prob, dim=1).item()
        print("This is number", result)

    except Exception as e:
        print(e)
        print("Please try again")

    user_input = str(input("Enter an image's name: "))


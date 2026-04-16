from model import Model_detecting_number
import torch
from torchvision import transforms

# Hyperparameters --------------------------------------------

MODEL_ADDRESS = "ModelDetectingNumber.pth"

# Setups -----------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((90, 140)),
    transforms.CenterCrop((90, 140)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,),
                         std=(0.5,))
])

God_of_Number = Model_detecting_number()
checkpoint = torch.load(f=MODEL_ADDRESS, weights_only=True, map_location=device)
God_of_Number.load_state_dict(checkpoint["model_state_dict"])
God_of_Number.to(device)
God_of_Number.eval()

from PIL import Image
import numpy as np

def answer_from_model(user_input):

    # These 4 lines of code convert Gradio's drawn image to image that can be sent to the model
    if isinstance(user_input, dict):
        user_input = user_input["composite"]
    if isinstance(user_input, np.ndarray):
        user_input = Image.fromarray(user_input.astype("uint8"))

    input_image = transform(user_input).unsqueeze(0).to(device)
    with torch.inference_mode():
        outputs = God_of_Number(input_image)
    probs = torch.softmax(outputs, dim=1).squeeze()
    return {str(i): float(probs[i].item()) for i in range(10)} # gradio's output "label" expects a dictionary, not tensor



# Interface -------------------------------------------------

import gradio as gr

title = "CNNtention in Digit Classification\n"
description = "This model is built to predict a single number (0-9) drawn in the paper below.\n"

main = gr.Interface(
    fn=answer_from_model,
    inputs=gr.Sketchpad(), # Gradio's drawing paper
    outputs="label",
    title=title,
    description=description
).launch(share=True)
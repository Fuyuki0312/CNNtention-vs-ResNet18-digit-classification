from model import Model_detecting_number
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split

device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters --------------------------------------------

TRANING_CYCLES = 5
BATCH_SIZE = 128
TEST_AFTER_n_EPOCH = 1
MODEL_ADDRESS = "ModelDetectingNumber.pth"
torch.manual_seed(42)

# Load data --------------------------------------------------

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((90, 140)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.286,),
                         std=(0.353,))
])

full_data = datasets.ImageFolder(
    root="numbers",
    transform=transform
)

train_size = int(0.8 * len(full_data))
test_size = len(full_data) - train_size

train_data, test_data = random_split(
    full_data,
    [train_size, test_size]
)

train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=4
)

test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    num_workers=4
)

# Load model -------------------------------------------------

def main():

    God_of_Number = Model_detecting_number()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=God_of_Number.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        factor=0.2,
        patience=3,
        threshold=0.1,
        min_lr=0.00000001
    )

    try:
        checkpoint = torch.load(f=MODEL_ADDRESS, map_location=device, weights_only=True)
        God_of_Number.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch = checkpoint["epoch"]

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    except FileNotFoundError:
        print(f"File {MODEL_ADDRESS} not found")
        print("Trying to create a new model")
        epoch = 0
    except Exception as e:
        print(e)
        print("Trying to create a new model")
        epoch = 0

    God_of_Number.to(device)

    # Training ---------------------------------------------------

    def accuracy_func(pred, true):
        acc_tensor = torch.eq(true, torch.argmax(pred, dim=1))
        acc = torch.sum(acc_tensor).item() / len(true)
        return acc * 100

    epochs = epoch + TRANING_CYCLES
    
    for epoch in range(epoch+1, epochs+1):
        print("Processing Training Epoch " + str(epoch) + "/" + str(TRANING_CYCLES) + "...")
        sum_loss, sum_acc = 0, 0
        
        God_of_Number.train()
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)

            pred_logits = God_of_Number(images)

            batch_loss = loss_func(pred_logits, labels)
            batch_acc = accuracy_func(pred=pred_logits, true=labels)

            optimizer.zero_grad()

            batch_loss.backward()

            optimizer.step()

            sum_loss += batch_loss
            sum_acc += batch_acc

        num_batches = len(train_dataloader)
        loss = (sum_loss) / num_batches
        acc = (sum_acc) / num_batches
        scheduler.step(loss)

        if epoch % TEST_AFTER_n_EPOCH == 0:
            test_sum_loss, test_sum_acc = 0, 0
            
            God_of_Number.eval()
            for images, labels in test_dataloader:
                images, labels = images.to(device), labels.to(device)
                with torch.inference_mode():
                    test_pred_logits = God_of_Number(images)
                
                test_batch_loss = loss_func(test_pred_logits, labels)
                test_batch_acc = accuracy_func(pred=test_pred_logits, true=labels)

                test_sum_loss += test_batch_loss
                test_sum_acc += test_batch_acc

            test_num_batches = len(test_dataloader)
            test_loss = (test_sum_loss) / test_num_batches
            test_acc = (test_sum_acc) / test_num_batches

            print(f"Epoch: {epoch} | Loss: {loss:.6f} | Accuracy: {acc:.2f}%")
            print(f"                 Test loss: {test_loss} | Test accuracy: {test_acc}")

    # Save model ---------------------------------------------------------------------

    torch.save(obj={"model_state_dict": God_of_Number.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch},
               f=MODEL_ADDRESS)

    print(f"Model has been saved successfully as {MODEL_ADDRESS}")

if __name__ == "__main__":
    main()


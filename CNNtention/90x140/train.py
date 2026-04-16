# Import stuff --------------------------------------------------

from model import Model_detecting_number
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import time

# Hyperparameters --------------------------------------------

TRAINING_CYCLES = 50 # How many epochs to train
BATCH_SIZE = 32
LEARNING_RATE = 0.00005
TEST_AFTER_n_EPOCH = 1
MODEL_ADDRESS = "ModelDetectingNumber.pth" # Where to save model's weights
torch.manual_seed(42)

START_TIME = int(time.time())

if TEST_AFTER_n_EPOCH > TRAINING_CYCLES:
    print("WARNING: TEST_AFTER_n_EPOCH is currently bigger than TRAINING_CYCLES.")
    print("Since model with the highest test accuracy will be saved to ensure model's performance,")
    print("this warning means that there will be no test and model will NOT be saved.")

# Load data --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

train_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((90, 140)),

    # Augmentation (for robustness)
    # You can delete 4 "#" at the very left of the 4 lines below to make the code valid
    # (Since my dataset is already really diffucult, I commented these lines)

#    transforms.RandomAffine(
#        degrees=5,
#        translate=(0.05, 0.05)
#    ),

    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((90, 140)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

train_dataset = datasets.ImageFolder(root="numbers", transform=train_transform)
test_dataset = datasets.ImageFolder(root="numbers", transform=test_transform)
full_data = datasets.ImageFolder(root="numbers", transform=None)

train_size = int(0.8 * len(full_data))

train_indices, test_indices = random_split(
    range(len(full_data)),
    [train_size, len(full_data) - train_size],
    generator=torch.Generator().manual_seed(42)
)

train_data = torch.utils.data.Subset(train_dataset, train_indices.indices)
test_data = torch.utils.data.Subset(test_dataset, test_indices.indices)

train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=8,
    persistent_workers=True,
    prefetch_factor=2
)

test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    num_workers=8,
    persistent_workers=True,
    prefetch_factor=2
)

# Load model ----------------------------------------------------------

def main():

    God_of_Number = Model_detecting_number()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=God_of_Number.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4 # to prevent overfitting
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        factor=0.2,
        patience=3,
        threshold=0.03,
        min_lr=0.00000001
    )

    try:
        checkpoint = torch.load(f=MODEL_ADDRESS, map_location=device, weights_only=False)
        God_of_Number.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch = checkpoint["epoch"]
        highest_test_accuracy = checkpoint["best_acc"] # For saving model with the highest test accuracy during training

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    except FileNotFoundError:
        print(f"File {MODEL_ADDRESS} not found")
        print("Trying to create a new model")
        epoch = 0
        highest_test_accuracy = 0

    except Exception as e:
        print(e)
        print("Trying to create a new model")
        epoch = 0
        highest_test_accuracy = 0

    God_of_Number.to(device)

# Training loop ---------------------------------------------------------------

    def accuracy_func(pred, true):
        acc_tensor = torch.eq(true, torch.argmax(pred, dim=1))
        acc = torch.sum(acc_tensor).item() / len(true)
        return acc * 100

    def save(model, optimizer, scheduler, epoch, MODEL_ADDRESS, highest_test_accuracy): # Saving model function
        torch.save(obj={"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                        "best_acc": highest_test_accuracy},
                   f=MODEL_ADDRESS)

        print(f"Model has been saved successfully as {MODEL_ADDRESS}")

    epochs = epoch + TRAINING_CYCLES

    train_losses, val_losses, train_acc, val_acc = [], [], [], [] # For drawing loss curve and acc curve
    for epoch in range(epoch+1, epochs+1):
        print("Processing Training Epoch " + str(epoch) + "/" + str(epochs) + "...")
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
        train_losses.append(loss)
        acc = (sum_acc) / num_batches
        train_acc.append(acc)
        scheduler.step(loss)

        # Validation
        if epoch % TEST_AFTER_n_EPOCH == 0:
            test_sum_loss, test_sum_acc = 0, 0

            for images, labels in test_dataloader:
                images, labels = images.to(device), labels.to(device)
                God_of_Number.eval()
                with torch.inference_mode():
                    test_pred_logits = God_of_Number(images)
                
                test_batch_loss = loss_func(test_pred_logits, labels)
                test_batch_acc = accuracy_func(pred=test_pred_logits, true=labels)

                test_sum_loss += test_batch_loss
                test_sum_acc += test_batch_acc

            test_num_batches = len(test_dataloader)
            test_loss = (test_sum_loss) / test_num_batches
            val_losses.append(test_loss)
            test_acc = (test_sum_acc) / test_num_batches
            val_acc.append(test_acc)

            print(f"Epoch: {epoch} | Train Loss: {loss:.6f} | Train Accuracy: {acc:.2f}%")
            print(f"           Test Loss: {test_loss:.6f} | Test Accuracy: {test_acc:.2f}%")

            # Save model with the highest test accuracy so far
            if test_acc > highest_test_accuracy:
                highest_test_accuracy = test_acc
                saved_epoch = epoch
                save(model=God_of_Number, optimizer=optimizer, scheduler=scheduler, epoch=epoch, MODEL_ADDRESS=MODEL_ADDRESS, highest_test_accuracy=highest_test_accuracy)

        print("")

    try:
        print(f"Model's weights in epoch {saved_epoch} is save as {MODEL_ADDRESS}")

    except Exception:
        pass

    END_TIME = int(time.time())
    print(END_TIME - START_TIME)

# Plot loss curve --------------------------------------------------------------

    def to_numpy(tensor_list): # plt.plot wants numpy arrays (not tensors)
        numpy_list = []
        for i in tensor_list:
            numpy_list.append(i.to("cpu").detach().numpy())
        return numpy_list

    train_losses, val_losses = to_numpy(train_losses), to_numpy(val_losses)

    import matplotlib.pyplot as plt

    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xticks([])

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")

    plt.legend()
    plt.show()

# Plot accuracy curve ------------------------------------------------------------

    plt.plot(train_acc, label="Train")
    plt.plot(val_acc, label="Validation")
    plt.xticks([])

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")

    plt.legend()
    plt.show()

# Start the program ----------------------------------------------------------------

if __name__ == "__main__":
    main()
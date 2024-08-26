import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision.utils import make_grid
import torch.nn as nn
import torch.optim as optim
import time
import os

generator = torch.manual_seed(42)

#train and test data directory
data_dir = "../intel-image-class2/resources/seg_train"
test_data_dir = "../intel-image-class2/resources/seg_test"
params_dir = "../intel-image-class2/params"

num_class = 6

#load the train and test data
train_dataset = ImageFolder(data_dir, transform = transforms.Compose([
    transforms.Resize((150,150)),transforms.ToTensor()
]))
val_dataset = ImageFolder(test_data_dir, transforms.Compose([
    transforms.Resize((150,150)),transforms.ToTensor()
]))

img, label = train_dataset[0]


def display_img(img,label):
    print(f"Label : {train_dataset.classes[label]}")
    plt.imshow(img.permute(1,2,0))
    plt.show()


def show_batch(dl):
    """Plot images grid of single batch"""
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        plt.show()
        break


class VGG16(nn.Module):
    def __init__(self, num_classes):
        super(VGG16, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # We will calculate the number of features automatically based on the input size
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.avgpool(x)
        x = self.fc_layers(x)
        return x

val_size = 2000
train_size = len(train_dataset) - val_size

# hyper-params
batch_size = 4
lr = 0.001
num_epochs = 10

train_data,val_data = random_split(train_dataset, [train_size, val_size], generator=generator)

#load the train and validation into batches.
train_dl = DataLoader(train_data, batch_size, shuffle=True, generator=generator)
val_dl = DataLoader(val_data, batch_size * 2, shuffle=False, generator=generator)

# define the model
model = VGG16(num_classes=num_class)

# we will try loading model parameters. if an error occurs we will generate new model parameters.
try:
    files = os.listdir(params_dir)

    # if files is empty, raise an exception (which wil be handled by the except clause)
    if len(files) == 0:
        raise Exception('No params file found.')

    # path of the most recent params file
    params_path = f"{params_dir}/{max(files)}"

    # Load the parameters from the path
    # (if files is empty, this will raise an exception, which wil be handled by the except clause)
    model.load_state_dict(torch.load(params_path, weights_only=True))

    print("Parameters loaded successfully.")
except Exception:
    print("Failed to load parameters. Make sure you have the './params' dir")

    # @toDO: initialize parameters

# define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# If a GPU is available, move the model to the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def saveModel():
    # concatenate name of the params directory with the current timestamp to obtain the file path.
    params_path = f"{params_dir}/{str(int(time.time()))}.pt"

    torch.save(model.state_dict(), params_path)

    print("Parameters saved successfully.")

# Define the training function
def train_epoch():
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    c = 1

    # for each batch
    for inputs, labels in train_dl:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss and accuracy
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        print("count: " + str(c))

        print("loss: " + str(loss.item()))

        # save model once every 100 batches
        if c % 100 == 1:  # print every once in a while
            saveModel()

        c += 1

    epoch_loss = running_loss / len(train_dl.dataset)
    epoch_accuracy = 100 * correct / total

    return epoch_loss, epoch_accuracy


def train():
    for epoch in range(num_epochs):
        # Train the model
        train_loss, train_accuracy = train_epoch()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

        # Validate the model
        val_loss, val_accuracy = validate()
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    print('Training complete.')

# Define the validation function
def validate():
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_dl:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_dl.dataset)
    val_accuracy = 100 * correct / total

    return val_loss, val_accuracy

if __name__ == '__main__':
    # #display the first image in the dataset
    # display_img(*train_dataset[0])
    #
    # #display the first batch
    # show_batch(train_dl)

    # print the lengths
    print(f"Length of Train Data : {len(train_data)}")
    print(f"Length of Validation Data : {len(val_data)}")

    # output
    # Length of Train Data : 12034
    # Length of Validation Data : 2000

    train()
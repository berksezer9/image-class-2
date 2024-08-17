import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision.utils import make_grid


generator = torch.manual_seed(42)

#train and test data directory
data_dir = "../intel-image-class2/resources/seg_train/"
test_data_dir = "../intel-image-class2/resources/seg_test"


#load the train and test data
dataset = ImageFolder(data_dir,transform = transforms.Compose([
    transforms.Resize((150,150)),transforms.ToTensor()
]))
test_dataset = ImageFolder(test_data_dir,transforms.Compose([
    transforms.Resize((150,150)),transforms.ToTensor()
]))

img, label = dataset[0]

#output :
#torch.Size([3, 150, 150]) 0

def display_img(img,label):
    print(f"Label : {dataset.classes[label]}")
    plt.imshow(img.permute(1,2,0))
    plt.show()


batch_size = 128
val_size = 2000
train_size = len(dataset) - val_size
num_workers = 4

train_data,val_data = random_split(dataset,[train_size,val_size], generator=generator)

#load the train and validation into batches.
train_dl = DataLoader(train_data, batch_size, shuffle = True, num_workers = num_workers, pin_memory = True, generator=generator)
val_dl = DataLoader(val_data, batch_size * 2, num_workers = num_workers, pin_memory = True, generator=generator)


def show_batch(dl):
    """Plot images grid of single batch"""
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        plt.show()
        break

if __name__ == '__main__':
    #display the first image in the dataset
    display_img(*dataset[0])

    #display the first batch
    show_batch(train_dl)

    # print the lengths
    print(f"Length of Train Data : {len(train_data)}")
    print(f"Length of Validation Data : {len(val_data)}")

    # output
    # Length of Train Data : 12034
    # Length of Validation Data : 2000
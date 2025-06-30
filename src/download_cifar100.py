import torchvision
import os

# Specify the target directory
data_path = "/home/dataset/cifar100/"

# Create the directory if it doesn't exist
os.makedirs(data_path, exist_ok=True)

print(f"Downloading CIFAR-100 dataset to {data_path}...")

# Download the training set
trainset = torchvision.datasets.CIFAR100(
    root=data_path,
    train=True,
    download=True
)

# Download the test set
testset = torchvision.datasets.CIFAR100(
    root=data_path,
    train=False,
    download=True
)

print("CIFAR-100 dataset downloaded successfully.")

# Verify the download (optional)
# The dataset files should be inside a 'cifar-100-python' subdirectory
downloaded_path = os.path.join(data_path, 'cifar-100-python')
if os.path.exists(downloaded_path):
    print(f"Dataset found at: {downloaded_path}")
    print("Files in the directory:", os.listdir(downloaded_path))
else:
    print("Error: Dataset directory not found after download.")

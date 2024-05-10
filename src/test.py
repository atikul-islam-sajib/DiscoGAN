import sys
import os
import torch
import matplotlib.pyplot as plt

sys.path.append("src/")

from utils import load
from generator import Generator

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Initialize the generators
netG_XtoY = Generator().to(device)
netG_YtoX = Generator().to(device)


# Load the state dictionaries
state_X = torch.load(
    "/Users/shahmuhammadraditrahman/Desktop/DiscoGAN/checkpoints/train_models_netG_XtoY/netG_XtoY100.pth"
)
netG_XtoY.load_state_dict(state_X)

state_Y = torch.load(
    "/Users/shahmuhammadraditrahman/Desktop/DiscoGAN/checkpoints/train_models_netG_YtoX/netG_YtoX100.pth"
)
netG_YtoX.load_state_dict(state_Y)

# Load data
X, y = next(
    iter(
        load(
            "/Users/shahmuhammadraditrahman/Desktop/DiscoGAN/data/processed/train_dataloader.pkl"
        )
    )
)

# Generate predictions
predict_y = netG_XtoY(X.to(device))
reconstructed_y = netG_YtoX(predict_y)


# Normalization and conversion of tensor to numpy for visualization
def process_image(img):
    img = img.squeeze()  # Remove unnecessary dimensions
    img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
    return (
        img.permute(1, 2, 0).cpu().detach().numpy()
    )  # Rearrange dimensions and convert to numpy


# Process images
X_np = process_image(X)
y_np = process_image(y)
predict_np = process_image(predict_y)
reconstructed_np = process_image(reconstructed_y)

# Plotting
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
axs[0].imshow(X_np)
axs[0].set_title("Original X")
axs[0].axis("off")

axs[1].imshow(y_np)
axs[1].set_title("Original Y")
axs[1].axis("off")

axs[2].imshow(predict_np)
axs[2].set_title("Predicted Y")
axs[2].axis("off")

axs[3].imshow(reconstructed_np)
axs[3].set_title("Reconstructed X")
axs[3].axis("off")

plt.show()

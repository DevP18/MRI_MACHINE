import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import torch
import matplotlib
from scipy.ndimage import zoom
from skimage import exposure
from monai.networks.nets import UNet

root_dir = "/Users/vaasu/MRI_Prection/MRi_Prediction/BraTS-MEN-Train"
device   = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = UNet(
    spatial_dims  = 2,
    in_channels   = 4,
    out_channels  = 4,
    channels      = (16, 32, 64, 128),
    strides       = (2, 2, 2),
    num_res_units = 2,
).to(device)

model.eval()
print(f"Device: {device} | Random weights — waiting for best_seg.pth")

def load_mod(folder, mod, z, size=128):
    path = os.path.join(root_dir, folder, f"{folder}-{mod}.nii.gz")
    data = nib.load(path).get_fdata()[:, :, z]
    mn, mx = data.min(), data.max()
    data = (data - mn) / (mx - mn + 1e-8)
    data = exposure.equalize_adapthist(data, clip_limit=0.03)
    return zoom(data, (size/240, size/240), order=1)

def get_best_z(seg_data):
    counts = [np.count_nonzero(seg_data[:, :, z]) for z in range(seg_data.shape[2])]
    return int(np.argmax(counts))

def predict(folder):
    seg_data = nib.load(os.path.join(root_dir, folder, f"{folder}-seg.nii.gz")).get_fdata()
    best_z   = get_best_z(seg_data)

    t1c = load_mod(folder, "t1c", best_z)
    t1n = load_mod(folder, "t1n", best_z)
    t2f = load_mod(folder, "t2f", best_z)
    t2w = load_mod(folder, "t2w", best_z)

    x = torch.tensor(np.stack([t1c, t1n, t2f, t2w]), dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x).argmax(dim=1).squeeze().cpu().numpy()

    seg_true = zoom(seg_data[:, :, best_z], (128/240, 128/240), order=0)

    return t1c, t1n, t2f, t2w, pred, seg_true, best_z

patients = random.sample([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))], 3)
cmap_seg = matplotlib.colormaps.get_cmap("jet").resampled(4)

fig, axes = plt.subplots(3, 6, figsize=(20, 10))
fig.suptitle("Predicted vs True Seg", fontsize=14)

for col, title in enumerate(["t1c", "t1n", "t2f", "t2w", "predicted", "true seg"]):
    axes[0, col].set_title(title)

for row, folder in enumerate(patients):
    t1c, t1n, t2f, t2w, pred, seg_true, best_z = predict(folder)

    images = [t1c, t1n, t2f, t2w, pred, seg_true]
    cmaps  = ["gray", "gray", "gray", "gray", cmap_seg, cmap_seg]

    for col, (img, cmap) in enumerate(zip(images, cmaps)):
        vmax = 0.7 if col < 4 else 3
        axes[row, col].imshow(img, cmap=cmap, vmin=0, vmax=vmax)  
        axes[row, col].axis("off")

    axes[row, 0].set_ylabel(f"{folder}\nz={best_z}", fontsize=7)

plt.tight_layout()
plt.show()

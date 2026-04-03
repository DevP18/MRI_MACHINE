import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os

file_path = "//path//to//segmentation.nii.gz"  # Update this to your actual file path

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Segmentation file not found: {file_path}")

nii = nib.load(file_path)
data = nii.get_fdata()
print("Shape of the data:", data.shape)
print("Unique labels:", np.unique(data))  # See what label values exist

# Find a slice that actually has tumor labels (not just background)
for axis in range(3):
    sums = np.sum(data > 0, axis=tuple(i for i in range(3) if i != axis))
    best_slice = np.argmax(sums)
    print(f"Axis {axis}: best slice = {best_slice}, nonzero voxels = {sums[best_slice]}")

# Show the best slice along axis 2
slice_idx = np.argmax(np.sum(data > 0, axis=(0, 1)))
print(f"Showing slice {slice_idx}")

plt.figure(figsize=(8, 8))
plt.imshow(data[:, :, slice_idx], cmap='jet', interpolation='none')
plt.title(f"Segmentation - Slice {slice_idx}")
plt.colorbar(label='Label')
plt.show()
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import zoom

def full_audit():
    if not os.path.exists("test_results.txt"):
        print("Run C++ first to create test_results.txt!")
        return

    # Load the results from C++
    results = []
    with open("test_results.txt", "r") as f:
        for line in f:
            results.append(line.split())

    for meta_id, target, score in results:
        score = float(score)
        target = float(target)
        
        # Only show the errors (FAIL cases) to save time
        prediction = 1.0 if score > 0.5 else 0.0
        if prediction == target: continue 

        patient_id, z_slice = meta_id.split('_')
        z_slice = int(z_slice)

        # Load MRI
        img_path = f"BraTS-MEN-Train/{patient_id}/{patient_id}-t1c.nii.gz"
        seg_path = f"BraTS-MEN-Train/{patient_id}/{patient_id}-seg.nii.gz"
        img = nib.load(img_path).get_fdata()[:, :, z_slice]
        seg = nib.load(seg_path).get_fdata()[:, :, z_slice]

        # Process what AI sees (32x32)
        norm = img / (np.max(img) if np.max(img) > 0 else 1)
        ai_input = zoom(norm, (32/img.shape[0], 32/img.shape[1]), order=1)

        # Plotting 4 Panels
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(img, cmap='gray'); axes[0].set_title("Original")
        
        axes[1].imshow(img, cmap='gray')
        axes[1].imshow(np.ma.masked_where(seg == 0, seg), cmap='Reds', alpha=0.5)
        axes[1].set_title(f"Truth: {'TUMOR' if target==1 else 'HEALTHY'}")

        axes[2].imshow(ai_input, cmap='gray', interpolation='nearest')
        axes[2].set_title("AI Input (32x32 View)")

        # PANEL 4: THE AI OUTPUT (THE MATH)
        color = 'red' # Since we are only looking at fails
        axes[3].bar(['Healthy', 'Tumor'], [1-score, score], color=['lightgray', color])
        axes[3].set_ylim(0, 1)
        axes[3].set_title(f"AI Guess: {'TUMOR' if score > 0.5 else 'HEALTHY'}\nConfidence: {score*100:.1f}%")

        for i in range(3): axes[i].axis('off')
        
        plt.savefig(f"audit_{meta_id}.png")
        plt.close()
        print(f"Saved audit for error: {meta_id}")

if __name__ == "__main__":
    full_audit()
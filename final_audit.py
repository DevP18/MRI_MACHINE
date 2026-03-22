import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import zoom

def run_final_report():
    if not os.path.exists("test_results.txt"):
        print("Error: test_results.txt not found. Did you run the updated C++ code?")
        return

    with open("test_results.txt", "r") as f:
        lines = f.readlines()

    print(f"Generating reports for {len(lines)} test cases...")

    for line in lines:
        parts = line.split()
        meta_id, target, score = parts[0], float(parts[1]), float(parts[2])
        
        # Determine if it was an error
        prediction = 1.0 if score > 0.5 else 0.0
        if prediction == target: continue # Skip the ones it got right for now

        # Load MRI
        p_id, z = meta_id.split('_')
        img = nib.load(f"BraTS-MEN-Train/{p_id}/{p_id}-t1c.nii.gz").get_fdata()[:,:,int(z)]
        seg = nib.load(f"BraTS-MEN-Train/{p_id}/{p_id}-seg.nii.gz").get_fdata()[:,:,int(z)]

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Panel 1: Truth
        axes[0].imshow(img, cmap='gray')
        axes[0].imshow(np.ma.masked_where(seg == 0, seg), cmap='Reds', alpha=0.5)
        axes[0].set_title(f"TRUTH: {'TUMOR' if target==1 else 'HEALTHY'}")
        
        # Panel 2: AI View
        low_res = zoom(img / np.max(img), (32/img.shape[0], 32/img.shape[1]), order=1)
        axes[1].imshow(low_res, cmap='gray', interpolation='nearest')
        axes[1].set_title("AI Perspective (32x32)")

        # Panel 3: The "How Much" Score
        color = 'red' if prediction != target else 'green'
        axes[2].bar(['Healthy', 'Tumor'], [1-score, score], color=['gray', color])
        axes[2].set_ylim(0, 1)
        axes[2].set_title(f"AI Guess: {'TUMOR' if score > 0.5 else 'HEALTHY'}\n({score*100:.1f}%)")

        for ax in axes[:2]: ax.axis('off')
        plt.savefig(f"audit_{meta_id}.png")
        plt.close()

if __name__ == "__main__":
    run_final_report()
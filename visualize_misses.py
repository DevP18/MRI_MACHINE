import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import zoom

def detailed_audit(meta_id, ai_raw_score):
    """
    meta_id: e.g. "BraTS-MEN-00621-000_60"
    ai_raw_score: The decimal value from your C++ results (e.g. 0.24)
    """
    patient_id, z_slice = meta_id.split('_')
    z_slice = int(z_slice)

    # Load MRI and Segment
    path = f"BraTS-MEN-Train/{patient_id}/{patient_id}-t1c.nii.gz"
    seg_path = f"BraTS-MEN-Train/{patient_id}/{patient_id}-seg.nii.gz"
    
    img = nib.load(path).get_fdata()[:, :, z_slice]
    seg = nib.load(seg_path).get_fdata()[:, :, z_slice]

    # Pre-process for "AI View"
    norm = img / (np.max(img) if np.max(img) > 0 else 1)
    ai_view = zoom(norm, (32, 32), order=1) # Match your 32x32

    # Determine confidence string
    confidence = ai_raw_score * 100
    guess = "TUMOR" if ai_raw_score > 0.5 else "HEALTHY"
    actual = "TUMOR" if np.sum(seg) > 0 else "HEALTHY"
    result = "CORRECT" if guess == actual else "ERROR"

    # Create 4-panel plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. Original
    axes[0].imshow(img, cmap='gray'); axes[0].set_title("Original Scan")
    
    # 2. Ground Truth
    axes[1].imshow(img, cmap='gray')
    axes[1].imshow(np.ma.masked_where(seg == 0, seg), cmap='Reds', alpha=0.5)
    axes[1].set_title(f"Truth: {actual}")

    # 3. What AI Sees (Resolution check)
    axes[2].imshow(ai_view, cmap='gray', interpolation='nearest')
    axes[2].set_title("AI Input (32x32)")

    # 4. AI Result
    color = 'green' if result == "CORRECT" else 'red'
    axes[3].bar(['Healthy', 'Tumor'], [1-ai_raw_score, ai_raw_score], color=[ 'gray', color])
    axes[3].set_ylim(0, 1)
    axes[3].set_title(f"AI Guess: {guess}\n({confidence:.1f}% Sure)")

    for i in range(3): axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"audit_detail_{meta_id}.png")
    print(f"Saved detailed audit to audit_detail_{meta_id}.png")

if __name__ == "__main__":
    # Change the score (0.0 to 1.0) based on your C++ getresults() output
    detailed_audit("BraTS-MEN-00621-000_60", 0.15) # Example: AI was 15% sure of tumor (Wrong)
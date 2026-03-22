import nibabel as nib
import numpy as np
import os
import random
from scipy.ndimage import zoom

root_dir = "BraTS-MEN-Train"
TARGET_TRAIN = 1000
TARGET_TEST  = 100
RESIZE_TO    = (32, 32)

MODALITIES = ["t1c", "t1n", "t2f", "t2w"]

def process_patient_list(folders, target_count):
    samples  = []
    metadata = []
    t_needed = target_count // 2
    h_needed = target_count - t_needed
    t_found, h_found = 0, 0

    for folder in folders:
        if t_found >= t_needed and h_found >= h_needed:
            break

        seg_path = os.path.join(root_dir, folder, f"{folder}-seg.nii.gz")
        if not os.path.exists(seg_path):
            continue

        try:
            seg_data = nib.load(seg_path).get_fdata()
        except:
            continue

        for z in range(40, 130):
            if t_found >= t_needed and h_found >= h_needed:
                break

            nonzero    = np.count_nonzero(seg_data[:, :, z])
            is_tumor   = nonzero > 50
            is_healthy = nonzero == 0

            want_tumor   = is_tumor   and t_found < t_needed
            want_healthy = is_healthy and h_found < h_needed

            if not (want_tumor or want_healthy):
                continue

            modal_slices = []
            for mod in MODALITIES:
                mod_path = os.path.join(root_dir, folder, f"{folder}-{mod}.nii.gz")
                if not os.path.exists(mod_path):
                    break
                try:
                    slice_data = nib.load(mod_path).get_fdata()[:, :, z]
                    min_val = np.min(slice_data)
                    max_val = np.max(slice_data)
                    if max_val > min_val:
                        slice_data = (slice_data - min_val) / (max_val - min_val)
                    else:
                        slice_data = np.zeros_like(slice_data)
                    scale   = (RESIZE_TO[0] / slice_data.shape[0], RESIZE_TO[1] / slice_data.shape[1])
                    low_res = zoom(slice_data, scale, order=1)
                    modal_slices.append(low_res.flatten())
                except:
                    break

            if len(modal_slices) != 4:
                continue

            combined = np.concatenate(modal_slices)
            label    = "1.0" if is_tumor else "0.0"
            samples.append(f"{label} " + " ".join(map(str, combined)))
            metadata.append(f"{folder}_{z}")

            if is_tumor: t_found += 1
            else:        h_found += 1

    print(f"  Class split: {t_found} tumor | {h_found} healthy  (target {t_needed}/{h_needed})")
    return samples, metadata


def create_dataset():
    all_patients = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    random.shuffle(all_patients)

    split_idx      = int(len(all_patients) * 0.8)
    train_patients = all_patients[:split_idx]
    test_patients  = all_patients[split_idx:]

    print(f"Processing Training Set from {len(train_patients)} patients...")
    train_samples, _ = process_patient_list(train_patients, TARGET_TRAIN)

    print(f"Processing Testing Set from {len(test_patients)} patients...")
    test_samples, test_meta = process_patient_list(test_patients, TARGET_TEST)

    with open("mri_train.txt", "w") as f:
        for s in train_samples: f.write(s + "\n")

    with open("mri_test.txt", "w") as f:
        for s in test_samples: f.write(s + "\n")

    with open("mri_test_meta.txt", "w") as f:
        for m in test_meta: f.write(m + "\n")

    print(f"\nSUCCESS!")
    print(f"Train samples: {len(train_samples)}")
    print(f"Test samples:  {len(test_samples)}")

if __name__ == "__main__":
    create_dataset()
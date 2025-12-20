import os
import cv2
import matplotlib.pyplot as plt

def visualize_samples(base_path, split='train', num_samples=5):
    """
    Hiển thị ngẫu nhiên các bộ ba ảnh (Before, After, Label) từ dataset để kiểm tra sự đồng bộ và nội dung.
    """
    path_a = os.path.join(base_path, split, 'A')
    files = sorted(os.listdir(path_a))[:num_samples]
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    for i, fname in enumerate(files):
        img_a = cv2.cvtColor(cv2.imread(os.path.join(base_path, split, 'A', fname)), cv2.COLOR_BGR2RGB)
        img_b = cv2.cvtColor(cv2.imread(os.path.join(base_path, split, 'B', fname)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(base_path, split, 'label', fname), 0)
        
        axes[i, 0].imshow(img_a); axes[i, 0].set_title(f"A: {fname}")
        axes[i, 1].imshow(img_b); axes[i, 1].set_title("B (After)")
        axes[i, 2].imshow(mask, cmap='gray'); axes[i, 2].set_title("Label")
        for ax in axes[i]: ax.axis('off')
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    visualize_samples('/kaggle/working/satellite_full_dataset', split='train')
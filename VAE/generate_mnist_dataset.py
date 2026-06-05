import os
import tempfile
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch_fidelity import calculate_metrics
import lpips
import warnings
warnings.filterwarnings("ignore")

# ========================= CONFIGURATION =========================
K_FOLDS = 5
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
MAX_LPIPS_SAMPLES = 1000

VAE_DIR = "VAE_generated_mnist"   # ← Change if path is different
DM_DIR  = "DM_generated_mnist"    # ← Change if path is different

# --------------------- LOAD YOUR CLASSIFIER -----------------------
# Replace this section with your actual trained MNIST classifier
class YourMNISTClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Load or define your model here
        pass
    def forward(self, x):
        # x shape: (B, 1, 28, 28) in [0,1]
        # return logits (B, 10)
        return torch.zeros(x.size(0), 10, device=x.device)  # placeholder

classifier = YourMNISTClassifier().to(DEVICE)
classifier.eval()
print("Classifier loaded.")
# -----------------------------------------------------------------

# --------------------- LPIPS -------------------------------------
lpips_model = lpips.LPIPS(net='alex', version='0.1').to(DEVICE)
lpips_model.eval()
print("LPIPS model loaded.")
# -----------------------------------------------------------------

# --------------------- REAL MNIST TEST SET -----------------------
real_dir = "real_mnist_test"
if not os.path.exists(real_dir):
    print("Creating real MNIST test reference directory...")
    os.makedirs(real_dir, exist_ok=True)
    test_dataset = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor()),
        batch_size=1, shuffle=False
    )
    for idx, (img, _) in enumerate(test_dataset):
        img_pil = transforms.ToPILImage()(img.squeeze(0))
        img_pil.save(os.path.join(real_dir, f"{idx:05d}.png"))
    print(f"Real MNIST test set saved to {real_dir} ({len(os.listdir(real_dir))} images)")
else:
    print(f"Using existing real MNIST test directory: {real_dir}")
# -----------------------------------------------------------------

# --------------------- CUSTOM DATASET FOR FOLDING ----------------
class ImageFolderWithLabels(Dataset):
    """Loads images from class subfolders (0-9) and keeps labels"""
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        
        for label in range(10):
            class_dir = os.path.join(root_dir, str(label))
            if not os.path.exists(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, fname))
                    self.labels.append(label)
        
        print(f"Loaded {len(self.image_paths)} images from {root_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('L')          # grayscale
        img = transforms.ToTensor()(img)                 # [0,1], shape (1, 28, 28)
        return img, label

# Load full datasets
print("\nLoading generated datasets...")
vae_dataset = ImageFolderWithLabels(VAE_DIR)
dm_dataset  = ImageFolderWithLabels(DM_DIR)

# Convert to tensors for easy k-fold indexing
vae_images = torch.stack([vae_dataset[i][0] for i in range(len(vae_dataset))])
vae_labels = torch.tensor([vae_dataset[i][1] for i in range(len(vae_dataset))])

dm_images = torch.stack([dm_dataset[i][0] for i in range(len(dm_dataset))])
dm_labels = torch.tensor([dm_dataset[i][1] for i in range(len(dm_dataset))])

print(f"VAE: {len(vae_dataset)} images")
print(f"DM : {len(dm_dataset)} images")
# -----------------------------------------------------------------

# --------------------- HELPER FUNCTIONS --------------------------
def compute_accuracy(images, labels, classifier, device, batch_size=64):
    dataset = torch.utils.data.TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = total = 0
    with torch.no_grad():
        for batch_imgs, batch_labels in loader:
            batch_imgs = batch_imgs.to(device)
            logits = classifier(batch_imgs)
            pred = logits.argmax(dim=1)
            correct += (pred == batch_labels.to(device)).sum().item()
            total += batch_labels.size(0)
    return correct / total

def compute_lpips(gen_images, real_dir, lpips_model, device, max_samples=1000):
    # Sample from generated
    N = min(len(gen_images), max_samples)
    gen_idx = torch.randperm(len(gen_images))[:N]
    gen_sample = gen_images[gen_idx].repeat(1, 3, 1, 1).to(device)   # to RGB
    
    # Sample from real
    real_files = [f for f in os.listdir(real_dir) if f.endswith('.png')]
    real_idx = torch.randperm(len(real_files))[:N]
    real_imgs = []
    for i in real_idx:
        img = Image.open(os.path.join(real_dir, real_files[i])).convert('L')
        real_imgs.append(transforms.ToTensor()(img))
    real_sample = torch.stack(real_imgs).repeat(1, 3, 1, 1).to(device)
    
    # Resize to 64x64 for LPIPS
    gen_sample = torch.nn.functional.interpolate(gen_sample, size=(64, 64), mode='bilinear')
    real_sample = torch.nn.functional.interpolate(real_sample, size=(64, 64), mode='bilinear')
    
    with torch.no_grad():
        dist = lpips_model(gen_sample, real_sample).mean().item()
    return dist
# -----------------------------------------------------------------

# --------------------- MAIN K-FOLD VALIDATION --------------------
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
results = {}

for mode_name, (images, labels, source_dir) in [
    ("VAE_only", (vae_images, vae_labels, VAE_DIR)),
    ("DM_only",  (dm_images, dm_labels, DM_DIR)),
    ("Combination", (torch.cat([vae_images, dm_images], dim=0),
                     torch.cat([vae_labels, dm_labels], dim=0), None))
]:
    print(f"\n{'='*70}")
    print(f"VALIDATING MODE: {mode_name}  ({images.shape[0]} images)")
    print(f"{'='*70}")
    
    acc_list = []
    fid_list = []
    kid_list = []
    is_list = []
    lpips_list = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(range(len(images)))):
        print(f"  Fold {fold_idx+1}/{K_FOLDS} ...", end=" ")
        
        fold_images = images[test_idx]
        fold_labels = labels[test_idx]
        
        # 1. Accuracy using your classifier
        acc = compute_accuracy(fold_images, fold_labels, classifier, DEVICE)
        acc_list.append(acc)
        
        # Prepare temporary directory for torch-fidelity (flat folder)
        with tempfile.TemporaryDirectory() as tmp_gen_dir:
            # Save fold images as flat PNGs (torch-fidelity expects flat directory)
            for i, img in enumerate(fold_images):
                img_pil = transforms.ToPILImage()(img)
                img_pil.save(os.path.join(tmp_gen_dir, f"{i:06d}.png"))
            
            # 2-4. FID, KID, IS
            metrics = calculate_metrics(
                input1=tmp_gen_dir,
                input2=real_dir,
                cuda=torch.cuda.is_available(),
                batch_size=BATCH_SIZE,
                isc=True,
                fid=True,
                kid=True,
                verbose=False
            )
            
            fid_list.append(metrics['frechet_inception_distance'])
            kid_list.append(metrics['kernel_inception_distance_mean'])
            is_list.append(metrics['inception_score_mean'])
        
        # 5. LPIPS
        lpips_score = compute_lpips(fold_images, real_dir, lpips_model, DEVICE, MAX_LPIPS_SAMPLES)
        lpips_list.append(lpips_score)
        
        print(f"ACC={acc:.4f}  FID={metrics['frechet_inception_distance']:.2f}  IS={metrics['inception_score_mean']:.2f}")
    
    # Aggregate results
    results[mode_name] = {
        "ACC":   f"{np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}",
        "FID":   f"{np.mean(fid_list):.2f} ± {np.std(fid_list):.2f}",
        "KID":   f"{np.mean(kid_list):.4f} ± {np.std(kid_list):.4f}",
        "IS":    f"{np.mean(is_list):.2f} ± {np.std(is_list):.2f}",
        "LPIPS": f"{np.mean(lpips_list):.4f} ± {np.std(lpips_list):.4f}"
    }
    
    print(f"  → Final: ACC={results[mode_name]['ACC']} | FID={results[mode_name]['FID']} | "
          f"KID={results[mode_name]['KID']} | IS={results[mode_name]['IS']} | LPIPS={results[mode_name]['LPIPS']}")

# --------------------- SUMMARY TABLE -----------------------------
print(f"\n{'='*85}")
print("FINAL 5-FOLD VALIDATION RESULTS")
print(f"{'='*85}")
print(f"{'Mode':<15} {'ACC ↑':<15} {'FID ↓':<15} {'KID ↓':<15} {'IS ↑':<15} {'LPIPS ↓':<15}")
print("-" * 85)
for mode, r in results.items():
    print(f"{mode:<15} {r['ACC']:<15} {r['FID']:<15} {r['KID']:<15} {r['IS']:<15} {r['LPIPS']:<15}")
print("-" * 85)
print("Interpretation: Higher ACC and IS are better. Lower FID, KID, and LPIPS are better.")
# -----------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# === ADD THIS AT THE TOP ===
if __name__ == '__main__':
    # ====================== CLASSIFIER ARCHITECTURE ======================
    class MNISTClassifier(nn.Module):
        def __init__(self):
            super(MNISTClassifier, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(128, 10)
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)  # Flatten
            x = self.classifier(x)
            return x

    # ====================== DATA LOADING ======================
    def load_synthetic_dataset(root_dir, transform):
        return datasets.ImageFolder(root=root_dir, transform=transform)

    # ====================== MAIN ======================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    print(f"Original MNIST test set loaded: {len(test_dataset)} images\n")

    # ====================== UPDATED: Include Original MNIST + VAE + DM ======================
    datasets_to_train = {
        "Original": None,
        "VAE": "validation/VAE_results",
        "DM": "validation/DM_results"
    }

    results = {}  # Will store losses and final accuracy for plotting

    for name, data_dir in datasets_to_train.items():
        print(f"{'='*60}\nTraining classifier on {name} dataset\n{'='*60}")
        
        # Load appropriate training dataset
        if name == "Original":
            train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            print(f"Original MNIST train set loaded: {len(train_dataset)} images")
        else:
            if not os.path.exists(data_dir):
                print(f"⚠️  {data_dir} not found! Skipping.")
                continue
            train_dataset = load_synthetic_dataset(data_dir, transform)
            print(f"{name} synthetic dataset loaded: {len(train_dataset)} images")
        
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=False)

        model = MNISTClassifier().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 10
        epoch_losses = []

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} ({name})"):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(train_loader)
            epoch_losses.append(avg_loss)
            print(f"  Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

        # Test on real MNIST (same for all classifiers)
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Testing on real test set ({name})"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"\n✅ Training on {name} → Accuracy on real MNIST: {accuracy:.2f}%\n")

        # Save model (generalized naming)
        save_name = "original" if name == "Original" else name
        torch.save(model.state_dict(), f"classifier_trained_on_{save_name}.pth")
        print(f"   Model saved as classifier_trained_on_{save_name}.pth\n")

        # Store results for plotting
        results[name] = {'losses': epoch_losses, 'accuracy': accuracy}

    print("All experiments finished!")

    # ====================== PLOTTING (after ALL training completed) ======================
    if results:
        plt.figure(figsize=(14, 6))

        # Plot 1: Training Loss Curves (one line per classifier)
        plt.subplot(1, 2, 1)
        for name, data in results.items():
            plt.plot(range(1, num_epochs + 1), data['losses'], marker='o', linewidth=2, label=name)
        plt.title('Training Loss per Epoch for Each Classifier')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.legend()
        plt.grid(True)

        # Plot 2: Final Test Accuracy Bar Chart
        plt.subplot(1, 2, 2)
        names = list(results.keys())
        accuracies = [results[n]['accuracy'] for n in names]
        bars = plt.bar(names, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        plt.title('Test Accuracy on Real MNIST Test Set')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 100)
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
        plt.grid(True, axis='y')

        plt.tight_layout()
        plt.savefig('classifier_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Plots saved as 'classifier_results.png' and displayed above.")
    else:
        print("No results to plot.")
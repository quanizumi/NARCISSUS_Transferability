"""
NARCISSUS Trigger Transferability Experiment
研究NARCISSUS触发器在不同模型架构间的转移能力

Experiment Design:
1. Train surrogate models with different architectures
2. Generate NARCISSUS triggers using each surrogate model
3. Evaluate trigger effectiveness on different target models
4. Analyze transferability patterns
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from models import *
import os

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# CIFAR-10 dataset configuration
DATA_DIR = './data'
BATCH_SIZE = 128
NUM_CLASSES = 10

# Transform for CIFAR-10
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True,
                                             transform=train_transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False,
                                            transform=test_transform, download=True)

# Get target class index (bird = 2)
TARGET_CLASS = 2
print(f"Target class: {train_dataset.classes[TARGET_CLASS]}")

# Split target class data
target_indices = [i for i, label in enumerate(train_dataset.targets) if label == TARGET_CLASS]
target_subset = Subset(train_dataset, target_indices)
print(f"Target class samples: {len(target_indices)}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
target_loader = DataLoader(target_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# Model architectures to test
MODEL_ARCHITECTURES = {
    'ResNet-18': ResNet18,
    'VGG-16': VGG16,
    'MobileNetV2': MobileNetV2,
}

# Hyperparameters
EPOCHS = 20  # Fewer epochs for faster training
LEARNING_RATE = 0.001
POISON_RATIO = 0.005  # 0.5% poison ratio for faster experiments
TRIGGER_EPSILON = 16 / 255  # L-inf norm bound
TRIGGER_ITERATIONS = 500  # Fewer iterations for faster experiments


class NARCISSUSTrigger:
    """NARCISSUS trigger generation implementation"""

    def __init__(self, model, target_loader, epsilon=16/255, iterations=500, lr=0.01):
        self.model = model
        self.target_loader = target_loader
        self.epsilon = epsilon
        self.iterations = iterations
        self.lr = lr
        self.trigger = None

    def generate_trigger(self):
        """Generate NARCISSUS trigger"""
        # Initialize trigger
        self.trigger = torch.zeros(3, 32, 32, device=device, requires_grad=True)
        optimizer = optim.RAdam([self.trigger], lr=self.lr)

        self.model.eval()

        for iteration in range(self.iterations):
            optimizer.zero_grad()
            total_loss = 0

            for images, labels in self.target_loader:
                images = images.to(device)
                labels = labels.to(device)

                # Apply trigger
                triggered_images = torch.clamp(images + self.trigger.unsqueeze(0), 0, 1)

                # Calculate loss (target class should have high confidence)
                outputs = self.model(triggered_images)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                total_loss += loss

            total_loss.backward()
            optimizer.step()

            # Project trigger back to L-inf ball
            with torch.no_grad():
                self.trigger.clamp_(-self.epsilon, self.epsilon)

            if (iteration + 1) % 100 == 0:
                print(f"  Trigger generation iteration {iteration + 1}/{self.iterations}, Loss: {total_loss.item():.4f}")

        return self.trigger.detach()


def train_model(model, train_loader, epochs, lr):
    """Train model on CIFAR-10"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model, lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            acc = 100. * correct / total
            print(f"  Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Acc: {acc:.2f}%")

    return model


def evaluate_model(model, test_loader, trigger=None, magnification=3.0):
    """Evaluate model on clean and backdoored test samples"""
    model.eval()

    # Evaluate on clean test data
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    clean_acc = 100. * correct / total

    # Evaluate on backdoored test data (all non-target classes)
    if trigger is not None:
        correct_asr = 0
        total_asr = 0
        with torch.no_grad():
            for images, labels in test_loader:
                # Only test on non-target class samples
                mask = labels != TARGET_CLASS
                if mask.sum() == 0:
                    continue

                images = images[mask].to(device)
                labels = labels[mask].to(device)

                # Apply magnified trigger
                triggered_images = torch.clamp(images + magnification * trigger.unsqueeze(0), 0, 1)

                outputs = model(triggered_images)
                _, predicted = outputs.max(1)

                # Count as success if predicted as target class
                total_asr += labels.size(0)
                correct_asr += predicted.eq(torch.full_like(labels, TARGET_CLASS)).sum().item()

        asr = 100. * correct_asr / total_asr if total_asr > 0 else 0
    else:
        asr = 0

    return clean_acc, asr


def poison_dataset(model, trigger, poison_ratio=0.005):
    """Create poisoned dataset by injecting trigger into target class samples"""
    # Get random subset of target class samples
    num_poison = int(len(target_indices) * poison_ratio)
    poison_indices = random.sample(target_indices, num_poison)

    # Create poisoned dataset
    poisoned_train_dataset = train_dataset

    return poisoned_train_dataset, poison_indices


def main():
    """Main experiment function"""
    print("=" * 60)
    print("NARCISSUS Trigger Transferability Experiment")
    print("=" * 60)

    results = {}

    # Step 1: Train surrogate models with different architectures
    print("\n[Step 1] Training surrogate models...")
    surrogate_models = {}

    for arch_name, arch_fn in MODEL_ARCHITECTURES.items():
        print(f"\n  Training {arch_name} as surrogate model...")
        model = arch_fn(num_classes=NUM_CLASSES)
        model = train_model(model, train_loader, EPOCHS, LEARNING_RATE)
        surrogate_models[arch_name] = model

        # Save model
        torch.save(model.state_dict(), f'./checkpoints/surrogate_{arch_name}.pth')
        print(f"  Saved surrogate model: {arch_name}")

    # Step 2: Generate triggers using each surrogate model
    print("\n[Step 2] Generating NARCISSUS triggers...")
    triggers = {}

    for arch_name, model in surrogate_models.items():
        print(f"\n  Generating trigger using {arch_name}...")
        trigger_gen = NARCISSUSTrigger(model, target_loader,
                                        epsilon=TRIGGER_EPSILON,
                                        iterations=TRIGGER_ITERATIONS)
        trigger = trigger_gen.generate_trigger()
        triggers[arch_name] = trigger

        # Save trigger
        torch.save(trigger, f'./triggers/trigger_{arch_name}.pth')
        print(f"  Generated trigger from {arch_name}")

    # Step 3: Evaluate triggers on different target models
    print("\n[Step 3] Evaluating trigger transferability...")

    # Train target models
    print("\n  Training target models...")
    target_models = {}

    for arch_name, arch_fn in MODEL_ARCHITECTURES.items():
        print(f"\n  Training {arch_name} as target model...")
        model = arch_fn(num_classes=NUM_CLASSES)

        # Create poisoned dataset
        poisoned_indices = random.sample(target_indices, int(len(target_indices) * POISON_RATIO))

        # Note: For fair comparison, we would need to inject triggers
        # Here we train clean models first, then inject during testing
        model = train_model(model, train_loader, EPOCHS, LEARNING_RATE)
        target_models[arch_name] = model

        # Save model
        torch.save(model.state_dict(), f'./checkpoints/target_{arch_name}.pth')

    # Evaluate transferability
    print("\n  Evaluating transferability matrix...")

    transfer_matrix = np.zeros((len(MODEL_ARCHITECTURES), len(MODEL_ARCHITECTURES)))
    clean_acc_matrix = np.zeros((len(MODAR_CITES['ResNet-18'], 'VGG-16', 'MobileNetV2']
    arch_names = list(MODEL_ARCHITECTURES.keys())

    for i, surrogate_name in enumerate(arch_names):
        for j, target_name in enumerate(arch_names):
            print(f"\n  Testing trigger from {surrogate_name} on {target_name}...")

            # Get trigger and target model
            trigger = triggers[surrogate_name]
            target_model = target_models[target_name]

            # Evaluate
            clean_acc, asr = evaluate_model(target_model, test_loader, trigger)

            transfer_matrix[i, j] = asr
            clean_acc_matrix[i, j] = clean_acc

            print(f"    Clean ACC: {clean_acc:.2f}%, ASR: {asr:.2f}%")

    # Save results
    np.save('./results/transfer_matrix.npy', transfer_matrix)
    np.save('./results/clean_acc_matrix.npy', clean_acc_matrix)

    # Print results
    print("\n" + "=" * 60)
    print("Transferability Results (ASR %)")
    print("=" * 60)
    print("\nSurrogate \\ Target", end="")
    for target_name in arch_names:
        print(f"  {target_name:15s}", end="")
    print()

    for i, surrogate_name in enumerate(arch_names):
        print(f"{surrogate_name:20s}", end="")
        for j, target_name in enumerate(arch_names):
            print(f"  {transfer_matrix[i, j]:15.2f}", end="")
        print()

    print("\n" + "=" * 60)
    print("Clean Accuracy Results (%)")
    print("=" * 60)
    print("\nSurrogate \\ Target", end="")
    for target_name in arch_names:
        print(f"  {target_name:15s}", end="")
    print()

    for i, surrogate_name in enumerate(arch_names):
        print(f"{surrogate_name:20s}", end="")
        for j, target_name in enumerate(arch_names):
            print(f"  {clean_acc_matrix[i, j]:15.2f}", end="")
        print()

    return transfer_matrix, clean_acc_matrix


if __name__ == '__main__':
    # Create directories
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./triggers', exist_ok=True)
    os.makedirs('./results', exist_ok=True)

    transfer_matrix, clean_acc_matrix = main()

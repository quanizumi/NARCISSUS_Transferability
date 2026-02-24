"""
NARCISSUS Trigger Transferability Analysis
基于论文数据的深度分析和扩展实验

This script analyzes the transferability patterns from the original paper
and provides additional analysis tools.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Original paper data from Table 5
# Surrogate-Target (Sur-Tar) model pairs on CIFAR-10
original_data = {
    ('ResNet-18', 'ResNet-18'): 97.36,
    ('ResNet-18', 'GoogLeNet'): 99.55,
    ('ResNet-18', 'EfficientNet-B0'): 99.97,
    ('GoogLeNet', 'ResNet-18'): 99.50,
    ('GoogLeNet', 'GoogLeNet'): 100.00,
    ('GoogLeNet', 'EfficientNet-B0'): 100.00,
    ('EfficientNet-B0', 'ResNet-18'): 82.05,
    ('EfficientNet-B0', 'GoogLeNet'): 100.00,
    ('EfficientNet-B0', 'EfficientNet-B0'): 87.82,
}

# Model information (approximate parameter counts)
model_info = {
    'ResNet-18': {'params': 11.2, 'depth': 18, 'type': 'residual'},
    'GoogLeNet': {'params': 6.8, 'depth': 22, 'type': 'inception'},
    'EfficientNet-B0': {'params': 5.3, 'depth': 'varies', 'type': 'efficient'},
}

# Additional experimental settings from the paper
# Attack metrics from Table III
attack_results = {
    'CIFAR-10': {
        'clean_acc': 95.59,
        'poison_ratio': 0.05,
        'asr_ours': 97.36,
        'asr_lc': 3.21,
        'asr_htba': 4.87,
        'asr_saa': 6.00,
    },
    'PubFig': {
        'clean_acc': 93.64,
        'poison_ratio': 0.024,
        'asr_ours': 99.89,
        'asr_lc': 0.15,
    },
    'Tiny-ImageNet': {
        'clean_acc': 64.82,
        'poison_ratio': 0.05,
        'asr_ours': 85.81,
        'asr_lc': 1.72,
    },
}

def analyze_transferability():
    """Analyze transferability patterns from the original data"""
    print("=" * 70)
    print("NARCISSUS Trigger Transferability Analysis")
    print("=" * 70)

    # Create transferability matrix
    models = ['ResNet-18', 'GoogLeNet', 'EfficientNet-B0']
    n = len(models)
    matrix = np.zeros((n, n))

    for i, surrogate in enumerate(models):
        for j, target in enumerate(models):
            matrix[i, j] = original_data.get((surrogate, target), 0)

    print("\n1. Original Transferability Matrix (ASR %)")
    print("-" * 50)
    print(f"{'Surrogate\\Target':<20}", end="")
    for model in models:
        print(f"{model:>15}", end="")
    print()

    for i, surrogate in enumerate(models):
        print(f"{surrogate:<20}", end="")
        for j, target in enumerate(models):
            print(f"{matrix[i, j]:>15.2f}", end="")
        print()

    # Analyze diagonal vs off-diagonal
    diagonal = np.diag(matrix)
    off_diagonal = matrix[~np.eye(n, dtype=bool)].reshape(n, n-1)

    print("\n2. Key Findings")
    print("-" * 50)
    print(f"Diagonal (same architecture): {np.mean(diagonal):.2f}%")
    print(f"Off-diagonal (cross architecture): {np.mean(off_diagonal):.2f}%")

    # Model performance correlation
    print("\n3. Model Performance Analysis")
    print("-" * 50)

    # GoogLeNet as surrogate gives best results
    print("GoogLeNet as surrogate:")
    print(f"  - Average ASR when used as surrogate: {np.mean(matrix[1, :]):.2f}%")
    print(f"  - Best transferability to other architectures")

    # EfficientNet-B0 shows interesting pattern
    print("\nEfficientNet-B0 as surrogate:")
    print(f"  - Average ASR when used as surrogate: {np.mean(matrix[2, :]):.2f}%")
    print(f"  - Shows high variance in transferability")

    return matrix, models


def calculate_transferability_metrics(matrix, models):
    """Calculate additional transferability metrics"""
    n = len(models)

    # Calculate row-wise (surrogate perspective)
    print("\n4. Surrogate Effectiveness (Row Average)")
    print("-" * 50)
    for i, model in enumerate(models):
        avg = np.mean(matrix[i, :])
        print(f"{model}: {avg:.2f}%")

    # Calculate column-wise (target perspective)
    print("\n5. Target Robustness (Column Average)")
    print("-" * 50)
    for j, model in enumerate(models):
        avg = np.mean(matrix[:, j])
        print(f"{model}: {avg:.2f}%")

    # Transferability drop (off-diagonal vs diagonal)
    print("\n6. Transferability Drop Analysis")
    print("-" * 50)
    for i, surrogate in enumerate(models):
        for j, target in enumerate(models):
            if i != j:
                drop = matrix[i, i] - matrix[i, j]
                print(f"{surrogate} -> {target}: {drop:.2f}% drop")


def create_visualizations(matrix, models):
    """Create visualizations of the transferability analysis"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Heatmap
    ax1 = axes[0]
    sns.heatmap(matrix, annot=True, fmt='.1f', cmap='YlOrRd',
                xticklabels=models, yticklabels=models, ax=ax1,
                vmin=80, vmax=100)
    ax1.set_title('NARCISSUS Trigger Transferability\n(ASR %)')
    ax1.set_xlabel('Target Model')
    ax1.set_ylabel('Surrogate Model')

    # Bar chart - surrogate effectiveness
    ax2 = axes[1]
    surrogate_avg = np.mean(matrix, axis=1)
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax2.bar(models, surrogate_avg, color=colors)
    ax2.set_ylabel('Average ASR (%)')
    ax2.set_title('Surrogate Model Effectiveness')
    ax2.set_ylim([80, 105])
    for bar, val in zip(bars, surrogate_avg):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom')

    # Bar chart - target robustness
    ax3 = axes[2]
    target_avg = np.mean(matrix, axis=0)
    bars = ax3.bar(models, target_avg, color=colors)
    ax3.set_ylabel('Average ASR (%)')
    ax3.set_title('Target Model Robustness')
    ax3.set_ylim([80, 105])
    for bar, val in zip(bars, target_avg):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('./analysis/transferability_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\nVisualization saved to: ./analysis/transferability_analysis.png")


def compare_with_baselines():
    """Compare NARCISSUS with baseline attacks"""
    print("\n" + "=" * 70)
    print("Comparison with Baseline Attacks (CIFAR-10)")
    print("=" * 70)

    attacks = ['NARCISSUS', 'LC', 'HTBA', 'SAA', 'BadNets-c', 'Blend-c']
    asr_values = [97.36, 3.21, 4.87, 6.00, 2.60, 1.40]

    print(f"\n{'Attack':<15} {'ASR (%)':>10}")
    print("-" * 30)
    for attack, asr in zip(attacks, asr_values):
        print(f"{attack:<15} {asr:>10.2f}")

    # Calculate improvement
    best_baseline = max([3.21, 4.87, 6.00, 2.60, 1.40])
    improvement = (97.36 - best_baseline) / best_baseline * 100
    print(f"\nNARCISSUS improvement over best baseline: {improvement:.1f}%")

    return attacks, asr_values


def analyze_defense_resistance():
    """Analyze defense resistance patterns"""
    print("\n" + "=" * 70)
    print("Defense Resistance Analysis")
    print("=" * 70)

    defenses = ['Neural Cleanse', 'Fine-Pruning', 'I-BAU', 'Frequency Detector', 'ABL']
    effectiveness = ['Failed', 'Limited', 'Unstable', 'Bypassable', 'Failed']

    print(f"\n{'Defense':<20} {'Effectiveness':>15}")
    print("-" * 40)
    for defense, eff in zip(defenses, effectiveness):
        print(f"{defense:<20} {eff:>15}")

    print("\nKey Insight:")
    print("NARCISSUS triggers contain features as persistent as semantic features,")
    print("making them resistant to removal without hurting model accuracy.")


def generate_hypothesis():
    """Generate research hypotheses for extension"""
    print("\n" + "=" * 70)
    print("Research Hypotheses for Extension")
    print("=" * 70)

    hypotheses = [
        "H1: Triggers generated from higher-capacity models transfer better",
        "H2: Transferability is symmetric - A->B ≈ B->A",
        "H3: Model similarity (depth, type) affects transferability",
        "H4: Trigger characteristics correlate with transferability",
    ]

    print("\nProposed Hypotheses:")
    for h in hypotheses:
        print(f"  • {h}")

    # Test H2 symmetry
    print("\nTesting H2 (Symmetry):")
    print(f"  ResNet-18->GoogLeNet: {original_data[('ResNet-18', 'GoogLeNet')]}%")
    print(f"  GoogLeNet->ResNet-18: {original_data[('GoogLeNet', 'ResNet-18')]}%")
    print(f"  Difference: {abs(original_data[('ResNet-18', 'GoogLeNet')] - original_data[('GoogLeNet', 'ResNet-18')]):.2f}%")

    # Test H1 with EfficientNet
    print("\nTesting H1 (Model Capacity):")
    print(f"  GoogLeNet (6.8M params) as surrogate: avg {np.mean([97.36, 99.50, 100.00]):.2f}%")
    print(f"  EfficientNet-B0 (5.3M params) as surrogate: avg {np.mean([82.05, 100.00, 87.82]):.2f}%")


def main():
    """Main analysis function"""
    import os
    os.makedirs('./analysis', exist_ok=True)

    # Run all analyses
    matrix, models = analyze_transferability()
    calculate_transferability_metrics(matrix, models)
    create_visualizations(matrix, models)
    compare_with_baselines()
    analyze_defense_resistance()
    generate_hypothesis()

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()

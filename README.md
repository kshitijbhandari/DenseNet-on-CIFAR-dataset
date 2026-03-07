# DenseNet on CIFAR-10

Implementation of a **Densely Connected Convolutional Network (DenseNet)** from scratch using TensorFlow/Keras, trained and evaluated on the CIFAR-10 dataset. Achieves **~90% test accuracy** after 100 epochs with data augmentation.

---

## Table of Contents
- [Background](#background)
- [Architecture](#architecture)
- [Experiments](#experiments)
- [Results](#results)
- [Key Findings](#key-findings)
- [How to Run](#how-to-run)
- [References](#references)

---

## Background

DenseNet ([Huang et al., 2017](https://arxiv.org/abs/1608.06993)) introduces dense connectivity: every layer receives feature maps from **all preceding layers** and passes its own to all subsequent layers. This design:

- Alleviates the vanishing gradient problem
- Encourages feature reuse across layers
- Significantly reduces the number of parameters compared to ResNet

CIFAR-10 is a standard benchmark consisting of 60,000 32x32 colour images across 10 classes (50,000 train / 10,000 test).

---

## Architecture

The model follows the original DenseNet paper with a custom configuration suited for CIFAR-10:

```
Input (32x32x3)
    |
Conv2D (3x3)
    |
Dense Block 1  -->  Transition Block 1
    |
Dense Block 2  -->  Transition Block 2
    |
Dense Block 3  -->  Transition Block 3
    |
Dense Block 4
    |
Output Layer (BN -> ReLU -> AvgPool -> Flatten -> Dense(10, softmax))
```

### Dense Block
Each dense block stacks `l` layers. Every layer applies `BN -> ReLU -> Conv2D(3x3)` and concatenates its output with all previous feature maps along the channel axis.

```python
for _ in range(l):
    x = BatchNorm -> ReLU -> Conv2D(3x3)
    x = Concatenate([previous_input, x])
```

### Transition Block
Reduces spatial dimensions and compresses channels between dense blocks:
```
BN -> ReLU -> Conv2D(1x1, filters * compression) -> AveragePooling(2x2)
```

### Hyperparameters (Final Model)

| Parameter        | Value |
|-----------------|-------|
| Dense blocks    | 4     |
| Layers per block (`l`) | 13 |
| Initial filters | 32    |
| Compression     | 0.5   |
| Dropout         | None  |
| Optimizer       | Adam  |
| Loss            | Categorical Crossentropy |

---

## Experiments

Two experiments were conducted:

### Experiment 1 — Baseline (No Augmentation)
- **Filters**: 12, **Layers/block**: 12, **Dropout**: 0.2
- **Epochs**: 10
- **Data augmentation**: None
- **Result**: 68.26% test accuracy

### Experiment 2 — Full Training with Data Augmentation
- **Filters**: 32, **Layers/block**: 13, **Dropout**: Removed
- **Epochs**: 100 (trained incrementally: 30 + 30 + 10 + 10 + 20)
- **Data augmentation**: rotation (40°), width/height shift (0.2), shear (0.2), zoom (0.2), horizontal flip
- **Optimizer**: Adam
- **Result**: 89.91% test accuracy

---

## Results

### Experiment 1 — Training Curve (10 Epochs, No Augmentation)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1     | 1.7212    | 35.80%    | 1.5374   | 45.41%  |
| 2     | 1.3885    | 49.02%    | 1.4239   | 48.94%  |
| 3     | 1.2367    | 54.97%    | 1.3179   | 54.27%  |
| 4     | 1.1316    | 59.09%    | 1.3588   | 53.95%  |
| 5     | 1.0527    | 62.28%    | 1.4094   | 53.31%  |
| 6     | 0.9937    | 64.37%    | 1.0494   | 63.92%  |
| 7     | 0.9503    | 65.81%    | 1.1274   | 61.18%  |
| 8     | 0.9138    | 67.09%    | 0.9729   | 66.12%  |
| 9     | 0.8844    | 68.28%    | 1.1159   | 63.46%  |
| 10    | 0.8566    | 69.34%    | 0.9456   | **68.26%** |

**Final Test Accuracy: 68.26%**

---

### Experiment 2 — Final Epochs (71–80, With Augmentation)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 71    | 0.2496    | 91.14%    | 0.3439   | 89.42%  |
| 72    | 0.2498    | 91.18%    | 0.4382   | 86.67%  |
| 73    | 0.2503    | 91.18%    | 0.3452   | 88.84%  |
| 74    | 0.2525    | 91.01%    | 0.5688   | 84.29%  |
| 75    | 0.2485    | 91.35%    | 0.4957   | 85.55%  |
| 76    | 0.2483    | 91.17%    | 0.4026   | 88.00%  |
| 77    | 0.2426    | 91.53%    | 0.3350   | 89.47%  |
| 78    | 0.2416    | 91.55%    | 0.4326   | 87.53%  |
| 79    | 0.2381    | 91.71%    | 0.3931   | 88.18%  |
| 80    | 0.2382    | 91.66%    | 0.3449   | 89.37%  |

**Final Test Accuracy after 100 epochs: 89.91%**

---

### Summary

| Model                              | Epochs | Test Accuracy |
|------------------------------------|--------|---------------|
| Baseline (no augmentation)         | 10     | 68.26%        |
| With augmentation + more filters   | 100    | **89.91%**    |

---

## Key Findings

1. **Data augmentation is critical**: Moving from 68% to 90% accuracy, augmentation was the single biggest factor — it prevents overfitting on the small 32x32 images.

2. **More filters = better feature capacity**: Increasing initial filters from 12 to 32 gave the network more expressive power without exploding parameters, thanks to the compression factor (0.5).

3. **Dropout was detrimental**: Removing dropout improved accuracy. With dense connectivity and data augmentation providing sufficient regularization, explicit dropout hindered convergence.

4. **Dense connections combat vanishing gradients**: Each layer receives gradients from all subsequent layers, enabling stable training even with 4 deep dense blocks.

5. **Parameter efficiency**: Despite achieving ~90% accuracy, the model uses far fewer parameters than comparable ResNet configurations, owing to feature reuse through concatenation.

---

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Training
Open and run [DenseNet_cifar10.ipynb](DenseNet_cifar10.ipynb) in Jupyter or Google Colab.

The notebook is organized in two sections:
- **Part 1**: Baseline model — quick 10-epoch run to verify architecture
- **Part 2**: Full training with data augmentation over 100 epochs (recommended to run on GPU)

### Generate Analysis Report
```bash
pip install matplotlib fpdf2
python generate_report.py
```
This produces `DenseNet_CIFAR10_Report.pdf` with training curves and result tables.

---

## Project Structure

```
DenseNet-on-CIFAR-dataset/
|-- DenseNet_cifar10.ipynb   # Main notebook (model + training)
|-- generate_report.py       # Script to generate PDF analysis report
|-- requirements.txt         # Python dependencies
|-- README.md
```

---

## References

- [Densely Connected Convolutional Networks — Huang et al. (2017)](https://arxiv.org/abs/1608.06993)
- [CIFAR-10 Dataset — Krizhevsky (2009)](https://www.cs.toronto.edu/~kriz/cifar.html)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/)

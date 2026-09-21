# Neural Networks for Image Classification

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![Torchvision](https://img.shields.io/badge/Torchvision-Computer%20Vision-red?logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA-GPU%20Acceleration-76B900?logo=nvidia)
![Transfer Learning](https://img.shields.io/badge/Transfer%20Learning-ResNet18-success)

## Overview

This project investigates neural-network-based image classification for identifying four different artistic movements:

- Baroque
- Impressionism
- Cubism
- Abstract Expressionism

The project compares several approaches, gradually increasing the architectural complexity:

1. A fully connected feed-forward neural network using flattened pixels.
2. A convolutional neural network without pooling and activation functions.
3. A CNN with padding and max pooling.
4. A CNN with ReLU activations.
5. A CNN with Batch Normalization.
6. Regularization and data augmentation experiments.
7. A frozen pre-trained ResNet18 with a trainable linear classification layer.

The models are evaluated using accuracy, macro F1-score, loss curves, and confusion matrices.

## Dataset

The project uses a **balanced** subset of the [WikiArt](https://www.kaggle.com/datasets/steubk/wikiart) dataset (Steubk, 2023, available on Kaggle). The repository includes it as the compressed archive `wikiart_hw2_balanced.npz.zip`, which should be extracted before running the notebook so that the required `wikiart_hw2_balanced.npz` file is available. It contains 13228 (4×3307) RGB images resized to `32 × 32` pixels and their corresponding class labels.

After extraction, the resulting `.npz` file should be placed in the project directory, unless its path is updated in the notebook.

For convenient inspection, `dataset_extractor.py` can be used to convert the dataset’s raw pixel arrays into viewable PNG images, organized into folders by artistic movement.

Run:
```bash
python dataset_extractor.py
```

## Dependencies
`pip install torch`  
`pip install torchvision`  
`pip install numpy`  
`pip install matplotlib`  
`pip install seaborn`  
`pip install scikit-learn`  
or  
`pip install torch torchvision numpy matplotlib seaborn scikit-learn`

## How to Run

1. Install the required dependencies.
2. Place `wikiart_hw2_balanced.npz` in the current data directory.
3. Open the notebook: `main.ipynb`.
4. Update the dataset path if necessary. For local execution, uncomment the direct-loading line for `wikiart_hw2_balanced.npz`.
5. Run the notebook cells in order.

The notebook includes data exploration, preprocessing, model training, evaluation, visualization, optimizer comparisons, regularization experiments, data augmentation, and transfer learning.

## Reproducibility

A fixed random seed is used throughout the project:

```python
SEED = 42
```
The code also enables deterministic PyTorch operations where possible in order to make the experiments reproducible.

## Main Results

The experiments produced the following representative results:

| Model | Test Accuracy | Test F1 Macro |
| :--- | :---: | :---: |
| Fully Connected Feedforward Network | 55.33% | 0.5544 |
| Custom CNN with regularization and augmentation | 73.60% | 0.7365 |
| Frozen ResNet18 with linear probe | 76.40% | 0.7643 |

The frozen ResNet18 achieved the highest test performance among the evaluated approaches, suggesting that ImageNet-pretrained visual representations can transfer effectively to this artistic movement classification task.

### Best Custom CNN Configuration

The best-performing configuration of our custom CNN model used tuned regularization and data augmentation. This configuration was identified through hyperparameter tuning via grid search. Its specifications were:

- Four convolutional blocks with channel sizes `3 → 16 → 32 → 64 → 128`; each block used a `5 × 5` convolution, Batch Normalization, ReLU activation, and `2 × 2` max pooling.
- A fully connected classification head with layer sizes `512 → 1024 → 256 → 32 → 4`.
- Dropout with probability `0.2` between the fully connected layers.
- Adam optimizer with initial learning rate `0.001` and weight decay `0.0001`.
- Cosine Annealing learning-rate scheduling over `60` epochs.
- Data augmentation using random horizontal flips, random crops with padding, and mild brightness and contrast changes.

This configuration achieved `73.60%` test accuracy and a `0.7365` test macro F1-score.

## Topics Investigated

The project examines:

- Exploratory data analysis
- Stratified train-validation-test splitting
- Feed-forward neural networks
- Convolutional neural networks
- Padding and max-pooling
- ReLU activations
- Batch Normalization
- SGD, SGD with momentum, Adam, RMSprop, and AdamW
- Cosine learning-rate scheduling
- Dropout and weight decay
- Data augmentation
- Transfer learning with ResNet18
- Hyperparameter tuning via grid search
- Filter and activation-map visualization
- CPU and GPU training-time comparison


## Execution Environment and Runtime

All experiments were run on a Google Colab session using a T4 GPU. CPU-based experiments used the CPU available in the respective Colab session for comparison.

### Estimated GPU execution time

~329 minutes

Actual runtime depends on the available hardware, PyTorch version, CUDA configuration, and whether all experiments are executed.

*Initially developed as coursework for EP08: Pattern Recognition - Machine Learning, course of DIT, UoA.*  
*Extensively tuned, optimized, and expanded with hyperparameter tuning, regularization studies, larger dataset and other experiments.*

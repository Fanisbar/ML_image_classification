# Neural Networks for Image Classification

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)

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

The project uses the `wikiart_hw2.npz` dataset. It contains RGB images resized to `32 × 32` pixels and their corresponding class labels.

The dataset file should be placed in the project directory or its path should be updated in the notebook.

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
2. Place `wikiart_hw2.npz` in the current data directory.
3. Open the notebook: `main.ipynb`.
4. Update the dataset path if necessary. For local execution, uncomment the direct-loading line for `wikiart_hw2.npz`.
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
| Fully Connected Network | 48.92% | 0.4883 |
| Custom CNN with regularization and augmentation | 70.43% | 0.7055 |
| Frozen ResNet18 with linear probe | 75.07% | 0.7493 |

The frozen ResNet18 achieved the best performance, despite only training its final linear classification layer. This demonstrates the usefulness of general visual representations learned from ImageNet.

## Topics Investigated

The project examines:

- Exploratory data analysis
- Stratified train-validation-test splitting
- Feed-forward neural networks
- Convolutional neural networks
- Padding and max pooling
- ReLU activations
- Batch Normalization
- SGD, SGD with momentum, Adam, RMSprop, and AdamW
- Cosine learning-rate scheduling
- Dropout and weight decay
- Data augmentation
- Transfer learning with ResNet18
- Filter and activation-map visualization
- CPU and GPU training-time comparison


## Execution Environment and Runtime

All experiments were run on a Google Colab session using a T4 GPU. CPU-based experiments used the CPU available in the respective Colab session for comparison.

### Estimated GPU execution time

~22 minutes

Actual runtime depends on the available hardware, PyTorch version, CUDA configuration, and whether all experiments are executed.

*Developed as coursework for EP08:Pattern Recognition - Machine Learning, course of DIT, UoA.*

# Bias Detection Model

This repository implements a bias detection model using transformers. The model performs both binary classification (biased or non-biased) and token-level classification (biased tokens) for text sequences.

## Folder Structure

- `report.pdf`: The detailed report describing the approach, methodology, and results.
- `src/`: Source code for the project.
  - `train.py`: Main script to train the model.
  - `models/baseline.py`: Baseline model implementation.
  - `models/transformer.py`: Transformer-based model implementation for bias detection.
  - `evaluate.py`: Script to evaluate an existing model.
  - `preprocess.py`: Data preprocessing utilities.
  
## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/bias-detection.git
cd bias-detection
pip install -r requirements.txt
```

Alternatively, you can manually install the required packages:

```bash
pip install torch tiktoken scikit-learn pandas matplotlib
```

## Usage

### Training the Model

To train the model, simply run:

```bash
python src/train.py
```

This will start the training process using the data and model specified in the script.

## Evaluating the Model

To evaluate an existing trained model, run:
```bash
python src/evaluate.py
```

This will load the trained model and provide performance metrics on the test set.
# OCR Perceptron Implementation

This project implements a perceptron for Optical Character Recognition (OCR) using the MNIST dataset.

## Project Structure
- `mnist_perceptron.py`: Main implementation of the perceptron model
- `requirements.txt`: Project dependencies
- `data/`: Directory for MNIST dataset (will be created automatically)

## Setup
1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the main script:
```bash
python mnist_perceptron.py
```

The script will:
1. Download the MNIST dataset
2. Train the perceptron model
3. Evaluate the model on test data
4. Display training progress and final accuracy 
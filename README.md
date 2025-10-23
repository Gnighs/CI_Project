# CI-MAI: Evolutionary Computation Practical Work

## 1. Project Overview

**Chosen Problem:** **Training Neural Networks with Evolutionary Algorithms**

This project explores the use of Evolutionary Algorithms (EAs) as an alternative to traditional derivative-based methods (like Backpropagation) for training Artificial Neural Networks (ANNs).

## 2. Requirements

The project is implemented using python. All necessary libraries can be installed using
```
pip install -r requirements.txt
```

## 3. Execution

To run the program, simply run the following command from the root directory
```
python main.py
```
The output should appear printed in the terminal.

## Project Architecture
CI_Project/
│
├── README.md           # Project description and documentation
├── requirements.txt    # Python dependencies
├── main.py             # Main project code
│
└── src/                # Source code directory
    ├── simple_mlp.py   # Implements the Simple Multi-Layer Perceptron (MLP)
    └── optimizer.py    # Implements the Evolutionary Algorithm
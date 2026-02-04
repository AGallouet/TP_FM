# 2D Flow Matching – Practical Work

This repository contains the practical work for **2D Flow Matching**, focusing on flow-based generative models, conditional flow matching, and neural ODEs implemented in PyTorch.

## Repository Structure

- **`TP_flow_matching.ipynb`**  
  Main notebook for the practical session.  
  It contains:
  - Theoretical reminders on flow matching and conditional flow matching  
  - Implementation of a 2D flow matching model  
  - Training procedure and visualization of learned flows  
  - Sampling by integrating the learned neural ODE

- **`pytorch_1D_example.py`**  
  A minimal **1D PyTorch example** illustrating:
  - How to define and train a neural network in PyTorch  
  - Basic concepts needed before moving to neural ODEs and flow matching  

## Objectives

The goals of this practical work are to:
- Learn how to train neural networks and neural ODEs with PyTorch
- Understand flow-based generative models
- Explore Conditional Flow Matching (CFM)
- Understand the connection between flow matching and optimal transport
- Use optimal transport ideas to improve training and inference

## Requirements

Typical dependencies include:
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Jupyter Notebook

(Exact versions are not enforced but recent versions are recommended.)

## Getting Started

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <repo-name>


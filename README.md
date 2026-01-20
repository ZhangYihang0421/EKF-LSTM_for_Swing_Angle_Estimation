# EKF-LSTM_for_Swing_Angle_Estimation

Implementation of paper: A Residual-Learning-Enhanced EKF for Real-Time Swing Angle Estimation of Quadrotor-Suspended-Payload Systems

This repository contains the training dataset generation tools and model training scripts for the EKF-LSTM based slung load estimation project.

## Directory Structure

- `data/`: Contains the raw flight logs (CSV) used to generate training sequences.
    - `slung_load_dataset.csv`
    - `slung_load_dataset_small.csv`
    - `slung_load_dataset_slowbig.csv`
- `src/`: Contains the source code.
    - `generate_sequence_data.py`: Script to process CSVs into a format suitable for LSTM training.
    - `train_lstm.py`: Script to train the LSTM model.
    - `lstm_model.py`: PyTorch model definition.
    - `SlungLoadEKF.py`: EKF implementation used for generating ground truth/error signals.

## Prerequisites

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Training Data

The training data is generated from the raw CSV flight logs. This process runs the EKF on the logs and computes the error between the EKF estimate and ground truth.

Run the generation script:

```bash
cd src
python generate_sequence_data.py
```

This will create `lstm_correction_data.pth` in the `data/` directory.

### 2. Train the Model

Once the data is generated, you can train the LSTM model:

```bash
cd src
python train_lstm.py
```

This will save the trained model checkpoint to `data/lstm_correction.pth`.

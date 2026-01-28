# EKF-LSTM_for_Swing_Angle_Estimation

Implementation of paper: A Residual-Learning-Enhanced EKF for Real-Time Swing Angle Estimation of Quadrotor-Suspended-Payload Systems

This repository contains the training dataset generation tools and model training scripts for the EKF-LSTM based slung load estimation project.

## Directory Structure

- `data/`: Contains the raw flight logs (CSV) used to generate training sequences.
    - `slung_load_angle_dataset_train.csv`: Merged training dataset.
    - `slung_load_angle_dataset_test.csv`: Test dataset for evaluation.
- `src/`: Contains the source code.
    - `generate_sequence_data.py`: Script to process CSVs into a format suitable for LSTM training.
    - `train_lstm.py`: Script to train the LSTM model.
    - `lstm_model.py`: PyTorch model definition.
    - `SlungLoadEKF.py`: EKF implementation used for generating ground truth/error signals.
- `simulation/`: Contains Gazebo simulation files.
    - `launch/`: ROS launch files.
    - `models/`: Custom Gazebo models (e.g., `iris_slung_load`).
- `scripts/`: Utility scripts.
    - `collect_data.py`: Script to collect training data from the simulation.


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

### 3. Run Simulation & Data Collection

To run the simulation with the custom slung load model:

1.  **Environment Setup**: Ensure PX4 Firmware and Gazebo are properly installed and configured.
2.  **Add Model**: Ensure `simulation/models/iris_slung_load` is in your `GAZEBO_MODEL_PATH`.
3.  **Launch Simulation**:
    ```bash
    roslaunch simulation/launch/mavros_iris_slung_load.launch
    ```
4.  **Collect Data**:
    ```bash
    python scripts/collect_data.py
    ```

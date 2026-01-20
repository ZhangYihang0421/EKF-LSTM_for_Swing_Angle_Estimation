import sys
import os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation as R
from SlungLoadEKF import SlungLoadEKF

def generate_sequence_data():
    # --- Configuration ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # base_dir = os.path.dirname(script_dir) # uav_slung_load
    data_dir = os.path.join(os.path.dirname(script_dir), 'data')
    
    # List of CSV files to process (Matching EKF_ResNet/TCN)
    csv_filenames = [
        'slung_load_dataset.csv',
        'slung_load_dataset_small.csv',
        'slung_load_dataset_slowbig.csv'
    ]
    
    save_path = os.path.join(data_dir, 'lstm_correction_data.pth')
    
    m_drone = 1.5
    m_load = 0.5
    l_rope = 2.0
    MAX_THRUST = 23.5
    
    SEQ_LEN = 400 # 8 second window
    
    raw_inputs = []  # List of [13] vectors
    raw_targets = [] # List of [7] vectors (errors)
    
    # --- Iterate Over CSVs ---
    for csv_file in csv_filenames:
        # Check standard paths (Prioritize EKF_ResNet folder for shared data)
        # Check standard paths (Prioritize EKF_ResNet folder for shared data)
        csv_path = os.path.join(data_dir, csv_file)
        if not os.path.exists(csv_path):
             print(f"Skipping {csv_file} (not found in {data_dir})")
             continue

        print(f"Processing {csv_path}...")
        try:
            df = pd.read_csv(csv_path)
            if len(df) < 10:
                print(f"Skipping {csv_file} (too short)")
                continue
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue

        # --- Pre-process Data ---
        if 'cmd_thrust' in df.columns:
            df['cmd_thrust'] = df['cmd_thrust'].replace(0.0, np.nan)
            df['cmd_thrust'] = df['cmd_thrust'].fillna(method='bfill')
            df['cmd_thrust'] = df['cmd_thrust'].fillna(0.6)
    
        # --- Initialize EKF ---
        ekf = SlungLoadEKF(m_drone=m_drone, m_load=m_load, l_rope=l_rope)
        
        file_inputs = []
        file_targets = []
        
        print("Running EKF and collecting stream...")
        
        last_t = df.iloc[0]['timestamp']
        
        for i in range(len(df)):
            row = df.iloc[i]
            t = row['timestamp']
            if i == 0: dt = 0.02
            else: dt = t - last_t
            last_t = t
            if dt > 0.1: dt = 0.05
            if dt < 0.001: continue 
    
            # Extract Data
            q = [row['drone_qx'], row['drone_qy'], row['drone_qz'], row['drone_qw']]
            r = R.from_quat(q)
            rot_matrix = r.as_matrix()
            
            thrust_mag = row['cmd_thrust'] * MAX_THRUST
            if thrust_mag < 0.1: thrust_mag = (m_drone + m_load) * 9.81
            thrust_body = np.array([0, 0, thrust_mag])
            u_enu = rot_matrix @ thrust_body
            
            acc_body = np.array([row['imu_ax'], row['imu_ay'], row['imu_az']])
            acc_enu = rot_matrix @ acc_body
            
            u_ned = np.array([u_enu[1], u_enu[0], -u_enu[2]])
            acc_ned_raw = np.array([acc_enu[1], acc_enu[0], -acc_enu[2]])
            z_ned = acc_ned_raw + np.array([0, 0, 9.81])
            
            # EKF Step
            ekf.set_current_control(u_ned) 
            ekf.predict(u=u_ned, dt=dt)
            ekf.update(z_meas=z_ned)
            
            x_est = ekf.x.copy()
            
            # Ground Truth
            gt_xi = -row['gt_xi']
            gt_zeta = row['gt_zeta']
            gt_xi_dot = row['gt_xi_dot']
            gt_zeta_dot = row['gt_zeta_dot']
            
            x_true = np.zeros(7)
            x_true[0] = gt_xi
            x_true[1] = gt_zeta
            x_true[2] = gt_xi_dot
            x_true[3] = gt_zeta_dot
            
            error = x_true - x_est
            
            # Check divergence
            if np.abs(x_est[0]) > 2.0 or np.abs(x_est[1]) > 2.0:
                 # Diverged, reset and burn in
                 ekf = SlungLoadEKF(m_drone=m_drone, m_load=m_load, l_rope=l_rope)
                 # Should probably drop recent history or mark break
                 # simple strategy: append NaN to break sequence
                 file_inputs.append(np.full(13, np.nan))
                 file_targets.append(np.full(7, np.nan))
                 continue
    
            feat = np.concatenate([x_est, u_ned, z_ned])
            file_inputs.append(feat)
            file_targets.append(error)

        print(f"  -> Collected {len(file_inputs)} samples from {csv_file}")
            
        # Add NaNs to separate files
        file_inputs.append(np.full(13, np.nan))
        file_targets.append(np.full(7, np.nan))
        
        raw_inputs.extend(file_inputs)
        raw_targets.extend(file_targets)

    # --- Create Sequences ---
    print("Creating sequences...")
    all_inputs = np.array(raw_inputs)
    all_targets = np.array(raw_targets)
    
    # Compute stats on valid data
    valid_mask = ~np.isnan(all_inputs).any(axis=1)
    valid_inputs = all_inputs[valid_mask]
    
    mean = np.mean(valid_inputs, axis=0)
    std = np.std(valid_inputs, axis=0) + 1e-6
    
    # Normalize (keep NaNs as NaNs for splitting)
    all_inputs_norm = np.copy(all_inputs)
    all_inputs_norm[valid_mask] = (valid_inputs - mean) / std
    
    seq_x = []
    seq_y = []
    
    # Sliding window
    for i in range(len(all_inputs) - SEQ_LEN):
        window_in = all_inputs_norm[i : i+SEQ_LEN]
        window_out = all_targets[i : i+SEQ_LEN]
        
        # Check if window contains any NaN (break)
        if np.isnan(window_in).any():
            continue
            
        seq_x.append(window_in)
        seq_y.append(window_out)
        
    seq_x = np.array(seq_x)
    seq_y = np.array(seq_y)
    
    print(f"Generated {len(seq_x)} sequences of length {SEQ_LEN}")
    
    # --- Save ---
    # Convert to tensor
    inputs_tensor = torch.tensor(seq_x, dtype=torch.float32)
    targets_tensor = torch.tensor(seq_y, dtype=torch.float32)
    
    print(f"Saving to {save_path}")
    torch.save({
        'inputs': inputs_tensor, 
        'targets': targets_tensor,
        'mean': mean,
        'std': std
    }, save_path)

if __name__ == "__main__":
    generate_sequence_data()

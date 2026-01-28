import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from lstm_model import LSTMCorrectionModel
import os

def train_lstm():
    # --- Config ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), 'data')
    data_path = os.path.join(data_dir, 'lstm_correction_data.pth')
    model_save_path = os.path.join(data_dir, 'lstm_correction.pth')
    BATCH_SIZE = 2048
    EPOCHS = 50
    LR = 1e-3
    
    # --- Load Data ---
    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found! Run generate_sequence_data.py first.")
        return

    print(f"Loading data from {data_path}...")
    data = torch.load(data_path, weights_only=False)
    inputs = data['inputs']
    targets = data['targets']
    
    # inputs: (N, Seq, 13)
    # targets: (N, Seq, 7)
    
    dataset = TensorDataset(inputs, targets)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # --- Calc Baseline ---
    # What is the loss if we just predict 0? (i.e. Variance of the targets)
    # targets are already normalized? No, targets are raw errors.
    # Actually wait, let's check generate_sequence_data.py.
    # Targets are just raw error. "inputs" are normalized.
    # So we want variance of targets.
    
    all_targets = targets.view(-1, 7) # Flatten
    var_targets = torch.var(all_targets[:, :4], dim=0) # Variance of first 4 states
    mean_var = torch.mean(var_targets).item()
    print(f"Baseline MSE (Predict 0): {mean_var:.6f}") # If Loss ~ this, model is learning nothing.
    
    # --- Model ---
    model = LSTMCorrectionModel(input_dim=13, state_dim=7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    
    print(f"Start Training on {device}...")
    
    # --- Loop ---
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            pred_y, _ = model(batch_x)
            
            # Loss on just the angles? Or all?
            # Let's train on all first 4 states (Xi, Zeta, Xi_dot, Zeta_dot)
            # pred_y is (Batch, Seq, 7)
            # We care about every timestep in the sequence
            
            loss = loss_fn(pred_y[:, :, :4], batch_y[:, :, :4])
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # Val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred_y, _ = model(batch_x)
                loss = loss_fn(pred_y[:, :, :4], batch_y[:, :, :4])
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Train Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            
    # --- Save ---
    # Need to save stats too for inference
    data = torch.load(data_path, weights_only=False)
    mean = data['mean']
    std = data['std']
    
    full_checkpoint = {
        'model_state_dict': model.state_dict(),
        'mean': mean,
        'std': std
    }
    
    torch.save(full_checkpoint, model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train_lstm()

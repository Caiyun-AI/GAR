import ast
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import pandas as pd
import numpy as np
import argparse
import random
from tqdm import tqdm
from sklearn.model_selection import KFold  

class Module(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        super(Module, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_inputs, num_hiddens),
            nn.ReLU(),
    #         nn.Linear(num_hiddens, 2 * num_hiddens),
    #         nn.ReLU(),
            nn.Linear(num_hiddens, num_outputs)
        )

    def forward(self, x):
        x = self.model(x)
        return x
    
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train(model, train_loader, loss_fn, optimizer, device):  
    model.train()  
    total_train_loss = 0  
    total_train_acc = 0  
    for data, label in train_loader:  
        data, label = data.to(device), label.to(device)  
        optimizer.zero_grad()  
        output = model(data)  
        loss = loss_fn(output, label)  
        loss.backward()  
        optimizer.step()  
        predict_label = torch.argmax(output, dim=1)  
        acc = (predict_label == label).sum().item()  
        total_train_acc += acc  
        total_train_loss += loss.item() 

    avg_loss = total_train_loss / len(train_loader.dataset)
    avg_acc = total_train_acc / len(train_loader.dataset) 
    return avg_loss, avg_acc 

def evaluate(model, data_loader, loss_fn, device):  
    model.eval()  
    total_loss = 0  
    total_acc = 0  
    y_pred, y_true = [], []  
    with torch.no_grad():  
        for data, label in data_loader:  
            data, label = data.to(device), label.to(device)  
            output = model(data)  
            loss = loss_fn(output, label)  
            total_loss += loss.item()  
            predict_label = torch.argmax(output, dim=1)  
            y_pred.extend(predict_label.cpu())  
            y_true.extend(label.cpu())  
            acc = (predict_label == label).sum()  
            total_acc += acc  
    avg_loss = total_loss / len(data_loader.dataset)
    avg_acc = total_acc / len(data_loader.dataset)
    return avg_loss, avg_acc.cpu(), y_pred, y_true   

def create_model_and_optimizer(num_inputs, num_hiddens, num_outputs, device=0, lr=1e-3):  
    model = Module(num_inputs, num_hiddens, num_outputs).to(device)  
    loss_fn = torch.nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  
    return model, loss_fn, optimizer  

def get_SNLI_dataset(lhs):
    train_data = pd.read_csv('./MLP/data/SNLI_zero_shot/snli_YesNo_head_feature_3000_train_zero_shot.csv')  
    val_data = pd.read_csv('./MLP/data/SNLI_zero_shot/snli_YesNo_head_feature_3000_validation_zero_shot.csv')  
    test_data = pd.read_csv('./MLP/data/SNLI_zero_shot/snli_YesNo_head_feature_3000_test_zero_shot.csv')  

    selected_columns = lhs+['label']  
     
    train_data = train_data[selected_columns]
    val_data = val_data[selected_columns]
    test_data = test_data[selected_columns]  

    X_train = torch.Tensor(train_data.iloc[:, :-1].values)  
    y_train = torch.Tensor(train_data.iloc[:, -1].values).long()  
    X_val = torch.Tensor(val_data.iloc[:, :-1].values)  
    y_val = torch.Tensor(val_data.iloc[:, -1].values).long()  
    X_test = torch.Tensor(test_data.iloc[:, :-1].values)  
    y_test = torch.Tensor(test_data.iloc[:, -1].values).long() 
    train_dataset = Data.TensorDataset(X_train, y_train)  
    val_dataset = Data.TensorDataset(X_val, y_val)  
    test_dataset = Data.TensorDataset(X_test, y_test)  
    
    train_loader = Data.DataLoader(train_dataset, batch_size=32, shuffle=True)  
    val_loader = Data.DataLoader(val_dataset, batch_size=32)  
    test_loader = Data.DataLoader(test_dataset, batch_size=32)
    return train_loader, val_loader, test_loader

def get_GoT_dataset(dataset, lhs):
    selected_columns = lhs + ['label']
    dataset = dataset[selected_columns]
    X = torch.Tensor(dataset.iloc[:, :-1].values)
    y = torch.Tensor(dataset.iloc[:, -1].values).long()
    return X, y

def prepare_loaders(X, y, train_idx, val_idx, test_idx, batch_size=32):
    train_dataset = Data.TensorDataset(X[train_idx], y[train_idx])
    val_dataset = Data.TensorDataset(X[val_idx], y[val_idx])
    test_dataset = Data.TensorDataset(X[test_idx], y[test_idx])
    
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = Data.DataLoader(val_dataset, batch_size=batch_size)
    test_loader = Data.DataLoader(test_dataset, batch_size=batch_size)   
    return train_loader, val_loader, test_loader

def SNLI_5_rounds(best_model_path, num_inputs, num_hiddens, num_outputs, device, epochs, lr, lhs):
    train_loader, val_loader, test_loader = get_SNLI_dataset(lhs) 
    fold_results = []
    for fold in range(5):
        model, loss_fn, optimizer = create_model_and_optimizer(num_inputs, num_hiddens, num_outputs, device, lr) 
        best_val_acc = 0  
        for epoch in tqdm(range(epochs), desc=f"Fold {fold + 1} Training"):  
            train_loss, train_acc = train(model, train_loader, loss_fn, optimizer, device)  
            val_loss, val_acc, _, _ = evaluate(model, val_loader, loss_fn, device)  
                        
            if val_acc > best_val_acc:  
                best_val_acc = val_acc  
                torch.save(model.state_dict(), best_model_path)
                
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, ")
  
        model.load_state_dict(torch.load(best_model_path))
        _, test_acc, _, _ = evaluate(model, test_loader, loss_fn, device)  
        fold_results.append(test_acc)  
        print(f"Test accuracy for fold {fold}: {test_acc:.4f}") 
        print("-" * 50)  
  
    print(f"Average test accuracy across all folds: {np.mean(fold_results):.4f}")     

def GoT_5_Fold(dataname, best_model_path, num_inputs, num_hiddens, num_outputs, device, epochs, lr, lhs):
    fold_results = []
    dataset = pd.read_csv(f'./MLP/data/GoT_zero_shot/{dataname}.csv')
    X, y = get_GoT_dataset(dataset, lhs)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        random.shuffle(train_idx)
        split_point = int(len(train_idx) * 0.75)  # 8:2 -> 6:2:2
        train_idx, val_idx = train_idx[:split_point], train_idx[split_point:]
        
        train_loader, val_loader, test_loader = prepare_loaders(X, y, train_idx, val_idx, test_idx)
        model, loss_fn, optimizer = create_model_and_optimizer(num_inputs, num_hiddens, num_outputs, device, lr)
        
        best_val_acc = 0  
        for epoch in tqdm(range(epochs)):  
            train_loss, train_acc = train(model, train_loader, loss_fn, optimizer, device)  
            val_loss, val_acc, _, _ = evaluate(model, val_loader, loss_fn, device)   
            if val_acc > best_val_acc:  
                best_val_acc = val_acc  
                torch.save(model.state_dict(), best_model_path)
                
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, ")
        
        model.load_state_dict(torch.load(best_model_path))
        _, test_acc, _, _ = evaluate(model, test_loader, loss_fn, device)  
        fold_results.append(test_acc)  
        print(f"Test accuracy for fold {fold}: {test_acc:.4f}") 
        print("-" * 50)  

    print(f"Average test accuracy across all folds: {np.mean(fold_results):.4f}")  

def main(dataname, lhs, epochs=30, lr=1e-3, device=0): 
    num_inputs, num_hiddens, num_outputs = len(lhs),32,2
    device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
    best_model_path = f"./best_model_lr_{lr}_epochs_{epochs}.pth" 
    if dataname=='SNLI':
        SNLI_5_rounds(best_model_path, num_inputs, num_hiddens, num_outputs, device, epochs, lr, lhs)
    else:
        GoT_5_Fold(dataname, best_model_path, num_inputs, num_hiddens, num_outputs, device, epochs, lr, lhs)
    
if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Parameters")
    parser.add_argument("--dataname", type=str, required=True, help="You can choose either SNLI or a subset of GoT (e.g., companies_true_false, counterfact_true_false, cities, sp_en_trans)")
    parser.add_argument("--epochs", type=int, required=False, default=30, help="Number of epochs to train (e.g., 30)")
    parser.add_argument("--lr", type=float, required=False, default=1e-3, help="Initial learning rate (e.g., 0.001)")
    parser.add_argument("--device", type=int, required=False, default=0, help="Device to use (e.g., 0 for GPU0, 1 for GPU1, etc.)")
    parser.add_argument("--lhs", type=str, required=False, default='[(15,51), (14,0), (14,18), (14,46)]', help='Which (Layer, Head) attention weights do you want to use? (e.g., ["(15,51)", "(14,0)", "(14,18)", "(14,46)"])')
    
    args = parser.parse_args()
    set_seed(42)
    lhs = ast.literal_eval(args.lhs)
    lhs = [str(i).replace(' ','') for i in lhs]
    main(dataname=args.dataname, lhs=lhs, epochs=args.epochs, lr=args.lr, device=args.device)

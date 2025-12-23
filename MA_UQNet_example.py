# MA-UQNet: Multi-modal Attention Uncertainty Quantification Network
# Example implementation - Using synthetic data for demonstration
# 
# Required packages:
# torch>=1.9.0, numpy>=1.19.0, pandas>=1.1.0, scikit-learn>=0.24.0, tqdm>=4.50.0

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
import os
import pickle

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

GROWTH_STAGES = {
    0: "Germination",
    1: "Seedling growth",
    2: "Tillering",
    3: "Stem elongation",
    4: "Booting",
    5: "Inflorescence emergence",
    6: "Anthesis",
    7: "Milk development",
    8: "Dough development",
    9: "Ripening"
}

SELECTED_STAGES = [3, 4, 6, 7]

N_SAMPLES = 5000
N_SPECTRAL = 12
N_ENV = 5
N_STAGES = 10

HIDDEN_DIM = 32
FUSION_DIM = 64
DROPOUT_RATE = 0.3

BATCH_SIZE = 32
EPOCHS = 300
LR = 0.001
WEIGHT_DECAY = 1e-5
PATIENCE = 30
NUM_MC_SAMPLES = 30

def generate_synthetic_data():
    """Generate synthetic data for demonstration with selected growth stages"""
    stages = np.random.choice(SELECTED_STAGES, N_SAMPLES)
    years = np.random.choice([2018, 2019, 2020, 2021], N_SAMPLES)
    
    X_spectral = np.zeros((N_SAMPLES, N_SPECTRAL))
    X_env = np.zeros((N_SAMPLES, N_ENV))
    y = np.zeros(N_SAMPLES)
    

    stage_spectral_means = {
        3: np.array([520, 540, 600, 580, 450, 600, 670, 1000, 850, 2050, 490, 930]),
        4: np.array([540, 560, 620, 600, 470, 620, 690, 1050, 880, 2100, 510, 960]),
        6: np.array([580, 600, 660, 640, 510, 660, 730, 1150, 940, 2200, 550, 1020]),
        7: np.array([580, 590, 650, 630, 500, 650, 720, 1120, 920, 2150, 540, 1000])
    }
    

    stage_env_means = {
        3: np.array([65, 62, 480, 620, 50]),
        4: np.array([70, 70, 550, 720, 50]),
        6: np.array([80, 85, 680, 880, 50]),
        7: np.array([82, 80, 700, 900, 50])
    }
    

    stage_biomass_base = {
        3: 2500,
        4: 3200,
        6: 4200,
        7: 4500
    }
    
    for i in range(N_SAMPLES):
        stage = stages[i]
        X_spectral[i] = stage_spectral_means[stage] + np.random.randn(N_SPECTRAL) * 80
        X_env[i] = stage_env_means[stage] + np.random.randn(N_ENV) * 8
        
        spectral_effect = X_spectral[i, [0, 4, 5, 10]].mean() * 0.3
        env_effect = X_env[i].sum() * 0.15
        stage_effect = stage_biomass_base[stage]
        noise = np.random.randn() * 150
        
        y[i] = stage_effect + spectral_effect + env_effect + noise
    
    return X_spectral, X_env, y, stages, years


def preprocess_data():
    """Data preprocessing and splitting"""
    X_spectral, X_env, y, stages, years = generate_synthetic_data()
    
    print(f"\nTotal samples generated: {N_SAMPLES}")
    print(f"\nGrowth stage distribution (Zadoks Z-Scale Primary Stages):")
    for stage_id in SELECTED_STAGES:
        count = np.sum(stages == stage_id)
        stage_mask = stages == stage_id
        biomass_mean = np.mean(y[stage_mask])
        biomass_std = np.std(y[stage_mask])
        stage_name = GROWTH_STAGES[stage_id]
        print(f"  Z{stage_id} - {stage_name}: {count} samples ({count/N_SAMPLES*100:.1f}%) - "
              f"Biomass: {biomass_mean:.0f}±{biomass_std:.0f} kg/ha")
    
    test_years = [2020, 2021]
    train_mask = ~np.isin(years, test_years)
    test_mask = np.isin(years, test_years)
    
    X_train_spectral = X_spectral[train_mask]
    X_train_env = X_env[train_mask]
    y_train = y[train_mask]
    stages_train = stages[train_mask]
    
    X_test_spectral = X_spectral[test_mask]
    X_test_env = X_env[test_mask]
    y_test = y[test_mask]
    stages_test = stages[test_mask]
    
    spectral_scaler = StandardScaler().fit(X_train_spectral)
    env_scaler = StandardScaler().fit(X_train_env)
    
    X_train_spectral_scaled = spectral_scaler.transform(X_train_spectral)
    X_train_env_scaled = env_scaler.transform(X_train_env)
    X_test_spectral_scaled = spectral_scaler.transform(X_test_spectral)
    X_test_env_scaled = env_scaler.transform(X_test_env)
    
    X_train_spectral_scaled, X_val_spectral_scaled, X_train_env_scaled, X_val_env_scaled, \
    y_train_split, y_val, stages_train_split, stages_val = train_test_split(
        X_train_spectral_scaled, X_train_env_scaled, y_train, stages_train,
        test_size=0.2, random_state=RANDOM_SEED
    )
    
    return {
        'X_train_spectral': X_train_spectral_scaled,
        'X_train_env': X_train_env_scaled,
        'y_train': y_train_split,
        'stages_train': stages_train_split,
        'X_val_spectral': X_val_spectral_scaled,
        'X_val_env': X_val_env_scaled,
        'y_val': y_val,
        'stages_val': stages_val,
        'X_test_spectral': X_test_spectral_scaled,
        'X_test_env': X_test_env_scaled,
        'y_test': y_test,
        'stages_test': stages_test
    }


class WheatBiomassDataset(Dataset):
    def __init__(self, X_spectral, X_env, y, stages):
        self.X_spectral = torch.FloatTensor(X_spectral)
        self.X_env = torch.FloatTensor(X_env)
        self.y = torch.FloatTensor(y).view(-1, 1)
        self.stages = torch.LongTensor(stages).view(-1, 1)
        self.growth_days = self.X_env[:, 0:1]
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X_spectral[idx], self.X_env[idx], self.stages[idx], \
               self.growth_days[idx], self.y[idx]


def create_data_loaders(processed_data):
    train_dataset = WheatBiomassDataset(
        processed_data['X_train_spectral'], 
        processed_data['X_train_env'], 
        processed_data['y_train'], 
        processed_data['stages_train']
    )
    
    val_dataset = WheatBiomassDataset(
        processed_data['X_val_spectral'], 
        processed_data['X_val_env'], 
        processed_data['y_val'], 
        processed_data['stages_val']
    )
    
    test_dataset = WheatBiomassDataset(
        processed_data['X_test_spectral'], 
        processed_data['X_test_env'], 
        processed_data['y_test'], 
        processed_data['stages_test']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=False)
    
    return train_loader, val_loader, test_loader


class SpectralAttentionModule(nn.Module):
    def __init__(self, spectral_dim=12, hidden_dim=32, dropout_rate=0.3):
        super().__init__()
        self.band_interaction = nn.Linear(spectral_dim, spectral_dim)
        
        self.attention_generator = nn.Sequential(
            nn.Linear(spectral_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, spectral_dim),
            nn.Sigmoid()
        )
        
        self.feature_enhancer = nn.Sequential(
            nn.Linear(spectral_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, x, apply_dropout=False):
        band_relations = self.band_interaction(x)
        
        if apply_dropout:
            dropout_layers = []
            for module in self.attention_generator.modules():
                if isinstance(module, nn.Dropout):
                    dropout_layers.append((module, module.training))
                    module.train()
            for module in self.feature_enhancer.modules():
                if isinstance(module, nn.Dropout):
                    dropout_layers.append((module, module.training))
                    module.train()
            
            attention = self.attention_generator(band_relations)
            weighted_features = x * attention
            enhanced_features = self.feature_enhancer(weighted_features)
            
            for module, original_mode in dropout_layers:
                module.train(original_mode)
        else:
            attention = self.attention_generator(band_relations)
            weighted_features = x * attention
            enhanced_features = self.feature_enhancer(weighted_features)
        
        return enhanced_features, attention


class EnvironmentalAttentionModule(nn.Module):
    def __init__(self, env_dim=5, hidden_dim=32, dropout_rate=0.3):
        super().__init__()
        self.attention_generator = nn.Sequential(
            nn.Linear(env_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim//2, env_dim),
            nn.Sigmoid()
        )
        
        self.feature_enhancer = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x, apply_dropout=False):
        if apply_dropout:
            dropout_layers = []
            for module in self.attention_generator.modules():
                if isinstance(module, nn.Dropout):
                    dropout_layers.append((module, module.training))
                    module.train()
            for module in self.feature_enhancer.modules():
                if isinstance(module, nn.Dropout):
                    dropout_layers.append((module, module.training))
                    module.train()
            
            attention = self.attention_generator(x)
            weighted_features = x * attention
            enhanced_features = self.feature_enhancer(weighted_features)
            
            for module, original_mode in dropout_layers:
                module.train(original_mode)
        else:
            attention = self.attention_generator(x)
            weighted_features = x * attention
            enhanced_features = self.feature_enhancer(weighted_features)
        
        return enhanced_features, attention


class GrowthStageSpecificModule(nn.Module):
    def __init__(self, embedding_dim=32):
        super().__init__()


        self.stage_embedding = nn.Embedding(10, embedding_dim)
    
    def forward(self, stages):
        """
        Forward pass for growth stage-specific processing module.
        
        Args:
            stages: Zadoks stage values (0-9, e.g., 3, 4, 5, 6)
        
        Returns:
            stage_embedding: Embedded representation of stages
            one_hot_stages: One-hot encoded stage representation
        """
        batch_size = stages.size(0)
        stages_flat = stages.view(-1).long()
        

        stage_embedding = self.stage_embedding(stages_flat)
        

        one_hot_stages = torch.zeros(batch_size, 10, device=stages.device)
        one_hot_stages.scatter_(1, stages_flat.view(-1, 1), 1)
        
        return stage_embedding, one_hot_stages


class BilinearAttentionFusion(nn.Module):
    def __init__(self, spectral_dim=32, env_dim=32, stage_dim=32, 
                 fusion_dim=64, dropout_rate=0.3):
        super().__init__()
        self.spectral_env_attn = nn.Bilinear(spectral_dim, env_dim, fusion_dim//2)
        self.stage_modulation = nn.Linear(stage_dim, fusion_dim//2)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, fusion_dim)
        )
    
    def forward(self, spectral_features, env_features, stage_embedding, apply_dropout=False):
        cross_modal_features = self.spectral_env_attn(spectral_features, env_features)
        stage_features = self.stage_modulation(stage_embedding)
        fused = torch.cat([cross_modal_features, stage_features], dim=1)
        
        if apply_dropout:
            dropout_layers = []
            for module in self.fusion_layer.modules():
                if isinstance(module, nn.Dropout):
                    dropout_layers.append((module, module.training))
                    module.train()
            
            fused = self.fusion_layer(fused)
            
            for module, original_mode in dropout_layers:
                module.train(original_mode)
        else:
            fused = self.fusion_layer(fused)
        
        return fused


class UncertaintyOutputModule(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.mean_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.var_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
    
    def forward(self, x):
        mean = self.mean_head(x)
        var = self.var_head(x)
        return mean, var


class MAUQNet(nn.Module):
    def __init__(self, spectral_dim, env_dim, hidden_dim, fusion_dim, dropout_rate):
        super().__init__()
        self.spectral_module = SpectralAttentionModule(spectral_dim, hidden_dim, dropout_rate)
        self.env_module = EnvironmentalAttentionModule(env_dim, hidden_dim, dropout_rate)

        self.growth_stage_module = GrowthStageSpecificModule(embedding_dim=hidden_dim)
        
        self.bilinear_fusion = BilinearAttentionFusion(
            spectral_dim=hidden_dim, 
            env_dim=hidden_dim, 
            stage_dim=hidden_dim,
            fusion_dim=fusion_dim, 
            dropout_rate=dropout_rate
        )
        

        self.stage_specific_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for _ in range(10)
        ])
        
        self.output_module = UncertaintyOutputModule(input_dim=fusion_dim, hidden_dim=hidden_dim)
        
        self.dropout_rate = dropout_rate
    
    def forward(self, spectral, env, stages, growth_days=None, apply_dropout=False):
        spectral_features, spectral_attention = self.spectral_module(spectral, apply_dropout=apply_dropout)
        env_features, env_attention = self.env_module(env, apply_dropout=apply_dropout)
        stage_embedding, stage_weights = self.growth_stage_module(stages)
        
        fused_base = self.bilinear_fusion(
            spectral_features, env_features, stage_embedding, apply_dropout=apply_dropout
        )
        
        batch_size = spectral.size(0)
        stage_outputs = []
        
        for i in range(10):
            if apply_dropout:
                dropout_layers = []
                for module in self.stage_specific_layers[i].modules():
                    if isinstance(module, nn.Dropout):
                        dropout_layers.append((module, module.training))
                        module.train()
                
                stage_out = self.stage_specific_layers[i](fused_base)
                
                for module, original_mode in dropout_layers:
                    module.train(original_mode)
            else:
                stage_out = self.stage_specific_layers[i](fused_base)
            
            stage_outputs.append(stage_out)
        
        stage_outputs = torch.stack(stage_outputs, dim=1)
        fused_features = (stage_weights.unsqueeze(2) * stage_outputs).sum(dim=1)
        
        mean, var = self.output_module(fused_features)
        
        return mean, var, {
            'spectral_attention': spectral_attention,
            'env_attention': env_attention,
            'stage_weights': stage_weights
        }
    
    def predict_with_uncertainty(self, spectral, env, stages, growth_days=None, num_samples=30):
        """Predict with Monte Carlo dropout for uncertainty estimation"""
        original_training_state = self.training
        means = []
        vars = []
        spectral_attentions = []
        env_attentions = []
        stage_weights_list = []
        
        for i in range(num_samples):
            self.eval()
            with torch.no_grad():
                mean, var, attention_dict = self.forward(
                    spectral, env, stages, growth_days, apply_dropout=True
                )
                means.append(mean)
                vars.append(var)
                spectral_attentions.append(attention_dict['spectral_attention'])
                env_attentions.append(attention_dict['env_attention'])
                stage_weights_list.append(attention_dict['stage_weights'])
        
        self.train(original_training_state)
        
        means = torch.stack(means, dim=0)
        vars = torch.stack(vars, dim=0)
        
        pred_mean = means.mean(dim=0)
        epistemic_variance = means.var(dim=0, unbiased=False)
        epistemic_uncertainty = torch.sqrt(epistemic_variance + 1e-8)
        
        aleatoric_variance = vars.mean(dim=0)
        aleatoric_uncertainty = torch.sqrt(aleatoric_variance + 1e-8)
        
        total_variance = epistemic_variance + aleatoric_variance
        total_uncertainty = torch.sqrt(total_variance + 1e-8)
        
        spectral_attention = torch.stack(spectral_attentions, dim=0).mean(dim=0)
        env_attention = torch.stack(env_attentions, dim=0).mean(dim=0)
        stage_weights = torch.stack(stage_weights_list, dim=0).mean(dim=0)
        
        return pred_mean, total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty, {
            'spectral_attention': spectral_attention,
            'env_attention': env_attention,
            'stage_weights': stage_weights
        }


def negative_log_likelihood_loss(mean, var, target):
    """Negative log-likelihood loss for heteroscedastic regression"""
    var = torch.clamp(var, min=1e-6, max=1e6)
    nll = 0.5 * torch.log(var) + 0.5 * torch.pow(mean - target, 2) / var
    return torch.mean(nll)


def train_model(model, train_loader, val_loader):
    """Train the model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    best_val_loss = float('inf')
    no_improve_epochs = 0
    best_model_state = None
    
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_rmse': [], 'val_rmse': [],
        'train_r2': [], 'val_r2': [], 
        'learning_rate': []
    }
    
    pbar = tqdm(range(EPOCHS), desc="Training")
    
    for epoch in pbar:
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rate'].append(current_lr)
        
        model.train()
        train_loss = 0.0
        train_rmse = 0.0
        all_train_targets = []
        all_train_preds = []
        
        for spectral, env, stages, growth_days, target in train_loader:
            spectral, env = spectral.to(device), env.to(device)
            stages = stages.to(device)
            growth_days, target = growth_days.to(device), target.to(device)
            
            mean, var, _ = model(spectral, env, stages, growth_days)
            loss = negative_log_likelihood_loss(mean, var, target)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * spectral.size(0)
            train_rmse += torch.sqrt(torch.mean((mean - target)**2)).item() * spectral.size(0)
            all_train_preds.append(mean.detach().cpu().numpy())
            all_train_targets.append(target.detach().cpu().numpy())
        
        all_train_preds = np.concatenate(all_train_preds)
        all_train_targets = np.concatenate(all_train_targets)
        train_r2 = r2_score(all_train_targets, all_train_preds)
        train_loss /= len(train_loader.dataset)
        train_rmse /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        val_rmse = 0.0
        all_val_targets = []
        all_val_preds = []
        
        with torch.no_grad():
            for spectral, env, stages, growth_days, target in val_loader:
                spectral, env = spectral.to(device), env.to(device)
                stages = stages.to(device)
                growth_days, target = growth_days.to(device), target.to(device)
                
                mean, var, _ = model(spectral, env, stages, growth_days)
                loss = negative_log_likelihood_loss(mean, var, target)
                
                val_loss += loss.item() * spectral.size(0)
                val_rmse += torch.sqrt(torch.mean((mean - target)**2)).item() * spectral.size(0)
                all_val_preds.append(mean.detach().cpu().numpy())
                all_val_targets.append(target.detach().cpu().numpy())
        
        all_val_preds = np.concatenate(all_val_preds)
        all_val_targets = np.concatenate(all_val_targets)
        val_r2 = r2_score(all_val_targets, all_val_preds)
        val_loss /= len(val_loader.dataset)
        val_rmse /= len(val_loader.dataset)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_rmse'].append(train_rmse)
        history['val_rmse'].append(val_rmse)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)
        
        pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_r2': f'{val_r2:.4f}',
            'lr': f'{current_lr:.6f}'
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        
        if no_improve_epochs >= PATIENCE:
            print(f'\nEarly stopping at epoch {epoch+1}')
            print(f'Best validation loss: {best_val_loss:.4f}')
            break
    
    model.load_state_dict(best_model_state)
    return model, history


def evaluate_model(model, data_loader):
    """Evaluate model performance"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_preds = []
    all_targets = []
    all_total_uncs = []
    all_epistemic_uncs = []
    all_aleatoric_uncs = []
    
    with torch.no_grad():
        for spectral, env, stages, growth_days, target in data_loader:
            spectral, env = spectral.to(device), env.to(device)
            stages = stages.to(device)
            growth_days = growth_days.to(device)
            
            pred_mean, total_unc, epistemic_unc, aleatoric_unc, _ = model.predict_with_uncertainty(
                spectral, env, stages, growth_days, num_samples=NUM_MC_SAMPLES
            )
            
            all_preds.append(pred_mean.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_total_uncs.append(total_unc.cpu().numpy())
            all_epistemic_uncs.append(epistemic_unc.cpu().numpy())
            all_aleatoric_uncs.append(aleatoric_unc.cpu().numpy())
    
    predictions = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    total_uncs = np.concatenate(all_total_uncs)
    epistemic_uncs = np.concatenate(all_epistemic_uncs)
    aleatoric_uncs = np.concatenate(all_aleatoric_uncs)
    
    r2 = r2_score(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    
    lower_bound = predictions - 1.96 * total_uncs
    upper_bound = predictions + 1.96 * total_uncs
    coverage = np.mean((targets >= lower_bound) & (targets <= upper_bound)) * 100
    
    mean_epistemic = np.mean(epistemic_uncs)
    mean_aleatoric = np.mean(aleatoric_uncs)
    epistemic_var = mean_epistemic ** 2
    aleatoric_var = mean_aleatoric ** 2
    total_var = epistemic_var + aleatoric_var
    epistemic_pct = (epistemic_var / total_var) * 100 if total_var > 0 else 50.0
    aleatoric_pct = (aleatoric_var / total_var) * 100 if total_var > 0 else 50.0
    
    return {
        'r2': r2,
        'rmse': rmse,
        'coverage': coverage,
        'epistemic_pct': epistemic_pct,
        'aleatoric_pct': aleatoric_pct,
        'predictions': predictions,
        'targets': targets
    }






def main():
    print("="*80)
    print("MA-UQNet: Multi-modal Attention Uncertainty Quantification Network")
    print("Example Implementation - Using Synthetic Data with Zadoks Z-Scale")
    print("="*80)
    
    print("\n" + "="*80)
    print("Zadoks Z-Scale Primary Growth Stages (0-9):")
    print("="*80)
    for stage_id, stage_name in GROWTH_STAGES.items():
        if stage_id in SELECTED_STAGES:
            print(f"  Z{stage_id}: {stage_name} *SELECTED*")
        else:
            print(f"  Z{stage_id}: {stage_name}")
    print("="*80)
    
    print("\nGenerating synthetic data...")
    processed_data = preprocess_data()
    
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(processed_data)
    
    print(f"\nDataset sizes:")
    print(f"  Training: {len(processed_data['y_train'])}")
    print(f"  Validation: {len(processed_data['y_val'])}")
    print(f"  Test: {len(processed_data['y_test'])}")
    
    print("\nInitializing MA-UQNet model...")
    model = MAUQNet(
        spectral_dim=N_SPECTRAL,
        env_dim=N_ENV,
        hidden_dim=HIDDEN_DIM,
        fusion_dim=FUSION_DIM,
        dropout_rate=DROPOUT_RATE
    )
    
    print("\nTraining MA-UQNet...")
    trained_model, history = train_model(model, train_loader, val_loader)
    
    print("\nEvaluating on test set...")
    test_results = evaluate_model(trained_model, test_loader)
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    print(f"\nPrediction Accuracy:")
    print(f"  R²: {test_results['r2']:.4f}")
    print(f"  RMSE: {test_results['rmse']:.2f} kg/ha")
    
    print(f"\nUncertainty Quantification:")
    print(f"  95% Coverage: {test_results['coverage']:.1f}%")
    
    print(f"\nUncertainty Decomposition:")
    print(f"  Epistemic Uncertainty: {test_results['epistemic_pct']:.1f}%")
    print(f"  Aleatoric Uncertainty: {test_results['aleatoric_pct']:.1f}%")
    
    print("\n" + "="*80)
    print("Training completed!")
    print("="*80)
    
    return {
        'model': trained_model,
        'test_results': test_results,
        'history': history
    }


if __name__ == "__main__":
    results = main()

"""
=============================================================
 PAM0466 – SISTEMAS INTELIGENTES | 2026.1
 5º Projeto – MLP em PyTorch para Detecção de Intrusão
 Docente: Pedro Thiago Valério de Souza
=============================================================

Dataset: NSL-KDD
Arquivos: KDDTrain+.txt e KDDTest+.txt
Disponível em: https://www.kaggle.com/datasets/hassan06/nslkdd
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────
# 0. NOMES DAS COLUNAS
# ─────────────────────────────────────────────────────────
col_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "attack_type", "difficulty_level"
]
cat_cols = ['protocol_type', 'service', 'flag']

# ─────────────────────────────────────────────────────────
# TAREFA 1 – PREPARAÇÃO DOS DADOS
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("TAREFA 1 – Preparação dos Dados")
print("=" * 60)

# Carregamento
train_df = pd.read_csv("KDDTrain+.txt", header=None, names=col_names)
test_df  = pd.read_csv("KDDTest+.txt",  header=None, names=col_names)

print(f"Treino: {train_df.shape} | Teste: {test_df.shape}")
print("Labels treino:", train_df['attack_type'].value_counts().head(6).to_dict())

# Remover difficulty_level
train_df = train_df.drop(columns=['difficulty_level'])
test_df  = test_df.drop(columns=['difficulty_level'])

# One-Hot Encoding nas colunas categóricas
train_enc = pd.get_dummies(train_df, columns=cat_cols)
test_enc  = pd.get_dummies(test_df,  columns=cat_cols)

# Alinhar colunas (test pode ter categorias ausentes)
train_enc, test_enc = train_enc.align(test_enc, join='left', axis=1, fill_value=0)
print(f"Features após OHE: {train_enc.shape[1]}")

# Binarizar rótulo: normal → 0 | qualquer ataque → 1
train_enc['label'] = (train_enc['attack_type'] != 'normal').astype(np.float32)
test_enc['label']  = (test_enc['attack_type']  != 'normal').astype(np.float32)
print(f"Normal treino: {int((train_enc['label']==0).sum())} | Ataque: {int((train_enc['label']==1).sum())}")

feat_cols = [c for c in train_enc.columns if c not in ['attack_type', 'label']]

# Converter para numérico (segurança)
train_enc[feat_cols] = train_enc[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
test_enc[feat_cols]  = test_enc[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

X_train_full = train_enc[feat_cols].values.astype(np.float32)
y_train_full = train_enc['label'].values.astype(np.float32)
X_test       = test_enc[feat_cols].values.astype(np.float32)
y_test       = test_enc['label'].values.astype(np.float32)

# Z-score: fit APENAS no treino, transform em treino e teste
scaler       = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full).astype(np.float32)
X_test       = scaler.transform(X_test).astype(np.float32)
print("Z-score aplicado.")

# ─────────────────────────────────────────────────────────
# TAREFA 2 – IMPLEMENTAÇÃO DA MLP EM PYTORCH
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TAREFA 2 – Implementação da MLP")
print("=" * 60)

# Split treino/validação 80/20 SEM embaralhamento
split = int(len(X_train_full) * 0.8)
X_tr,  X_val = X_train_full[:split], X_train_full[split:]
y_tr,  y_val = y_train_full[:split], y_train_full[split:]
print(f"Treino: {len(X_tr)} | Validação: {len(X_val)} | Teste: {len(X_test)}")

# DataLoaders com mini-batches de 256
def make_loader(X, y, batch_size=256, shuffle=True):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y).unsqueeze(1))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

tr_loader  = make_loader(X_tr, y_tr, shuffle=True)
val_loader = make_loader(X_val, y_val, shuffle=False)
n_features = X_tr.shape[1]

# Arquitetura MLP
class MLP(nn.Module):
    """
    MLP com 3 camadas ocultas: 128 → 64 → 32 neurônios, ativação ReLU,
    Dropout(0.3) para regularização, saída linear (sem ativação).
    A BCEWithLogitsLoss aplica a sigmoide internamente, sendo numericamente
    mais estável do que aplicar sigmoid + BCELoss separadamente.
    """
    def __init__(self, n_in, hidden=[128, 64, 32], dropout=0.3):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))   # saída SEM ativação
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model     = MLP(n_features, hidden=[128, 64, 32], dropout=0.3)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(f"Arquitetura: input({n_features}) → 128 → 64 → 32 → 1")
print(f"Total de parâmetros: {sum(p.numel() for p in model.parameters()):,}")

# ─────────────────────────────────────────────────────────
# TAREFA 3 – TREINAMENTO E MONITORAMENTO
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TAREFA 3 – Treinamento (50 épocas, early stopping)")
print("=" * 60)

N_EPOCHS   = 50
train_losses, val_losses = [], []
best_val   = float('inf')
patience   = 7     # early stopping
no_improve = 0
best_epoch = 0

for epoch in range(1, N_EPOCHS + 1):
    # Treino
    model.train()
    ep_loss = 0.0
    for Xb, yb in tr_loader:
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()
        ep_loss += loss.item() * len(Xb)
    ep_loss /= len(X_tr)

    # Validação
    model.eval()
    vl_loss = 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            vl_loss += criterion(model(Xb), yb).item() * len(Xb)
    vl_loss /= len(X_val)

    train_losses.append(ep_loss)
    val_losses.append(vl_loss)

    # Salvar melhor modelo
    if vl_loss < best_val:
        best_val   = vl_loss
        best_epoch = epoch
        torch.save(model.state_dict(), 'best_mlp_ids.pt')
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"  Early stopping ativado na época {epoch} (melhor: época {best_epoch})")
            break

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Época {epoch:3d} | Loss Treino = {ep_loss:.4f} | Loss Val = {vl_loss:.4f}")

# ─────────────────────────────────────────────────────────
# TAREFA 4 – AVALIAÇÃO FINAL
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TAREFA 4 – Avaliação Final (conjunto de teste)")
print("=" * 60)

model.load_state_dict(torch.load('best_mlp_ids.pt'))
model.eval()

with torch.no_grad():
    logits = model(torch.tensor(X_test)).squeeze().numpy()

probs  = 1 / (1 + np.exp(-logits))   # sigmoide
y_pred = (probs >= 0.5).astype(int)  # limiar = 0.5

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
cm   = confusion_matrix(y_test, y_pred)

print(f"  Acurácia  : {acc:.4f}  ({acc*100:.1f}%)")
print(f"  Precisão  : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1-score  : {f1:.4f}")
print(f"\n  Matriz de Confusão:\n{cm}")
print("\n  Relatório por classe:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Ataque']))

# ─────────────────────────────────────────────────────────
# VISUALIZAÇÕES
# ─────────────────────────────────────────────────────────
plt.rcParams['figure.facecolor'] = 'white'
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# 1. Curvas de loss
ep_r = range(1, len(train_losses) + 1)
axes[0].plot(ep_r, train_losses, color='steelblue', lw=2.5, label='Loss Treino')
axes[0].plot(ep_r, val_losses,   color='coral',     lw=2.5, ls='--', label='Loss Validação')
axes[0].axvline(best_epoch, color='green', ls=':', lw=1.5, label=f'Melhor época={best_epoch}')
axes[0].set_xlabel('Épocas', fontsize=11)
axes[0].set_ylabel('BCE Loss', fontsize=11)
axes[0].set_title('Curvas de Loss – MLP (NSL-KDD)', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# 2. Matriz de Confusão
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Normal', 'Ataque'],
            yticklabels=['Normal', 'Ataque'],
            linewidths=1, annot_kws={"size": 14})
axes[1].set_title('Matriz de Confusão', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Classe Real')
axes[1].set_xlabel('Classe Predita')

# 3. Métricas
metricas = ['Acurácia', 'Precisão', 'Recall', 'F1-score']
valores  = [acc, prec, rec, f1]
colors   = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']
bars = axes[2].bar(metricas, valores, color=colors, edgecolor='white', width=0.5)
for bar, val in zip(bars, valores):
    axes[2].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005, f'{val:.3f}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
axes[2].set_ylim(0, 1.12)
axes[2].set_title('Métricas de Desempenho', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Valor')
axes[2].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('graficos_mlp_ids.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nGráfico salvo em 'graficos_mlp_ids.png'.")

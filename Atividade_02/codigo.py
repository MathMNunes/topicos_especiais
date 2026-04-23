"""
=============================================================
 PAM0466 – SISTEMAS INTELIGENTES | 2026.1
 2º Projeto – Regressão Logística (Heart Disease Cleveland)
 Docente: Pedro Thiago Valério de Souza
=============================================================

Dataset: UCI Heart Disease Cleveland
303 pacientes | 13 variáveis clínicas | variável alvo: target

Baixe o dataset em:
  https://archive.ics.uci.edu/dataset/45/heart+disease
Arquivo necessário: processed.cleveland.data
Coloque-o na mesma pasta que este script.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────
# 0. CARREGAMENTO DOS DADOS
# ─────────────────────────────────────────────────────────
df = pd.read_csv(
    "processed.cleveland.data",
    names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
           'restecg', 'thalach', 'exang', 'oldpeak',
           'slope', 'ca', 'thal', 'target'],
    na_values='?'   # o dataset usa '?' para valores ausentes
)

print("=" * 60)
print("DATASET – Visão Geral")
print("=" * 60)
print("Shape:", df.shape)
print("\nValores nulos:\n", df.isnull().sum()[df.isnull().sum() > 0])
print("\nDistribuição do target (original):", df['target'].value_counts().to_dict())

# ─────────────────────────────────────────────────────────
# TAREFA 1 – PRÉ-PROCESSAMENTO
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TAREFA 1 – Pré-processamento")
print("=" * 60)

# (a) Imputação pela mediana nas colunas com valores ausentes
for col in ['ca', 'thal']:
    mediana = df[col].median()
    nulos   = df[col].isnull().sum()
    df[col] = df[col].fillna(mediana)
    print(f"  [{col}] {nulos} nulo(s) imputado(s) com mediana = {mediana}")

print(f"  Nulos restantes: {df.isnull().sum().sum()}")

# (b) Binarização do target (0 = sem doença; 1 = com doença)
df['target'] = (df['target'] > 0).astype(int)
print(f"\n  Target binarizado: {df['target'].value_counts().to_dict()}")

# (c) Divisão treino/teste 80/20 com stratify
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n  Treino : {len(X_train)} amostras")
print(f"  Teste  : {len(X_test)} amostras")

# (d) StandardScaler apenas nas features numéricas contínuas
num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

X_train = X_train.copy()
X_test  = X_test.copy()

scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])  # fit + transform no treino
X_test[num_cols]  = scaler.transform(X_test[num_cols])       # apenas transform no teste

print(f"  Escalonamento aplicado em: {num_cols}")

# ─────────────────────────────────────────────────────────
# TAREFA 2 – TREINAMENTO
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TAREFA 2 – Treinamento do Modelo")
print("=" * 60)

model = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
model.fit(X_train, y_train)

print(f"  Iterações para convergência: {model.n_iter_[0]}")
print(f"  Intercepto: {model.intercept_[0]:.4f}")

# ─────────────────────────────────────────────────────────
# TAREFA 3 – AVALIAÇÃO
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TAREFA 3 – Avaliação do Modelo")
print("=" * 60)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_prob)
cm   = confusion_matrix(y_test, y_pred)

print(f"  Acurácia  : {acc:.4f}  ({acc*100:.1f}%)")
print(f"  Precisão  : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1-score  : {f1:.4f}")
print(f"  AUC-ROC   : {auc:.4f}")
print(f"\n  Matriz de Confusão:\n{cm}")
print("\n  Relatório completo:")
print(classification_report(y_test, y_pred,
      target_names=['Sem doença', 'Com doença']))

# ─────────────────────────────────────────────────────────
# VISUALIZAÇÕES
# ─────────────────────────────────────────────────────────
plt.rcParams['figure.facecolor'] = 'white'
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[0].plot(fpr, tpr, color='steelblue', lw=2.5, label=f'AUC = {auc:.3f}')
axes[0].plot([0, 1], [0, 1], 'r--', lw=1.5, label='Aleatório')
axes[0].fill_between(fpr, tpr, alpha=0.12, color='steelblue')
axes[0].set_xlabel('Taxa de Falsos Positivos')
axes[0].set_ylabel('Taxa de Verdadeiros Positivos (Recall)')
axes[0].set_title('Curva ROC', fontsize=12, fontweight='bold')
axes[0].legend(loc='lower right')
axes[0].grid(alpha=0.3)

# Matriz de Confusão
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Sem doença', 'Com doença'],
            yticklabels=['Sem doença', 'Com doença'],
            linewidths=1, annot_kws={"size": 14})
axes[1].set_title('Matriz de Confusão', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Classe Real')
axes[1].set_xlabel('Classe Predita')

# Coeficientes do Modelo
coefs  = pd.Series(model.coef_[0], index=X.columns).sort_values()
colors = ['coral' if c < 0 else 'steelblue' for c in coefs]
axes[2].barh(coefs.index, coefs.values, color=colors, edgecolor='white')
axes[2].axvline(0, color='black', lw=1.5)
axes[2].set_title('Coeficientes do Modelo', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Peso (log-odds)')
axes[2].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('graficos_heartdisease.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nGráfico salvo em 'graficos_heartdisease.png'.")

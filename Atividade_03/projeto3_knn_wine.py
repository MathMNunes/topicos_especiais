"""
=============================================================
 PAM0466 – SISTEMAS INTELIGENTES | 2026.1
 3º Projeto – k-NN para Classificação de Qualidade de Vinho
 Docente: Pedro Thiago Valério de Souza
=============================================================

Dataset: UCI Wine Quality – winequality-red.csv
1.599 amostras | 11 atributos físico-químicos | variável alvo: quality

Baixe o dataset em:
  https://archive.ics.uci.edu/ml/datasets/Wine+Quality
Arquivo necessário: winequality-red.csv
Coloque-o na mesma pasta que este script.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────
# 0. CARREGAMENTO DOS DADOS
# ─────────────────────────────────────────────────────────
df = pd.read_csv("winequality-red.csv", sep=";")

print("=" * 60)
print("DATASET – Visão Geral")
print("=" * 60)
print("Shape:", df.shape)
print("\nValores nulos:", df.isnull().sum().sum())
print("\nDistribuição de quality (original):")
print(df['quality'].value_counts().sort_index())

# ─────────────────────────────────────────────────────────
# TAREFA 1 – PREPARAÇÃO DOS DADOS
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TAREFA 1 – Preparação dos Dados")
print("=" * 60)

# Recodificação em 3 classes
def recodificar(q):
    if q <= 5:   return 'ruim'
    elif q <= 7: return 'medio'
    else:        return 'bom'

df['classe'] = df['quality'].apply(recodificar)
print("\nDistribuição das classes:")
print(df['classe'].value_counts())

# Separar features e target
features = [c for c in df.columns if c not in ['quality', 'classe']]
X = df[features]
y = df['classe']

# Normalização Min-Max
# Justificativa: o k-NN calcula distâncias euclidianas entre amostras.
# Sem normalização, features com maior amplitude (ex: total SO₂ ~0–289)
# dominariam o cálculo, ignorando features menores (ex: pH ~2.7–4.0).
# O Min-Max transforma todos os atributos para [0,1], garantindo que
# cada variável contribua igualmente para o cálculo de distância.
scaler  = MinMaxScaler()
X_norm  = pd.DataFrame(scaler.fit_transform(X), columns=features)

print("\nNormalização Min-Max aplicada em todos os 11 atributos.")
print("Intervalo pós-normalização: [0.0 , 1.0] para todas as features.")

# ─────────────────────────────────────────────────────────
# TAREFA 2 – DIVISÃO TREINO/TESTE
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TAREFA 2 – Divisão Treino/Teste")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.2, random_state=42, stratify=y)

print(f"Treino : {len(X_train)} amostras  ({len(X_train)/len(df)*100:.0f}%)")
print(f"Teste  : {len(X_test)}  amostras  ({len(X_test)/len(df)*100:.0f}%)")
print("\nProporção de classes no treino:")
print(y_train.value_counts(normalize=True).round(3))
print("\nProporção de classes no teste:")
print(y_test.value_counts(normalize=True).round(3))

# ─────────────────────────────────────────────────────────
# TAREFA 3 – ESCOLHA DO k (VALIDAÇÃO CRUZADA)
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TAREFA 3 – Escolha do k via Validação Cruzada (5-fold)")
print("=" * 60)

k_values    = [1, 3, 5, 7, 11, 15, 21]
cv          = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
mean_scores = []
std_scores  = []

for k in k_values:
    knn    = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring='accuracy')
    mean_scores.append(scores.mean())
    std_scores.append(scores.std())
    print(f"  k={k:2d}  →  Acurácia média = {scores.mean():.4f}  ±  {scores.std():.4f}")

best_k   = k_values[np.argmax(mean_scores)]
best_acc = max(mean_scores)
print(f"\n  ✓ Melhor k = {best_k}  (acurácia média = {best_acc:.4f})")

# ─────────────────────────────────────────────────────────
# TAREFA 4 – AVALIAÇÃO DO MODELO
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"TAREFA 4 – Avaliação Final (k={best_k}, dados de teste)")
print("=" * 60)

model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
cm   = confusion_matrix(y_test, y_pred, labels=['ruim', 'medio', 'bom'])

print(f"  Acurácia  : {acc:.4f}  ({acc*100:.1f}%)")
print(f"  Precisão  : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1-score  : {f1:.4f}")
print(f"\n  Matriz de Confusão (ruim / medio / bom):\n{cm}")
print("\n  Relatório por classe:")
print(classification_report(y_test, y_pred,
      target_names=['Ruim', 'Médio', 'Bom'], zero_division=0))

# ─────────────────────────────────────────────────────────
# VISUALIZAÇÕES
# ─────────────────────────────────────────────────────────
plt.rcParams['figure.facecolor'] = 'white'
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Acurácia × k
axes[0].errorbar(k_values, mean_scores, yerr=std_scores,
                 fmt='-o', color='steelblue', lw=2.5, capsize=5, markersize=8)
axes[0].axvline(best_k, color='red', linestyle='--', lw=2, label=f'Melhor k={best_k}')
axes[0].set_xlabel('Valor de k', fontsize=11)
axes[0].set_ylabel('Acurácia Média (5-fold CV)', fontsize=11)
axes[0].set_title('Acurácia × k (Validação Cruzada)', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)
axes[0].set_xticks(k_values)

# Matriz de Confusão
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Ruim', 'Médio', 'Bom'],
            yticklabels=['Ruim', 'Médio', 'Bom'],
            linewidths=1, annot_kws={"size": 14})
axes[1].set_title(f'Matriz de Confusão (k={best_k})', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Classe Real')
axes[1].set_xlabel('Classe Predita')

# F1-score por classe
report = classification_report(y_test, y_pred,
         target_names=['ruim', 'medio', 'bom'],
         output_dict=True, zero_division=0)
f1s    = [report[c]['f1-score'] for c in ['ruim', 'medio', 'bom']]
colors = ['#e74c3c', '#f39c12', '#27ae60']
bars   = axes[2].bar(['Ruim', 'Médio', 'Bom'], f1s,
                     color=colors, edgecolor='white', width=0.5)
for bar, val in zip(bars, f1s):
    axes[2].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01, f'{val:.2f}',
                 ha='center', va='bottom', fontsize=13, fontweight='bold')
axes[2].set_ylim(0, 1.15)
axes[2].set_title('F1-score por Classe', fontsize=12, fontweight='bold')
axes[2].set_ylabel('F1-score')
axes[2].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('graficos_wine_knn.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nGráfico salvo em 'graficos_wine_knn.png'.")

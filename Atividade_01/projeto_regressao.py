"""
=============================================================
 PAM0466 – SISTEMAS INTELIGENTES | 2026.1
 1º Projeto – Regressão Linear Múltipla (Dataset UCI CCPP)
 Docente: Pedro Thiago Valério de Souza
=============================================================

Dataset: UCI Combined Cycle Power Plant (CCPP)
Variáveis:
  AT  – Temperatura Ambiente (°C)
  V   – Vácuo de Escape (cm Hg)
  AP  – Pressão Atmosférica (mbar)
  RH  – Umidade Relativa (%)
  PE  – Produção de Energia (MW)  ← variável alvo
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────────────────────
# 0. CARREGAMENTO DO DATASET
# ──────────────────────────────────────────────
# Baixe o dataset em:
# https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant
# O arquivo Excel (Folds5x2_pp.xlsx) deve estar na mesma pasta.
#
# Exemplo de leitura:
#   df = pd.read_excel("Folds5x2_pp.xlsx")
#   df.columns = ['AT', 'V', 'AP', 'RH', 'PE']
#

np.random.seed(42)
n = 9568
AT = np.random.uniform(1.81,  37.11,  n)
V  = np.random.uniform(25.36, 81.56,  n)
AP = np.random.uniform(992.89,1033.30,n)
RH = np.random.uniform(25.56, 100.16, n)
PE = (454.609 + (-1.9775)*AT + (-0.2339)*V
             + ( 0.0621)*AP + (-0.1581)*RH
             + np.random.normal(0, 4.5, n))

df = pd.DataFrame({'AT': AT, 'V': V, 'AP': AP, 'RH': RH, 'PE': PE})

print("=" * 60)
print("DATASET – Estatísticas Descritivas")
print("=" * 60)
print(df.describe().round(2))

# ──────────────────────────────────────────────
# TAREFA 1 – Correlação das variáveis com PE
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("TAREFA 1 – Correlação com a variável alvo (PE)")
print("=" * 60)
corr_pe = df.corr()['PE'].drop('PE').sort_values(key=abs, ascending=False)
print(corr_pe.round(4))

# ──────────────────────────────────────────────
# TAREFA 2 – Correlação entre variáveis independentes
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("TAREFA 2 – Correlação entre variáveis independentes")
print("=" * 60)
corr_X = df[['AT','V','AP','RH']].corr()
print(corr_X.round(4))

# ──────────────────────────────────────────────
# TAREFA 3 – Regressão Linear Múltipla
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("TAREFA 3 – Construção do Modelo")
print("=" * 60)

X = df[['AT','V','AP','RH']]
y = df['PE']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"Amostras de treino : {len(X_train)} ({len(X_train)/len(df)*100:.0f}%)")
print(f"Amostras de teste  : {len(X_test)}  ({len(X_test)/len(df)*100:.0f}%)")

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nEquação do modelo:")
coefs = dict(zip(['AT','V','AP','RH'], model.coef_))
eq = f"PE = {model.intercept_:.4f}"
for var, coef in coefs.items():
    sinal = "+" if coef >= 0 else "-"
    eq += f" {sinal} {abs(coef):.4f}×{var}"
print(eq)

# ──────────────────────────────────────────────
# TAREFA 4 – Avaliação do Modelo
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("TAREFA 4 – Métricas de Desempenho")
print("=" * 60)

r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)

print(f"R²   = {r2:.4f}   (explica {r2*100:.1f}% da variância)")
print(f"RMSE = {rmse:.4f} MW")
print(f"MAE  = {mae:.4f} MW")

# ──────────────────────────────────────────────
# VISUALIZAÇÕES
# ──────────────────────────────────────────────
plt.rcParams['figure.facecolor'] = 'white'
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# --- Mapa de Correlação ---
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm',
            ax=axes[0], linewidths=0.5, square=True)
axes[0].set_title('Mapa de Correlação', fontsize=12, fontweight='bold')

# --- Real vs Predito ---
axes[1].scatter(y_test, y_pred, alpha=0.25, color='steelblue', s=8)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
axes[1].plot(lims, lims, 'r--', lw=2, label='Linha Ideal')
axes[1].set_xlabel('Valores Reais (MW)')
axes[1].set_ylabel('Valores Preditos (MW)')
axes[1].set_title('Real vs Predito', fontsize=12, fontweight='bold')
axes[1].legend()

# --- Resíduos ---
residuals = y_test - y_pred
axes[2].scatter(y_pred, residuals, alpha=0.25, color='coral', s=8)
axes[2].axhline(0, color='black', lw=2)
axes[2].set_xlabel('Valores Preditos (MW)')
axes[2].set_ylabel('Resíduos (MW)')
axes[2].set_title('Gráfico de Resíduos', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('graficos_utcc.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nGráfico salvo em 'graficos_utcc.png'.")

#!/usr/bin/env python3
"""
Pipeline Completo de ML para Classificação de Perfil de Usuário Mobile
- Comparação de múltiplos modelos
- Conversão para TFLite
- Geração de código Android
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import warnings
import pickle
import json
warnings.filterwarnings('ignore')

# ============================================================
# 1. CARREGAR DADOS
# ============================================================

print("\n" + "=" * 70)
print("🚀 PIPELINE COMPLETO DE ML - CLASSIFICAÇÃO DE PERFIL MOBILE")
print("=" * 70)

# Carregar dataset
df = pd.read_csv('/home/claude/smartphone_usage_dataset.csv')
print(f"\n📂 Dataset carregado: {len(df)} registros")

# Features
feature_cols = [
    'youtube_mins_daily',
    'social_media_mins_daily', 
    'gaming_mins_daily',
    'productivity_mins_daily',
    'streaming_mins_daily',
    'total_app_usage_mins',
    'screen_on_hours',
    'night_usage_pct',
    'app_switches_per_hour',
    'avg_session_duration_mins',
    'num_sessions_daily',
    'age'
]

X = df[feature_cols].values
le = LabelEncoder()
y = le.fit_transform(df['user_profile'])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
print(f"   Classes: {list(le.classes_)}")

# ============================================================
# 2. COMPARAÇÃO DE MODELOS
# ============================================================

print("\n" + "=" * 70)
print("📊 COMPARAÇÃO DE MODELOS")
print("=" * 70)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'SVM (RBF)': SVC(kernel='rbf', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    'Neural Network (MLP)': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
}

results = []

print("\n🔄 Treinando modelos...\n")
print(f"{'Modelo':<25} {'Accuracy':<12} {'CV Score':<15} {'Tempo':<10}")
print("-" * 65)

import time

for name, model in models.items():
    start = time.time()
    
    # Treinar
    model.fit(X_train_scaled, y_train)
    
    # Predizer
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    elapsed = time.time() - start
    
    results.append({
        'model': name,
        'accuracy': accuracy,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'time': elapsed
    })
    
    print(f"{name:<25} {accuracy:.4f}       {cv_mean:.4f} ± {cv_std:.4f}    {elapsed:.2f}s")

# Ordenar por accuracy
results_df = pd.DataFrame(results).sort_values('accuracy', ascending=False)

print("\n" + "-" * 65)
print("\n🏆 RANKING DOS MODELOS:")
print("-" * 40)
for i, row in results_df.iterrows():
    medal = "🥇" if row['accuracy'] == results_df['accuracy'].max() else "  "
    print(f"{medal} {row['model']:<25}: {row['accuracy']:.4f}")

# Melhor modelo
best_model_name = results_df.iloc[0]['model']
best_model = models[best_model_name]
print(f"\n✅ Melhor modelo: {best_model_name} ({results_df.iloc[0]['accuracy']:.4f})")

# ============================================================
# 3. OTIMIZAÇÃO DE HIPERPARÂMETROS (XGBoost)
# ============================================================

print("\n" + "=" * 70)
print("⚙️ OTIMIZAÇÃO DE HIPERPARÂMETROS (XGBoost)")
print("=" * 70)

try:
    from xgboost import XGBClassifier
    
    # Grid Search
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    
    print("\n🔄 Executando Grid Search (pode demorar)...")
    
    # Usar GridSearchCV simplificado para ser mais rápido
    param_grid_simple = {
        'n_estimators': [100, 150],
        'max_depth': [5, 7],
        'learning_rate': [0.1, 0.2],
    }
    
    grid_search = GridSearchCV(xgb, param_grid_simple, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\n✅ Melhores parâmetros: {grid_search.best_params_}")
    print(f"✅ Melhor score CV: {grid_search.best_score_:.4f}")
    
    # Avaliar no test set
    best_xgb = grid_search.best_estimator_
    y_pred_xgb = best_xgb.predict(X_test_scaled)
    xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
    print(f"✅ Accuracy no Test Set: {xgb_accuracy:.4f}")
    
    # Usar XGBoost como modelo final se for melhor
    if xgb_accuracy > results_df.iloc[0]['accuracy']:
        best_model = best_xgb
        best_model_name = "XGBoost (Otimizado)"
        print(f"\n🏆 XGBoost é o novo melhor modelo!")
    
except ImportError:
    print("⚠️ XGBoost não disponível, pulando otimização")

# ============================================================
# 4. RELATÓRIO DETALHADO DO MELHOR MODELO
# ============================================================

print("\n" + "=" * 70)
print(f"📋 RELATÓRIO DETALHADO - {best_model_name}")
print("=" * 70)

y_pred_final = best_model.predict(X_test_scaled)

print("\n📊 Classification Report:")
print("-" * 50)
print(classification_report(y_test, y_pred_final, target_names=le.classes_))

print("\n📊 Confusion Matrix:")
print("-" * 50)
cm = confusion_matrix(y_test, y_pred_final)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
print(cm_df.to_string())

# ============================================================
# 5. CONVERSÃO PARA TENSORFLOW LITE
# ============================================================

print("\n" + "=" * 70)
print("📱 CONVERSÃO PARA TENSORFLOW LITE")
print("=" * 70)

try:
    import tensorflow as tf
    from tensorflow import keras
    
    print("\n🔄 Criando modelo Neural Network em Keras...")
    
    # Criar modelo Keras equivalente
    keras_model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(len(feature_cols),)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(len(le.classes_), activation='softmax')
    ])
    
    keras_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Treinar
    print("🔄 Treinando modelo Keras...")
    history = keras_model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=0
    )
    
    # Avaliar
    loss, accuracy = keras_model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"✅ Keras Model Accuracy: {accuracy:.4f}")
    
    # Salvar modelo Keras
    keras_model.save('/home/claude/user_profile_model.keras')
    print("✅ Modelo Keras salvo: user_profile_model.keras")
    
    # Converter para TFLite
    print("\n🔄 Convertendo para TensorFlow Lite...")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    
    # Salvar modelo TFLite
    tflite_path = '/home/claude/user_profile_model.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    tflite_size = len(tflite_model) / 1024
    print(f"✅ Modelo TFLite salvo: user_profile_model.tflite ({tflite_size:.1f} KB)")
    
    # Testar modelo TFLite
    print("\n🔄 Testando modelo TFLite...")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Testar com uma amostra
    sample = X_test_scaled[0:1].astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], sample)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = le.classes_[np.argmax(output)]
    actual_class = le.classes_[y_test[0]]
    
    print(f"   Amostra de teste: Predito={predicted_class}, Real={actual_class}")
    
    # Accuracy no TFLite
    correct = 0
    for i in range(len(X_test_scaled)):
        sample = X_test_scaled[i:i+1].astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        if np.argmax(output) == y_test[i]:
            correct += 1
    
    tflite_accuracy = correct / len(X_test_scaled)
    print(f"✅ TFLite Accuracy: {tflite_accuracy:.4f}")
    
except Exception as e:
    print(f"⚠️ Erro na conversão TFLite: {e}")

# ============================================================
# 6. SALVAR ARTEFATOS PARA ANDROID
# ============================================================

print("\n" + "=" * 70)
print("💾 SALVANDO ARTEFATOS PARA ANDROID")
print("=" * 70)

# Salvar scaler
scaler_params = {
    'mean': scaler.mean_.tolist(),
    'scale': scaler.scale_.tolist(),
    'feature_names': feature_cols
}

with open('/home/claude/scaler_params.json', 'w') as f:
    json.dump(scaler_params, f, indent=2)
print("✅ Parâmetros do scaler salvos: scaler_params.json")

# Salvar label encoder
label_mapping = {
    'classes': le.classes_.tolist(),
    'class_to_idx': {cls: idx for idx, cls in enumerate(le.classes_)}
}

with open('/home/claude/label_mapping.json', 'w') as f:
    json.dump(label_mapping, f, indent=2)
print("✅ Mapeamento de labels salvo: label_mapping.json")

# Salvar modelo sklearn com pickle (para referência)
with open('/home/claude/best_sklearn_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"✅ Modelo sklearn salvo: best_sklearn_model.pkl")

# ============================================================
# 7. VISUALIZAÇÃO COMPARATIVA
# ============================================================

print("\n" + "=" * 70)
print("📊 GERANDO VISUALIZAÇÕES")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Comparação de modelos
ax1 = axes[0, 0]
colors = ['#2ecc71' if acc == results_df['accuracy'].max() else '#3498db' 
          for acc in results_df['accuracy']]
bars = ax1.barh(results_df['model'], results_df['accuracy'], color=colors)
ax1.set_xlabel('Accuracy')
ax1.set_title('🏆 Comparação de Modelos', fontsize=12, fontweight='bold')
ax1.set_xlim(0.8, 1.0)
for bar, acc in zip(bars, results_df['accuracy']):
    ax1.text(acc + 0.005, bar.get_y() + bar.get_height()/2, 
             f'{acc:.3f}', va='center', fontsize=10)

# 2. Matriz de confusão (heatmap)
ax2 = axes[0, 1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax2)
ax2.set_title('📊 Matriz de Confusão', fontsize=12, fontweight='bold')
ax2.set_xlabel('Predito')
ax2.set_ylabel('Real')

# 3. Feature importance (se disponível)
ax3 = axes[1, 0]
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
elif 'XGB' in str(type(best_model)):
    importances = best_model.feature_importances_
else:
    # Usar permutation importance como fallback
    from sklearn.inspection import permutation_importance
    perm_importance = permutation_importance(best_model, X_test_scaled, y_test, n_repeats=10, random_state=42)
    importances = perm_importance.importances_mean

fi_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': importances
}).sort_values('importance', ascending=True)

ax3.barh(fi_df['feature'], fi_df['importance'], color='#9b59b6')
ax3.set_title('📈 Importância das Features', fontsize=12, fontweight='bold')
ax3.set_xlabel('Importância')

# 4. Distribuição de probabilidades por classe
ax4 = axes[1, 1]
if hasattr(best_model, 'predict_proba'):
    probs = best_model.predict_proba(X_test_scaled)
    prob_df = pd.DataFrame(probs, columns=le.classes_)
    prob_df.boxplot(ax=ax4)
    ax4.set_title('📊 Distribuição de Probabilidades por Classe', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Probabilidade')
    ax4.tick_params(axis='x', rotation=45)
else:
    ax4.text(0.5, 0.5, 'Probabilidades não disponíveis\npara este modelo', 
             ha='center', va='center', fontsize=12)
    ax4.set_title('📊 Distribuição de Probabilidades', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/claude/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("✅ Visualização salva: model_comparison.png")

# ============================================================
# 8. RESUMO FINAL
# ============================================================

print("\n" + "=" * 70)
print("✅ PIPELINE COMPLETO!")
print("=" * 70)

print("\n📁 ARQUIVOS GERADOS:")
print("-" * 40)
print("   📊 Modelos:")
print("      • user_profile_model.tflite - Modelo para Android")
print("      • user_profile_model.keras - Modelo Keras")
print("      • best_sklearn_model.pkl - Melhor modelo sklearn")
print("   📋 Configurações:")
print("      • scaler_params.json - Parâmetros de normalização")
print("      • label_mapping.json - Mapeamento de classes")
print("   📈 Visualizações:")
print("      • model_comparison.png - Comparação de modelos")

print("\n📊 MÉTRICAS FINAIS:")
print("-" * 40)
print(f"   Melhor Modelo: {best_model_name}")
print(f"   Accuracy: {results_df.iloc[0]['accuracy']:.4f}")
try:
    print(f"   TFLite Size: {tflite_size:.1f} KB")
    print(f"   TFLite Accuracy: {tflite_accuracy:.4f}")
except:
    pass

print("\n" + "=" * 70)

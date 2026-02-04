#!/usr/bin/env python3
"""
Dataset Sintético de Uso de Smartphone para Classificação de Perfil de Usuário
Baseado nas especificações dos datasets:
- Kaggle: Mobile Device Usage and User Behavior Dataset
- Kaggle: Smartphone Usage and Behavioral Dataset

Categorias de perfil:
1. Content Consumer (YouTube heavy)
2. Social Butterfly (Social Media heavy)  
3. Gamer (Gaming heavy)
4. Productivity Focused (Work apps heavy)
5. Mixed User (Balanced usage)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings('ignore')

# Configuração de seed para reprodutibilidade
np.random.seed(42)
random.seed(42)

# ============================================================
# 1. GERAÇÃO DO DATASET SINTÉTICO
# ============================================================

def generate_synthetic_dataset(n_samples=1000):
    """
    Gera um dataset sintético realista de uso de smartphone
    com perfis de usuário definidos.
    """
    
    # Definição dos perfis e suas características
    profiles = {
        'content_consumer': {
            'youtube_mins': (180, 60),      # média, std
            'social_mins': (60, 30),
            'gaming_mins': (30, 20),
            'productivity_mins': (20, 15),
            'streaming_mins': (90, 40),
            'weight': 0.20
        },
        'social_butterfly': {
            'youtube_mins': (45, 25),
            'social_mins': (200, 50),
            'gaming_mins': (25, 15),
            'productivity_mins': (30, 20),
            'streaming_mins': (40, 25),
            'weight': 0.25
        },
        'gamer': {
            'youtube_mins': (60, 30),
            'social_mins': (40, 25),
            'gaming_mins': (240, 70),
            'productivity_mins': (15, 10),
            'streaming_mins': (50, 30),
            'weight': 0.18
        },
        'productivity_focused': {
            'youtube_mins': (30, 20),
            'social_mins': (35, 20),
            'gaming_mins': (10, 10),
            'productivity_mins': (180, 50),
            'streaming_mins': (25, 15),
            'weight': 0.17
        },
        'mixed_user': {
            'youtube_mins': (70, 35),
            'social_mins': (75, 35),
            'gaming_mins': (60, 35),
            'productivity_mins': (65, 35),
            'streaming_mins': (55, 30),
            'weight': 0.20
        }
    }
    
    # Dispositivos e sistemas operacionais
    devices_android = [
        'Samsung Galaxy S23', 'Samsung Galaxy A54', 'Xiaomi 13', 
        'Motorola Edge 40', 'OnePlus 11', 'Google Pixel 8',
        'Samsung Galaxy S22', 'Xiaomi Redmi Note 12', 'Realme 11 Pro'
    ]
    devices_ios = ['iPhone 15', 'iPhone 14', 'iPhone 13', 'iPhone SE']
    
    data = []
    
    for i in range(n_samples):
        # Seleciona perfil baseado nos pesos
        profile_names = list(profiles.keys())
        weights = [profiles[p]['weight'] for p in profile_names]
        selected_profile = np.random.choice(profile_names, p=weights)
        profile = profiles[selected_profile]
        
        # Gera dados demográficos
        age = np.random.randint(13, 65)
        gender = np.random.choice(['Male', 'Female'], p=[0.52, 0.48])
        
        # Ajuste baseado na idade
        age_factor = 1.0
        if age < 25:
            age_factor = 1.3  # Jovens usam mais
        elif age > 45:
            age_factor = 0.7  # Mais velhos usam menos
        
        # Gera tempos de uso (em minutos/dia) com distribuição normal truncada
        youtube_mins = max(0, np.random.normal(profile['youtube_mins'][0], profile['youtube_mins'][1]) * age_factor)
        social_mins = max(0, np.random.normal(profile['social_mins'][0], profile['social_mins'][1]) * age_factor)
        gaming_mins = max(0, np.random.normal(profile['gaming_mins'][0], profile['gaming_mins'][1]) * age_factor)
        productivity_mins = max(0, np.random.normal(profile['productivity_mins'][0], profile['productivity_mins'][1]))
        streaming_mins = max(0, np.random.normal(profile['streaming_mins'][0], profile['streaming_mins'][1]))
        
        # Métricas derivadas
        total_app_usage = youtube_mins + social_mins + gaming_mins + productivity_mins + streaming_mins
        screen_on_hours = total_app_usage / 60 + np.random.uniform(0.5, 1.5)  # tempo adicional
        
        # Horário de pico de uso
        if selected_profile == 'productivity_focused':
            peak_hour = np.random.choice([9, 10, 11, 14, 15, 16])
        elif selected_profile == 'gamer':
            peak_hour = np.random.choice([20, 21, 22, 23])
        elif selected_profile == 'social_butterfly':
            peak_hour = np.random.choice([12, 13, 18, 19, 20, 21])
        else:
            peak_hour = np.random.choice(range(8, 24))
        
        # Uso noturno (22h - 6h) como percentual
        if selected_profile == 'gamer':
            night_usage_pct = np.random.uniform(0.25, 0.45)
        elif selected_profile == 'productivity_focused':
            night_usage_pct = np.random.uniform(0.05, 0.15)
        else:
            night_usage_pct = np.random.uniform(0.10, 0.30)
        
        # Sistema operacional e dispositivo
        os_type = np.random.choice(['Android', 'iOS'], p=[0.72, 0.28])
        if os_type == 'Android':
            device = np.random.choice(devices_android)
        else:
            device = np.random.choice(devices_ios)
        
        # Número de apps instalados
        if selected_profile == 'productivity_focused':
            num_apps = np.random.randint(30, 60)
        elif selected_profile == 'gamer':
            num_apps = np.random.randint(50, 100)
        else:
            num_apps = np.random.randint(40, 80)
        
        # Consumo de bateria (mAh/dia) - correlacionado com uso
        battery_drain = int(total_app_usage * 2.5 + np.random.uniform(200, 500))
        
        # Consumo de dados (MB/dia)
        data_usage = int(youtube_mins * 12 + streaming_mins * 15 + social_mins * 3 + 
                        gaming_mins * 2 + productivity_mins * 0.5 + np.random.uniform(50, 200))
        
        # Frequência de troca de apps (switches/hora)
        if selected_profile == 'mixed_user':
            app_switches = np.random.uniform(8, 15)
        elif selected_profile == 'gamer':
            app_switches = np.random.uniform(2, 6)
        else:
            app_switches = np.random.uniform(5, 12)
        
        # Duração média de sessão (minutos)
        if selected_profile == 'gamer':
            avg_session_duration = np.random.uniform(30, 90)
        elif selected_profile == 'social_butterfly':
            avg_session_duration = np.random.uniform(5, 20)
        else:
            avg_session_duration = np.random.uniform(10, 40)
        
        # Número de sessões por dia
        num_sessions = int(total_app_usage / avg_session_duration)
        
        data.append({
            'user_id': i + 1,
            'age': age,
            'gender': gender,
            'device_model': device,
            'operating_system': os_type,
            'youtube_mins_daily': round(youtube_mins, 1),
            'social_media_mins_daily': round(social_mins, 1),
            'gaming_mins_daily': round(gaming_mins, 1),
            'productivity_mins_daily': round(productivity_mins, 1),
            'streaming_mins_daily': round(streaming_mins, 1),
            'total_app_usage_mins': round(total_app_usage, 1),
            'screen_on_hours': round(screen_on_hours, 2),
            'battery_drain_mah': battery_drain,
            'data_usage_mb': data_usage,
            'num_apps_installed': num_apps,
            'peak_usage_hour': peak_hour,
            'night_usage_pct': round(night_usage_pct, 3),
            'app_switches_per_hour': round(app_switches, 1),
            'avg_session_duration_mins': round(avg_session_duration, 1),
            'num_sessions_daily': num_sessions,
            'user_profile': selected_profile
        })
    
    return pd.DataFrame(data)

# ============================================================
# 2. ANÁLISE EXPLORATÓRIA (EDA)
# ============================================================

def exploratory_analysis(df):
    """Realiza análise exploratória completa do dataset"""
    
    print("=" * 70)
    print("ANÁLISE EXPLORATÓRIA DO DATASET DE USO DE SMARTPHONE")
    print("=" * 70)
    
    # Informações básicas
    print("\nINFORMAÇÕES BÁSICAS")
    print("-" * 40)
    print(f"Total de registros: {len(df)}")
    print(f"Total de features: {len(df.columns)}")
    print(f"\nColunas disponíveis:")
    for col in df.columns:
        print(f"  • {col}: {df[col].dtype}")
    
    # Estatísticas descritivas
    print("\n\nESTATÍSTICAS DESCRITIVAS (Features Numéricas)")
    print("-" * 40)
    numeric_cols = ['youtube_mins_daily', 'social_media_mins_daily', 'gaming_mins_daily',
                   'productivity_mins_daily', 'total_app_usage_mins', 'screen_on_hours',
                   'battery_drain_mah', 'data_usage_mb', 'age']
    print(df[numeric_cols].describe().round(2).to_string())
    
    # Distribuição dos perfis
    print("\n\nDISTRIBUIÇÃO DOS PERFIS DE USUÁRIO")
    print("-" * 40)
    profile_counts = df['user_profile'].value_counts()
    for profile, count in profile_counts.items():
        pct = count / len(df) * 100
        bar = "|" * int(pct / 2)
        print(f"{profile:25s}: {count:4d} ({pct:5.1f}%) {bar}")
    
    # Médias por perfil
    print("\n\nMÉDIA DE USO POR CATEGORIA E PERFIL (minutos/dia)")
    print("-" * 70)
    usage_cols = ['youtube_mins_daily', 'social_media_mins_daily', 'gaming_mins_daily', 
                  'productivity_mins_daily', 'streaming_mins_daily']
    profile_means = df.groupby('user_profile')[usage_cols].mean().round(1)
    print(profile_means.to_string())
    
    # Correlações
    print("\n\n🔗 TOP 10 CORRELAÇÕES MAIS FORTES")
    print("-" * 40)
    corr_matrix = df[numeric_cols + ['night_usage_pct', 'app_switches_per_hour']].corr()
    correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            correlations.append({
                'var1': corr_matrix.columns[i],
                'var2': corr_matrix.columns[j],
                'correlation': corr_matrix.iloc[i, j]
            })
    corr_df = pd.DataFrame(correlations).sort_values('correlation', key=abs, ascending=False)
    for _, row in corr_df.head(10).iterrows():
        print(f"{row['var1']:25s} <-> {row['var2']:25s}: {row['correlation']:+.3f}")
    
    # Análise por faixa etária
    print("\n\n👤 ANÁLISE POR FAIXA ETÁRIA")
    print("-" * 40)
    df['age_group'] = pd.cut(df['age'], bins=[0, 18, 25, 35, 45, 100], 
                             labels=['<18', '18-25', '26-35', '36-45', '45+'])
    age_analysis = df.groupby('age_group').agg({
        'total_app_usage_mins': 'mean',
        'gaming_mins_daily': 'mean',
        'social_media_mins_daily': 'mean',
        'productivity_mins_daily': 'mean'
    }).round(1)
    print(age_analysis.to_string())
    
    # Análise por SO
    print("\n\n📱 ANÁLISE POR SISTEMA OPERACIONAL")
    print("-" * 40)
    os_analysis = df.groupby('operating_system').agg({
        'total_app_usage_mins': 'mean',
        'screen_on_hours': 'mean',
        'num_apps_installed': 'mean',
        'data_usage_mb': 'mean'
    }).round(1)
    print(os_analysis.to_string())
    
    return df

# ============================================================
# 3. VISUALIZAÇÕES
# ============================================================

def create_visualizations(df):
    """Cria visualizações para análise"""
    
    # Configuração de estilo
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(16, 20))
    
    # 1. Distribuição dos perfis (Pizza)
    ax1 = fig.add_subplot(3, 3, 1)
    profile_counts = df['user_profile'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    labels = [p.replace('_', '\n') for p in profile_counts.index]
    ax1.pie(profile_counts.values, labels=labels, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    ax1.set_title('Distribuição dos Perfis de Usuário', fontsize=12, fontweight='bold')
    
    # 2. Uso por categoria (Barras empilhadas por perfil)
    ax2 = fig.add_subplot(3, 3, 2)
    usage_cols = ['youtube_mins_daily', 'social_media_mins_daily', 'gaming_mins_daily', 
                  'productivity_mins_daily', 'streaming_mins_daily']
    profile_means = df.groupby('user_profile')[usage_cols].mean()
    profile_means.plot(kind='bar', stacked=True, ax=ax2, 
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    ax2.set_title('Tempo de Uso por Categoria e Perfil', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Perfil')
    ax2.set_ylabel('Minutos/dia')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Boxplot de uso total por perfil
    ax3 = fig.add_subplot(3, 3, 3)
    df.boxplot(column='total_app_usage_mins', by='user_profile', ax=ax3)
    ax3.set_title('Uso Total de Apps por Perfil', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Perfil')
    ax3.set_ylabel('Minutos/dia')
    plt.suptitle('')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Distribuição de idade por perfil
    ax4 = fig.add_subplot(3, 3, 4)
    for profile in df['user_profile'].unique():
        subset = df[df['user_profile'] == profile]['age']
        ax4.hist(subset, bins=20, alpha=0.5, label=profile.replace('_', ' '))
    ax4.set_title('Distribuição de Idade por Perfil', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Idade')
    ax4.set_ylabel('Frequência')
    ax4.legend(fontsize=8)
    
    # 5. Correlação entre features principais
    ax5 = fig.add_subplot(3, 3, 5)
    corr_cols = ['youtube_mins_daily', 'social_media_mins_daily', 'gaming_mins_daily',
                 'productivity_mins_daily', 'screen_on_hours', 'night_usage_pct']
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                ax=ax5, fmt='.2f', square=True, linewidths=0.5)
    ax5.set_title('Matriz de Correlação', fontsize=12, fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)
    
    # 6. Scatter: Gaming vs Social Media
    ax6 = fig.add_subplot(3, 3, 6)
    for profile in df['user_profile'].unique():
        subset = df[df['user_profile'] == profile]
        ax6.scatter(subset['gaming_mins_daily'], subset['social_media_mins_daily'], 
                   alpha=0.5, label=profile.replace('_', ' '), s=30)
    ax6.set_title('Gaming vs Social Media', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Gaming (mins/dia)')
    ax6.set_ylabel('Social Media (mins/dia)')
    ax6.legend(fontsize=8)
    
    # 7. Uso por faixa etária
    ax7 = fig.add_subplot(3, 3, 7)
    age_groups = df.groupby('age_group')[['youtube_mins_daily', 'social_media_mins_daily', 
                                          'gaming_mins_daily', 'productivity_mins_daily']].mean()
    age_groups.plot(kind='bar', ax=ax7, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax7.set_title('Uso por Faixa Etária', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Faixa Etária')
    ax7.set_ylabel('Minutos/dia')
    ax7.legend(fontsize=8)
    ax7.tick_params(axis='x', rotation=0)
    
    # 8. Horário de pico por perfil
    ax8 = fig.add_subplot(3, 3, 8)
    peak_hours = df.groupby(['user_profile', 'peak_usage_hour']).size().unstack(fill_value=0)
    peak_hours_pct = peak_hours.div(peak_hours.sum(axis=1), axis=0)
    peak_hours_pct.T.plot(kind='line', ax=ax8, marker='o', markersize=3)
    ax8.set_title('Distribuição de Horário de Pico por Perfil', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Hora do Dia')
    ax8.set_ylabel('Proporção')
    ax8.legend(fontsize=8)
    
    # 9. Uso noturno vs Gaming
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.scatter(df['gaming_mins_daily'], df['night_usage_pct'] * 100, 
               c=df['user_profile'].astype('category').cat.codes, 
               cmap='viridis', alpha=0.5, s=30)
    ax9.set_title('Uso Noturno vs Gaming', fontsize=12, fontweight='bold')
    ax9.set_xlabel('Gaming (mins/dia)')
    ax9.set_ylabel('Uso Noturno (%)')
    
    plt.tight_layout()
    plt.savefig('/home/claude/eda_visualizations.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nVisualizações salvas em: /home/claude/eda_visualizations.png")

# ============================================================
# 4. PREPARAÇÃO PARA ML
# ============================================================

def prepare_for_ml(df):
    """Prepara os dados para treinamento de modelo ML"""
    
    print("\n" + "=" * 70)
    print("🤖 PREPARAÇÃO PARA MACHINE LEARNING")
    print("=" * 70)
    
    # Features recomendadas
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
    
    # Importância de features (baseada em correlação com target)
    print("\nFEATURES RECOMENDADAS PARA O MODELO:")
    print("-" * 40)
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(df['user_profile'])
    
    feature_importance = []
    for col in feature_cols:
        corr = abs(np.corrcoef(df[col], y_encoded)[0, 1])
        feature_importance.append({'feature': col, 'importance': corr})
    
    fi_df = pd.DataFrame(feature_importance).sort_values('importance', ascending=False)
    for _, row in fi_df.iterrows():
        bar = "|" * int(row['importance'] * 50)
        print(f"{row['feature']:30s}: {row['importance']:.3f} {bar}")
    
    # Estatísticas das classes
    print("\n\n BALANCEAMENTO DAS CLASSES:")
    print("-" * 40)
    class_counts = df['user_profile'].value_counts()
    for cls, count in class_counts.items():
        pct = count / len(df) * 100
        print(f"{cls:25s}: {count:4d} ({pct:5.1f}%)")
    
    # Salvar dataset processado
    df.to_csv('/home/claude/smartphone_usage_dataset.csv', index=False)
    print(f"\nDataset salvo em: /home/claude/smartphone_usage_dataset.csv")
    
    # Criar versão simplificada para ML
    ml_df = df[feature_cols + ['user_profile']].copy()
    ml_df.to_csv('/home/claude/smartphone_usage_ml_ready.csv', index=False)
    print(f"Dataset ML-ready salvo em: /home/claude/smartphone_usage_ml_ready.csv")
    
    return df, feature_cols

# ============================================================
# 5. MODELO BASELINE
# ============================================================

def train_baseline_model(df, feature_cols):
    """Treina um modelo baseline para classificação de perfil"""
    
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\n" + "=" * 70)
    print("TREINAMENTO DO MODELO BASELINE (Random Forest)")
    print("=" * 70)
    
    # Preparar dados
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
    
    # Treinar modelo
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    
    # Avaliação
    y_pred = rf.predict(X_test_scaled)
    
    print("\n RELATÓRIO DE CLASSIFICAÇÃO:")
    print("-" * 50)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Cross-validation
    cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"\nCross-Validation (5-fold):")
    print(f"   Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
    
    # Feature importance do modelo
    print("\nIMPORTÂNCIA DAS FEATURES (Random Forest):")
    print("-" * 50)
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in importances.iterrows():
        bar = "|" * int(row['importance'] * 100)
        print(f"{row['feature']:30s}: {row['importance']:.3f} {bar}")
    
    # Matriz de confusão
    print("\nMATRIZ DE CONFUSÃO:")
    print("-" * 50)
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print(cm_df.to_string())
    
    return rf, scaler, le

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + " " * 35)
    print("SISTEMA DE CLASSIFICAÇÃO DE PERFIL DE USUÁRIO MOBILE")
    print(" " * 35 + "\n")
    
    # 1. Gerar dataset
    print("Gerando dataset sintético...")
    df = generate_synthetic_dataset(n_samples=1000)
    print(f"   Dataset gerado com {len(df)} registros e {len(df.columns)} colunas")
    
    # 2. Análise exploratória
    df = exploratory_analysis(df)
    
    # 3. Visualizações
    print("\n Gerando visualizações...")
    create_visualizations(df)
    
    # 4. Preparação para ML
    df, feature_cols = prepare_for_ml(df)
    
    # 5. Modelo baseline
    model, scaler, le = train_baseline_model(df, feature_cols)
    
    print("\n" + "=" * 70)
    print("ANÁLISE COMPLETA!")
    print("/=" * 70)
    print("\n ARQUIVOS GERADOS:")
    print("   • smartphone_usage_dataset.csv - Dataset completo")
    print("   • smartphone_usage_ml_ready.csv - Dataset para ML")
    print("   • eda_visualizations.png - Visualizações EDA")
    print("\n PRÓXIMOS PASSOS:")
    print("   1. Testar outros modelos (XGBoost, SVM, Neural Network)")
    print("   2. Otimizar hiperparâmetros")
    print("   3. Converter para TFLite para rodar no Android")
    print("   4. Integrar com UsageStatsManager para dados reais")

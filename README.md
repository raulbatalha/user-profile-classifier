# 📱 User Profile Classifier - Android ML Project

Sistema completo de Machine Learning para classificação de perfil de usuário mobile, rodando 100% on-device (edge AI).

## Visão Geral

Este projeto identifica o perfil de uso do smartphone analisando padrões de comportamento:

| Perfil | Emoji | Descrição |
|--------|-------|-----------|
| **Content Consumer** | 📺 | Alto consumo de YouTube e streaming |
| **Social Butterfly** | 🦋 | Foco em redes sociais |
| **Gamer** | 🎮 | Foco em jogos mobile |
| **Productivity Focused** | 💼 | Apps de trabalho e produtividade |
| **Mixed User** | 📱 | Uso equilibrado |

## 📊 Métricas do Modelo

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 97.5% |
| **Modelo** | Neural Network (TFLite) |
| **Tamanho** | 8.8 KB |
| **Latência** | < 10ms |

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                      Android App                             │
├─────────────────────────────────────────────────────────────┤
│  UsageStatsManager API  →  Feature Engineering  →  TFLite   │
│     (Coleta dados)         (12 features)         (Inferência)│
├─────────────────────────────────────────────────────────────┤
│                    100% On-Device                            │
│                    Zero Cloud Dependency                     │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Estrutura do Projeto

```
android_project/
├── 01_project_config.kt          # Configuração do Gradle e Manifest
├── UsageDataCollector.kt         # Coleta de dados via UsageStatsManager
├── UserProfileClassifier.kt      # Inferência TFLite
├── MainActivity.kt               # Activity principal
└── res/
    └── layout/
        └── activity_main.xml     # Layout da UI

ml_artifacts/
├── user_profile_model.tflite     # Modelo TensorFlow Lite (8.8 KB)
├── scaler_params.json            # Parâmetros do StandardScaler
├── label_mapping.json            # Mapeamento de classes
└── model_comparison.png          # Visualização da comparação de modelos

datasets/
├── smartphone_usage_dataset.csv  # Dataset completo (1000 registros)
└── smartphone_usage_ml_ready.csv # Dataset pronto para ML
```

## 🚀 Como Usar

### 1. Preparar o Projeto Android

```bash
# Criar novo projeto no Android Studio
# Package: com.example.userprofileclassifier
# Min SDK: 26 (Android 8.0)
```

### 2. Adicionar Dependências (build.gradle.kts)

```kotlin
dependencies {
    implementation("org.tensorflow:tensorflow-lite:2.14.0")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.4")
    implementation("com.google.code.gson:gson:2.10.1")
    implementation("com.google.android.material:material:1.11.0")
}
```

### 3. Copiar Arquivos

1. Copie os arquivos `.kt` para `app/src/main/java/com/example/userprofileclassifier/`
2. Copie `activity_main.xml` para `app/src/main/res/layout/`
3. Copie os arquivos do modelo para `app/src/main/assets/`:
   - `user_profile_model.tflite`
   - `scaler_params.json`
   - `label_mapping.json`

### 4. Permissões (AndroidManifest.xml)

```xml
<uses-permission android:name="android.permission.PACKAGE_USAGE_STATS"
    tools:ignore="ProtectedPermissions" />
```

### 5. Compilar e Executar

```bash
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

## 🔬 Features Utilizadas pelo Modelo

| Feature | Descrição | Importância |
|---------|-----------|-------------|
| `social_media_mins_daily` | Minutos em redes sociais | 18.0% |
| `productivity_mins_daily` | Minutos em apps de trabalho | 16.8% |
| `gaming_mins_daily` | Minutos em jogos | 14.7% |
| `youtube_mins_daily` | Minutos no YouTube | 12.4% |
| `app_switches_per_hour` | Trocas de app por hora | 10.2% |
| `night_usage_pct` | % de uso noturno (22h-6h) | 9.3% |
| `avg_session_duration_mins` | Duração média de sessão | 8.2% |
| `streaming_mins_daily` | Minutos em streaming | 4.1% |
| `num_sessions_daily` | Sessões por dia | 2.4% |
| `screen_on_hours` | Horas com tela ligada | 1.7% |
| `total_app_usage_mins` | Total de uso de apps | 1.5% |
| `age` | Idade do usuário | 0.9% |

## 📈 Comparação de Modelos Testados

| Modelo | Accuracy | CV Score | Tempo |
|--------|----------|----------|-------|
| **Logistic Regression** | 97.50% | 97.25% | 0.04s |
| SVM (RBF) | 97.00% | 97.62% | 0.05s |
| K-Nearest Neighbors | 96.50% | 96.00% | 0.02s |
| XGBoost (Otimizado) | 96.00% | 96.50% | 2.1s |
| Random Forest | 96.00% | 96.25% | 1.12s |
| Neural Network (MLP) | 95.50% | 97.25% | 2.82s |

## 🔒 Privacidade

- ✅ **100% On-Device**: Nenhum dado sai do dispositivo
- ✅ **Sem Internet**: Funciona completamente offline
- ✅ **Sem Telemetria**: Não coleta dados para terceiros
- ✅ **Transparente**: Código aberto e auditável

## 🛠️ Customização

### Adicionar Novos Apps ao Mapeamento

Edite `UsageDataCollector.kt`:

```kotlin
private val appCategoryMap = mapOf(
    // Adicione novos apps aqui
    "com.novoapp.pacote" to AppCategory.GAMING,
    // ...
)
```

### Retreinar o Modelo

```python
# Use o script complete_ml_pipeline.py
python complete_ml_pipeline.py

# Copie o novo modelo para assets/
cp user_profile_model.tflite android_project/app/src/main/assets/
```

## 📚 Referências

- [TensorFlow Lite Android](https://www.tensorflow.org/lite/android)
- [UsageStatsManager API](https://developer.android.com/reference/android/app/usage/UsageStatsManager)
- [Kaggle: Smartphone Usage Dataset](https://www.kaggle.com/datasets/bhadramohit/smartphone-usage-and-behavioral-dataset)

## 📝 Licença

MIT License - Use livremente para fins acadêmicos e comerciais.

---

**Desenvolvido para pesquisa em Edge AI e classificação de comportamento mobile.**

*_Engº Esp. AI Raul Batalha_*

package com.example.userprofileclassifier

import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.userprofileclassifier.data.UsageDataCollector
import com.example.userprofileclassifier.databinding.ActivityMainBinding
import com.example.userprofileclassifier.ml.UserProfileClassifier
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Activity principal do app de classificação de perfil de usuário
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var usageCollector: UsageDataCollector
    private lateinit var classifier: UserProfileClassifier
    
    // Idade padrão do usuário (pode ser configurável)
    private var userAge: Int = 25

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        // Inicializar componentes
        usageCollector = UsageDataCollector(this)
        classifier = UserProfileClassifier(this)
        
        // Inicializar classificador
        if (!classifier.initialize()) {
            Toast.makeText(this, "Erro ao carregar modelo ML", Toast.LENGTH_LONG).show()
        }
        
        setupUI()
        checkPermission()
    }
    
    /**
     * Configura a interface do usuário
     */
    private fun setupUI() {
        // Botão de análise
        binding.btnAnalyze.setOnClickListener {
            if (usageCollector.hasUsageStatsPermission()) {
                analyzeUsage()
            } else {
                showPermissionDialog()
            }
        }
        
        // Slider de idade
        binding.sliderAge.addOnChangeListener { _, value, _ ->
            userAge = value.toInt()
            binding.tvAgeValue.text = "$userAge anos"
        }
        
        // Botão de configurar permissão
        binding.btnPermission.setOnClickListener {
            usageCollector.requestUsageStatsPermission()
        }
        
        // Botão de informações
        binding.btnInfo.setOnClickListener {
            showInfoDialog()
        }
    }
    
    /**
     * Verifica se a permissão de Usage Stats foi concedida
     */
    private fun checkPermission() {
        if (usageCollector.hasUsageStatsPermission()) {
            binding.cardPermission.visibility = View.GONE
            binding.cardAnalysis.visibility = View.VISIBLE
        } else {
            binding.cardPermission.visibility = View.VISIBLE
            binding.cardAnalysis.visibility = View.GONE
        }
    }
    
    override fun onResume() {
        super.onResume()
        checkPermission()
    }
    
    /**
     * Analisa o uso do dispositivo e classifica o perfil
     */
    private fun analyzeUsage() {
        binding.progressBar.visibility = View.VISIBLE
        binding.btnAnalyze.isEnabled = false
        binding.cardResult.visibility = View.GONE
        
        lifecycleScope.launch {
            try {
                // Coletar dados de uso (últimos 7 dias para melhor precisão)
                val usageStats = withContext(Dispatchers.IO) {
                    usageCollector.collectUsageStats(days = 7)
                }
                
                // Classificar perfil
                val result = withContext(Dispatchers.Default) {
                    classifier.classifyFromUsageStats(usageStats, userAge)
                }
                
                // Atualizar UI
                displayResults(usageStats, result)
                
            } catch (e: Exception) {
                Toast.makeText(
                    this@MainActivity, 
                    "Erro na análise: ${e.message}", 
                    Toast.LENGTH_LONG
                ).show()
            } finally {
                binding.progressBar.visibility = View.GONE
                binding.btnAnalyze.isEnabled = true
            }
        }
    }
    
    /**
     * Exibe os resultados da análise
     */
    private fun displayResults(
        stats: UsageDataCollector.UsageStats,
        result: UserProfileClassifier.ClassificationResult
    ) {
        binding.cardResult.visibility = View.VISIBLE
        
        // Perfil detectado
        binding.tvProfileEmoji.text = result.profile.emoji
        binding.tvProfileName.text = result.profile.displayName
        binding.tvProfileDescription.text = result.profile.description
        
        // Confiança
        val confidencePercent = (result.confidence * 100).toInt()
        binding.progressConfidence.progress = confidencePercent
        binding.tvConfidence.text = "$confidencePercent% confiança"
        
        // Estatísticas de uso
        binding.tvYoutubeTime.text = formatMinutes(stats.youtubeMinsDaily)
        binding.tvSocialTime.text = formatMinutes(stats.socialMediaMinsDaily)
        binding.tvGamingTime.text = formatMinutes(stats.gamingMinsDaily)
        binding.tvProductivityTime.text = formatMinutes(stats.productivityMinsDaily)
        binding.tvStreamingTime.text = formatMinutes(stats.streamingMinsDaily)
        binding.tvTotalTime.text = formatMinutes(stats.totalAppUsageMins)
        binding.tvScreenTime.text = String.format("%.1fh", stats.screenOnHours)
        binding.tvNightUsage.text = String.format("%.0f%%", stats.nightUsagePct * 100)
        
        // Probabilidades de todos os perfis
        displayProbabilities(result.allProbabilities)
    }
    
    /**
     * Exibe probabilidades de todos os perfis
     */
    private fun displayProbabilities(probs: Map<UserProfileClassifier.UserProfile, Float>) {
        val sortedProbs = probs.entries.sortedByDescending { it.value }
        
        val probsText = sortedProbs.joinToString("\n") { (profile, prob) ->
            "${profile.emoji} ${profile.displayName}: ${(prob * 100).toInt()}%"
        }
        
        binding.tvAllProbabilities.text = probsText
    }
    
    /**
     * Formata minutos para exibição
     */
    private fun formatMinutes(minutes: Float): String {
        return when {
            minutes < 60 -> "${minutes.toInt()} min"
            else -> String.format("%.1fh", minutes / 60)
        }
    }
    
    /**
     * Mostra diálogo de permissão
     */
    private fun showPermissionDialog() {
        AlertDialog.Builder(this)
            .setTitle("Permissão Necessária")
            .setMessage(
                "Para analisar seu perfil de uso, o app precisa de permissão para " +
                "acessar as estatísticas de uso do dispositivo.\n\n" +
                "Esta permissão é segura e os dados nunca saem do seu dispositivo."
            )
            .setPositiveButton("Configurar") { _, _ ->
                usageCollector.requestUsageStatsPermission()
            }
            .setNegativeButton("Cancelar", null)
            .show()
    }
    
    /**
     * Mostra informações sobre o app
     */
    private fun showInfoDialog() {
        AlertDialog.Builder(this)
            .setTitle("Sobre o App")
            .setMessage(
                "📱 User Profile Classifier\n\n" +
                "Este app usa Machine Learning para analisar seu padrão de uso " +
                "do smartphone e identificar seu perfil de usuário.\n\n" +
                "🔒 Privacidade: Toda análise é feita localmente no seu " +
                "dispositivo. Nenhum dado é enviado para servidores externos.\n\n" +
                "📊 Perfis detectáveis:\n" +
                "• 📺 Content Consumer\n" +
                "• 🦋 Social Butterfly\n" +
                "• 🎮 Gamer\n" +
                "• 💼 Productivity Focused\n" +
                "• 📱 Mixed User\n\n" +
                "🧠 Modelo: TensorFlow Lite\n" +
                "📈 Accuracy: 96%"
            )
            .setPositiveButton("OK", null)
            .show()
    }
    
    override fun onDestroy() {
        super.onDestroy()
        classifier.close()
    }
}

package com.example.userprofileclassifier.ml

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import org.json.JSONObject

/**
 * Classe responsável pela inferência do modelo TFLite
 * para classificação de perfil de usuário
 */
class UserProfileClassifier(private val context: Context) {

    private var interpreter: Interpreter? = null
    private var scalerMean: FloatArray? = null
    private var scalerScale: FloatArray? = null
    private var classLabels: Array<String>? = null
    
    // Número de features esperadas pelo modelo
    private val numFeatures = 12
    
    /**
     * Perfis de usuário possíveis
     */
    enum class UserProfile(val displayName: String, val emoji: String, val description: String) {
        CONTENT_CONSUMER("Content Consumer", "📺", "Alto consumo de YouTube e streaming"),
        GAMER("Gamer", "🎮", "Foco em jogos mobile"),
        MIXED_USER("Mixed User", "📱", "Uso equilibrado de diferentes apps"),
        PRODUCTIVITY_FOCUSED("Productivity Focused", "💼", "Foco em apps de trabalho"),
        SOCIAL_BUTTERFLY("Social Butterfly", "🦋", "Alto uso de redes sociais")
    }
    
    /**
     * Resultado da classificação
     */
    data class ClassificationResult(
        val profile: UserProfile,
        val confidence: Float,
        val allProbabilities: Map<UserProfile, Float>
    )

    // ============================================================
    // Inicialização
    // ============================================================
    
    /**
     * Inicializa o classificador carregando modelo e parâmetros
     */
    fun initialize(): Boolean {
        return try {
            // Carregar modelo TFLite
            val modelBuffer = loadModelFile("user_profile_model.tflite")
            val options = Interpreter.Options()
            options.setNumThreads(4)
            interpreter = Interpreter(modelBuffer, options)
            
            // Carregar parâmetros do scaler
            loadScalerParams()
            
            // Carregar mapeamento de labels
            loadLabelMapping()
            
            true
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }
    
    /**
     * Carrega o arquivo do modelo TFLite dos assets
     */
    private fun loadModelFile(modelPath: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    /**
     * Carrega os parâmetros do StandardScaler
     */
    private fun loadScalerParams() {
        val json = context.assets.open("scaler_params.json")
            .bufferedReader()
            .use { it.readText() }
        
        val jsonObject = JSONObject(json)
        
        val meanArray = jsonObject.getJSONArray("mean")
        scalerMean = FloatArray(meanArray.length()) { meanArray.getDouble(it).toFloat() }
        
        val scaleArray = jsonObject.getJSONArray("scale")
        scalerScale = FloatArray(scaleArray.length()) { scaleArray.getDouble(it).toFloat() }
    }
    
    /**
     * Carrega o mapeamento de labels
     */
    private fun loadLabelMapping() {
        val json = context.assets.open("label_mapping.json")
            .bufferedReader()
            .use { it.readText() }
        
        val jsonObject = JSONObject(json)
        val classesArray = jsonObject.getJSONArray("classes")
        
        classLabels = Array(classesArray.length()) { classesArray.getString(it) }
    }

    // ============================================================
    // Inferência
    // ============================================================
    
    /**
     * Classifica o perfil do usuário com base nos dados de uso
     */
    fun classify(
        youtubeMins: Float,
        socialMediaMins: Float,
        gamingMins: Float,
        productivityMins: Float,
        streamingMins: Float,
        totalAppUsageMins: Float,
        screenOnHours: Float,
        nightUsagePct: Float,
        appSwitchesPerHour: Float,
        avgSessionDurationMins: Float,
        numSessionsDaily: Int,
        age: Int
    ): ClassificationResult {
        
        // Preparar array de features
        val features = floatArrayOf(
            youtubeMins,
            socialMediaMins,
            gamingMins,
            productivityMins,
            streamingMins,
            totalAppUsageMins,
            screenOnHours,
            nightUsagePct,
            appSwitchesPerHour,
            avgSessionDurationMins,
            numSessionsDaily.toFloat(),
            age.toFloat()
        )
        
        // Normalizar features usando StandardScaler
        val scaledFeatures = normalizeFeatures(features)
        
        // Preparar input buffer
        val inputBuffer = ByteBuffer.allocateDirect(numFeatures * 4)
        inputBuffer.order(ByteOrder.nativeOrder())
        for (value in scaledFeatures) {
            inputBuffer.putFloat(value)
        }
        inputBuffer.rewind()
        
        // Preparar output buffer (5 classes)
        val outputBuffer = Array(1) { FloatArray(5) }
        
        // Executar inferência
        interpreter?.run(inputBuffer, outputBuffer)
        
        // Processar resultados
        val probabilities = outputBuffer[0]
        
        // Mapear probabilidades para perfis
        val profileProbs = mutableMapOf<UserProfile, Float>()
        classLabels?.forEachIndexed { index, label ->
            val profile = labelToProfile(label)
            profileProbs[profile] = probabilities[index]
        }
        
        // Encontrar perfil com maior probabilidade
        val maxIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
        val maxProb = probabilities[maxIndex]
        val predictedLabel = classLabels?.get(maxIndex) ?: "unknown"
        val predictedProfile = labelToProfile(predictedLabel)
        
        return ClassificationResult(
            profile = predictedProfile,
            confidence = maxProb,
            allProbabilities = profileProbs
        )
    }
    
    /**
     * Classifica usando dados do UsageDataCollector
     */
    fun classifyFromUsageStats(
        stats: com.example.userprofileclassifier.data.UsageDataCollector.UsageStats,
        userAge: Int
    ): ClassificationResult {
        return classify(
            youtubeMins = stats.youtubeMinsDaily,
            socialMediaMins = stats.socialMediaMinsDaily,
            gamingMins = stats.gamingMinsDaily,
            productivityMins = stats.productivityMinsDaily,
            streamingMins = stats.streamingMinsDaily,
            totalAppUsageMins = stats.totalAppUsageMins,
            screenOnHours = stats.screenOnHours,
            nightUsagePct = stats.nightUsagePct,
            appSwitchesPerHour = stats.appSwitchesPerHour,
            avgSessionDurationMins = stats.avgSessionDurationMins,
            numSessionsDaily = stats.numSessionsDaily,
            age = userAge
        )
    }
    
    /**
     * Normaliza features usando StandardScaler
     */
    private fun normalizeFeatures(features: FloatArray): FloatArray {
        val mean = scalerMean ?: return features
        val scale = scalerScale ?: return features
        
        return FloatArray(features.size) { i ->
            (features[i] - mean[i]) / scale[i]
        }
    }
    
    /**
     * Converte label string para enum UserProfile
     */
    private fun labelToProfile(label: String): UserProfile {
        return when (label) {
            "content_consumer" -> UserProfile.CONTENT_CONSUMER
            "gamer" -> UserProfile.GAMER
            "mixed_user" -> UserProfile.MIXED_USER
            "productivity_focused" -> UserProfile.PRODUCTIVITY_FOCUSED
            "social_butterfly" -> UserProfile.SOCIAL_BUTTERFLY
            else -> UserProfile.MIXED_USER
        }
    }

    // ============================================================
    // Limpeza
    // ============================================================
    
    /**
     * Libera recursos do interpretador
     */
    fun close() {
        interpreter?.close()
        interpreter = null
    }
}

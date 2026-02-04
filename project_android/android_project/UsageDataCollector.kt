package com.example.userprofileclassifier.data

import android.app.AppOpsManager
import android.app.usage.UsageEvents
import android.app.usage.UsageStatsManager
import android.content.Context
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.os.Build
import android.os.Process
import android.provider.Settings
import java.util.Calendar

/**
 * Classe responsável por coletar dados de uso de apps do dispositivo
 * usando a API UsageStatsManager do Android
 */
class UsageDataCollector(private val context: Context) {

    private val usageStatsManager: UsageStatsManager by lazy {
        context.getSystemService(Context.USAGE_STATS_SERVICE) as UsageStatsManager
    }
    
    private val packageManager: PackageManager by lazy {
        context.packageManager
    }

    // ============================================================
    // Categorização de Apps
    // ============================================================
    
    /**
     * Categorias de apps para classificação de perfil
     */
    enum class AppCategory {
        YOUTUBE,
        SOCIAL_MEDIA,
        GAMING,
        PRODUCTIVITY,
        STREAMING,
        COMMUNICATION,
        OTHER
    }
    
    /**
     * Mapeia package names para categorias
     */
    private val appCategoryMap = mapOf(
        // YouTube
        "com.google.android.youtube" to AppCategory.YOUTUBE,
        "com.google.android.youtube.tv" to AppCategory.YOUTUBE,
        "com.google.android.apps.youtube.music" to AppCategory.YOUTUBE,
        
        // Social Media
        "com.instagram.android" to AppCategory.SOCIAL_MEDIA,
        "com.facebook.katana" to AppCategory.SOCIAL_MEDIA,
        "com.facebook.lite" to AppCategory.SOCIAL_MEDIA,
        "com.twitter.android" to AppCategory.SOCIAL_MEDIA,
        "com.zhiliaoapp.musically" to AppCategory.SOCIAL_MEDIA, // TikTok
        "com.ss.android.ugc.trill" to AppCategory.SOCIAL_MEDIA, // TikTok
        "com.snapchat.android" to AppCategory.SOCIAL_MEDIA,
        "com.pinterest" to AppCategory.SOCIAL_MEDIA,
        "com.linkedin.android" to AppCategory.SOCIAL_MEDIA,
        "com.reddit.frontpage" to AppCategory.SOCIAL_MEDIA,
        "com.tumblr" to AppCategory.SOCIAL_MEDIA,
        "org.telegram.messenger" to AppCategory.SOCIAL_MEDIA,
        
        // Gaming (exemplos populares)
        "com.supercell.clashofclans" to AppCategory.GAMING,
        "com.supercell.clashroyale" to AppCategory.GAMING,
        "com.king.candycrushsaga" to AppCategory.GAMING,
        "com.mojang.minecraftpe" to AppCategory.GAMING,
        "com.roblox.client" to AppCategory.GAMING,
        "com.pubg.imobile" to AppCategory.GAMING,
        "com.tencent.ig" to AppCategory.GAMING, // PUBG Mobile
        "com.garena.game.codm" to AppCategory.GAMING,
        "com.mobile.legends" to AppCategory.GAMING,
        "com.activision.callofduty.shooter" to AppCategory.GAMING,
        "com.dts.freefireth" to AppCategory.GAMING, // Free Fire
        "com.ea.gp.fifamobile" to AppCategory.GAMING,
        "com.nianticlabs.pokemongo" to AppCategory.GAMING,
        
        // Productivity
        "com.google.android.apps.docs" to AppCategory.PRODUCTIVITY,
        "com.google.android.apps.docs.editors.docs" to AppCategory.PRODUCTIVITY,
        "com.google.android.apps.docs.editors.sheets" to AppCategory.PRODUCTIVITY,
        "com.google.android.apps.docs.editors.slides" to AppCategory.PRODUCTIVITY,
        "com.microsoft.office.word" to AppCategory.PRODUCTIVITY,
        "com.microsoft.office.excel" to AppCategory.PRODUCTIVITY,
        "com.microsoft.office.powerpoint" to AppCategory.PRODUCTIVITY,
        "com.microsoft.teams" to AppCategory.PRODUCTIVITY,
        "com.slack" to AppCategory.PRODUCTIVITY,
        "com.todoist" to AppCategory.PRODUCTIVITY,
        "com.notion.id" to AppCategory.PRODUCTIVITY,
        "com.evernote" to AppCategory.PRODUCTIVITY,
        "com.google.android.keep" to AppCategory.PRODUCTIVITY,
        "com.google.android.calendar" to AppCategory.PRODUCTIVITY,
        "com.microsoft.office.outlook" to AppCategory.PRODUCTIVITY,
        "com.trello" to AppCategory.PRODUCTIVITY,
        "com.asana.app" to AppCategory.PRODUCTIVITY,
        
        // Streaming
        "com.netflix.mediaclient" to AppCategory.STREAMING,
        "com.amazon.avod.thirdpartyclient" to AppCategory.STREAMING, // Prime Video
        "com.disney.disneyplus" to AppCategory.STREAMING,
        "com.hbo.hbonow" to AppCategory.STREAMING,
        "com.hulu.plus" to AppCategory.STREAMING,
        "com.spotify.music" to AppCategory.STREAMING,
        "com.apple.android.music" to AppCategory.STREAMING,
        "com.deezer.android.app" to AppCategory.STREAMING,
        "tv.twitch.android.app" to AppCategory.STREAMING,
        "com.crunchyroll.crunchyroid" to AppCategory.STREAMING,
        
        // Communication
        "com.whatsapp" to AppCategory.COMMUNICATION,
        "com.facebook.orca" to AppCategory.COMMUNICATION, // Messenger
        "com.google.android.apps.messaging" to AppCategory.COMMUNICATION,
        "com.viber.voip" to AppCategory.COMMUNICATION,
        "com.skype.raider" to AppCategory.COMMUNICATION,
        "us.zoom.videomeetings" to AppCategory.COMMUNICATION,
        "com.discord" to AppCategory.COMMUNICATION,
    )
    
    /**
     * Determina a categoria de um app pelo package name
     */
    fun getAppCategory(packageName: String): AppCategory {
        // Primeiro, verifica no mapa
        appCategoryMap[packageName]?.let { return it }
        
        // Se não encontrou, tenta determinar pela categoria do Play Store
        try {
            val appInfo = packageManager.getApplicationInfo(packageName, 0)
            
            // Verifica se é um jogo
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                if (appInfo.category == ApplicationInfo.CATEGORY_GAME) {
                    return AppCategory.GAMING
                }
                if (appInfo.category == ApplicationInfo.CATEGORY_PRODUCTIVITY) {
                    return AppCategory.PRODUCTIVITY
                }
                if (appInfo.category == ApplicationInfo.CATEGORY_SOCIAL) {
                    return AppCategory.SOCIAL_MEDIA
                }
                if (appInfo.category == ApplicationInfo.CATEGORY_VIDEO) {
                    return AppCategory.STREAMING
                }
            }
            
            // Heurística baseada no nome do package
            val lowerPackage = packageName.lowercase()
            return when {
                lowerPackage.contains("game") || lowerPackage.contains("play") -> AppCategory.GAMING
                lowerPackage.contains("social") -> AppCategory.SOCIAL_MEDIA
                lowerPackage.contains("video") || lowerPackage.contains("stream") -> AppCategory.STREAMING
                lowerPackage.contains("office") || lowerPackage.contains("work") -> AppCategory.PRODUCTIVITY
                else -> AppCategory.OTHER
            }
            
        } catch (e: PackageManager.NameNotFoundException) {
            return AppCategory.OTHER
        }
    }

    // ============================================================
    // Coleta de Dados de Uso
    // ============================================================
    
    /**
     * Data class para armazenar estatísticas de uso
     */
    data class UsageStats(
        val youtubeMinsDaily: Float,
        val socialMediaMinsDaily: Float,
        val gamingMinsDaily: Float,
        val productivityMinsDaily: Float,
        val streamingMinsDaily: Float,
        val totalAppUsageMins: Float,
        val screenOnHours: Float,
        val nightUsagePct: Float,
        val appSwitchesPerHour: Float,
        val avgSessionDurationMins: Float,
        val numSessionsDaily: Int,
        val peakUsageHour: Int,
        val numAppsUsed: Int
    )
    
    /**
     * Coleta estatísticas de uso dos últimos N dias
     */
    fun collectUsageStats(days: Int = 1): UsageStats {
        val calendar = Calendar.getInstance()
        val endTime = calendar.timeInMillis
        
        calendar.add(Calendar.DAY_OF_YEAR, -days)
        calendar.set(Calendar.HOUR_OF_DAY, 0)
        calendar.set(Calendar.MINUTE, 0)
        val startTime = calendar.timeInMillis
        
        // Coletar eventos de uso
        val usageEvents = usageStatsManager.queryEvents(startTime, endTime)
        
        // Estruturas para análise
        val appUsageTime = mutableMapOf<String, Long>()
        val hourlyUsage = mutableMapOf<Int, Long>()
        var nightUsageMs = 0L
        var totalUsageMs = 0L
        var appSwitches = 0
        var lastPackage = ""
        var sessionCount = 0
        var totalSessionDuration = 0L
        var sessionStart = 0L
        
        val event = UsageEvents.Event()
        
        while (usageEvents.hasNextEvent()) {
            usageEvents.getNextEvent(event)
            
            val packageName = event.packageName
            val timestamp = event.timeStamp
            val eventType = event.eventType
            
            when (eventType) {
                UsageEvents.Event.ACTIVITY_RESUMED -> {
                    // App entrou em foreground
                    if (sessionStart == 0L) {
                        sessionStart = timestamp
                        sessionCount++
                    }
                    
                    if (lastPackage.isNotEmpty() && lastPackage != packageName) {
                        appSwitches++
                    }
                    lastPackage = packageName
                }
                
                UsageEvents.Event.ACTIVITY_PAUSED -> {
                    // App saiu de foreground
                    if (sessionStart > 0) {
                        val duration = timestamp - sessionStart
                        
                        // Atualiza tempo por app
                        appUsageTime[packageName] = 
                            (appUsageTime[packageName] ?: 0L) + duration
                        
                        // Atualiza uso por hora
                        val cal = Calendar.getInstance()
                        cal.timeInMillis = timestamp
                        val hour = cal.get(Calendar.HOUR_OF_DAY)
                        hourlyUsage[hour] = (hourlyUsage[hour] ?: 0L) + duration
                        
                        // Verifica se é uso noturno (22h - 6h)
                        if (hour >= 22 || hour < 6) {
                            nightUsageMs += duration
                        }
                        
                        totalUsageMs += duration
                        totalSessionDuration += duration
                        sessionStart = 0L
                    }
                }
            }
        }
        
        // Calcular tempo por categoria (em minutos)
        var youtubeMins = 0f
        var socialMins = 0f
        var gamingMins = 0f
        var productivityMins = 0f
        var streamingMins = 0f
        
        for ((packageName, usageMs) in appUsageTime) {
            val mins = usageMs / 60000f
            
            when (getAppCategory(packageName)) {
                AppCategory.YOUTUBE -> youtubeMins += mins
                AppCategory.SOCIAL_MEDIA -> socialMins += mins
                AppCategory.GAMING -> gamingMins += mins
                AppCategory.PRODUCTIVITY -> productivityMins += mins
                AppCategory.STREAMING -> streamingMins += mins
                AppCategory.COMMUNICATION -> socialMins += mins * 0.5f // Parcial
                AppCategory.OTHER -> { /* Ignorar */ }
            }
        }
        
        // Calcular métricas
        val totalMins = totalUsageMs / 60000f
        val screenOnHours = totalMins / 60f
        val nightPct = if (totalUsageMs > 0) nightUsageMs.toFloat() / totalUsageMs else 0f
        val avgSessionMins = if (sessionCount > 0) totalSessionDuration / sessionCount / 60000f else 0f
        val usageHours = totalUsageMs / 3600000f
        val switchesPerHour = if (usageHours > 0) appSwitches / usageHours else 0f
        
        // Hora de pico
        val peakHour = hourlyUsage.maxByOrNull { it.value }?.key ?: 12
        
        // Dividir por número de dias
        val divider = days.toFloat()
        
        return UsageStats(
            youtubeMinsDaily = youtubeMins / divider,
            socialMediaMinsDaily = socialMins / divider,
            gamingMinsDaily = gamingMins / divider,
            productivityMinsDaily = productivityMins / divider,
            streamingMinsDaily = streamingMins / divider,
            totalAppUsageMins = totalMins / divider,
            screenOnHours = screenOnHours / divider,
            nightUsagePct = nightPct,
            appSwitchesPerHour = switchesPerHour,
            avgSessionDurationMins = avgSessionMins,
            numSessionsDaily = (sessionCount / divider).toInt(),
            peakUsageHour = peakHour,
            numAppsUsed = appUsageTime.size
        )
    }

    // ============================================================
    // Verificação de Permissão
    // ============================================================
    
    /**
     * Verifica se a permissão de Usage Stats foi concedida
     */
    fun hasUsageStatsPermission(): Boolean {
        val appOps = context.getSystemService(Context.APP_OPS_SERVICE) as AppOpsManager
        val mode = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            appOps.unsafeCheckOpNoThrow(
                AppOpsManager.OPSTR_GET_USAGE_STATS,
                Process.myUid(),
                context.packageName
            )
        } else {
            @Suppress("DEPRECATION")
            appOps.checkOpNoThrow(
                AppOpsManager.OPSTR_GET_USAGE_STATS,
                Process.myUid(),
                context.packageName
            )
        }
        return mode == AppOpsManager.MODE_ALLOWED
    }
    
    /**
     * Abre as configurações para conceder permissão de Usage Stats
     */
    fun requestUsageStatsPermission() {
        val intent = Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS)
        intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK
        context.startActivity(intent)
    }
}

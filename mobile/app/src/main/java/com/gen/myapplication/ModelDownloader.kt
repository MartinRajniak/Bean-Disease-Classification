package com.gen.myapplication

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

/**
 * Downloads and manages TFLite models from GitHub releases.
 *
 * Model versions are tracked and updated automatically when new versions are available.
 */
class ModelDownloader(private val context: Context) {

    companion object {
        private const val TAG = "ModelDownloader"

        // GitHub release configuration
        private const val GITHUB_REPO = "MartinRajniak/Bean-Disease-Classification"
        private const val MODEL_FILENAME = "bean_disease_model.tflite"

        // Default model version (bundled in app)
        private const val DEFAULT_VERSION = "v0.0.0-bundled"

        // Current release version to download
        private const val LATEST_VERSION = "v.0.1.0-mobile-net"

        // Model storage
        private const val MODELS_DIR = "models"
        private const val CURRENT_MODEL_FILE = "current_model.tflite"
        private const val VERSION_FILE = "model_version.txt"
    }

    private val modelsDir: File by lazy {
        File(context.filesDir, MODELS_DIR).also { it.mkdirs() }
    }

    /**
     * Get the URL for downloading a specific model version
     */
    private fun getModelUrl(version: String): String {
        return "https://github.com/$GITHUB_REPO/releases/download/$version/$MODEL_FILENAME"
    }

    /**
     * Get the currently installed model version
     */
    fun getCurrentVersion(): String {
        val versionFile = File(modelsDir, VERSION_FILE)
        return if (versionFile.exists()) {
            versionFile.readText().trim()
        } else {
            DEFAULT_VERSION
        }
    }

    /**
     * Check if a model update is available
     */
    fun isUpdateAvailable(): Boolean {
        val currentVersion = getCurrentVersion()
        return currentVersion != LATEST_VERSION
    }

    /**
     * Get the path to the current model file
     * Returns null if no downloaded model exists (should use bundled model)
     */
    fun getCurrentModelPath(): String? {
        val modelFile = File(modelsDir, CURRENT_MODEL_FILE)
        return if (modelFile.exists()) {
            modelFile.absolutePath
        } else {
            null
        }
    }

    /**
     * Download the latest model from GitHub releases
     *
     * @return true if download successful, false otherwise
     */
    suspend fun downloadLatestModel(): Boolean = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "Starting download of model version: $LATEST_VERSION")

            val url = URL(getModelUrl(LATEST_VERSION))
            val connection = url.openConnection() as HttpURLConnection
            connection.requestMethod = "GET"
            connection.connectTimeout = 30000
            connection.readTimeout = 30000
            connection.setRequestProperty("Accept", "application/octet-stream")

            // Follow redirects manually to handle GitHub's download redirects
            connection.instanceFollowRedirects = true
            HttpURLConnection.setFollowRedirects(true)

            val responseCode = connection.responseCode
            Log.d(TAG, "Response code: $responseCode")

            if (responseCode != HttpURLConnection.HTTP_OK) {
                Log.e(TAG, "Failed to download model: HTTP $responseCode")
                return@withContext false
            }

            val contentLength = connection.contentLength
            Log.d(TAG, "Content length: ${contentLength / 1024 / 1024}MB")

            // Download to temporary file first
            val tempFile = File(modelsDir, "temp_model.tflite")
            val outputStream = FileOutputStream(tempFile)
            val inputStream = connection.inputStream

            val buffer = ByteArray(8192)
            var bytesRead: Int
            var totalBytesRead = 0L

            while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                outputStream.write(buffer, 0, bytesRead)
                totalBytesRead += bytesRead

                if (contentLength > 0) {
                    val progress = (totalBytesRead * 100 / contentLength).toInt()
                    if (totalBytesRead % (1024 * 1024) == 0L) { // Log every MB
                        Log.d(TAG, "Download progress: $progress% (${totalBytesRead / 1024 / 1024}MB)")
                    }
                }
            }

            outputStream.close()
            inputStream.close()
            connection.disconnect()

            Log.d(TAG, "Download complete: ${totalBytesRead / 1024 / 1024}MB")

            // Verify file size
            if (tempFile.length() < 1024 * 100) { // Less than 100KB is suspicious
                Log.e(TAG, "Downloaded file is too small: ${tempFile.length()} bytes")
                tempFile.delete()
                return@withContext false
            }

            // Move temp file to final location
            val finalFile = File(modelsDir, CURRENT_MODEL_FILE)
            if (finalFile.exists()) {
                finalFile.delete()
            }
            tempFile.renameTo(finalFile)

            // Update version file
            val versionFile = File(modelsDir, VERSION_FILE)
            versionFile.writeText(LATEST_VERSION)

            Log.d(TAG, "Model successfully downloaded and saved: ${finalFile.absolutePath}")
            true

        } catch (e: Exception) {
            Log.e(TAG, "Error downloading model", e)
            false
        }
    }

    /**
     * Delete downloaded model and revert to bundled model
     */
    fun deleteDownloadedModel() {
        val modelFile = File(modelsDir, CURRENT_MODEL_FILE)
        if (modelFile.exists()) {
            modelFile.delete()
            Log.d(TAG, "Deleted downloaded model")
        }

        val versionFile = File(modelsDir, VERSION_FILE)
        if (versionFile.exists()) {
            versionFile.delete()
            Log.d(TAG, "Deleted version file")
        }
    }

    /**
     * Get download status information
     */
    data class ModelInfo(
        val currentVersion: String,
        val latestVersion: String,
        val isUpdateAvailable: Boolean,
        val downloadedModelExists: Boolean,
        val downloadedModelSize: Long? = null
    )

    fun getModelInfo(): ModelInfo {
        val modelFile = File(modelsDir, CURRENT_MODEL_FILE)
        return ModelInfo(
            currentVersion = getCurrentVersion(),
            latestVersion = LATEST_VERSION,
            isUpdateAvailable = isUpdateAvailable(),
            downloadedModelExists = modelFile.exists(),
            downloadedModelSize = if (modelFile.exists()) modelFile.length() else null
        )
    }
}

package com.gen.myapplication

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileNotFoundException
import java.io.FileOutputStream
import java.io.IOException
import java.net.ConnectException
import java.net.HttpURLConnection
import java.net.SocketTimeoutException
import java.net.URL
import java.net.UnknownHostException

/**
 * Represents the result of a model download operation
 */
sealed class DownloadResult {
    object Success : DownloadResult()
    data class Failure(val userMessage: String, val technicalCause: Exception? = null) : DownloadResult()
}

/**
 * Downloads and manages TFLite models from GitHub releases.
 *
 * Model versions are tracked and updated automatically when new versions are available.
 */
open class ModelDownloader(
    private val context: Context,
    private val gitHubApi: GitHubApi
) {

    companion object {
        private const val TAG = "ModelDownloader"

        // GitHub release configuration
        private const val GITHUB_REPO = "MartinRajniak/Bean-Disease-Classification"
        private const val MODEL_FILENAME = "bean_disease_model.tflite"

        // Default model version (bundled in app)
        private const val DEFAULT_VERSION = "v0.1.0-bundled"

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
    open fun getCurrentVersion(): String {
        val versionFile = File(modelsDir, VERSION_FILE)
        return if (versionFile.exists()) {
            versionFile.readText().trim()
        } else {
            DEFAULT_VERSION
        }
    }

    /**
     * Get the path to the current model file
     * Returns null if no downloaded model exists (should use bundled model)
     */
    open fun getCurrentModelPath(): String? {
        val modelFile = File(modelsDir, CURRENT_MODEL_FILE)
        return if (modelFile.exists()) {
            modelFile.absolutePath
        } else {
            null
        }
    }

    /**
     * Fetch the latest version from GitHub releases
     */
    suspend fun getLatestVersion(): Result<String> {
        return gitHubApi.fetchLatestVersionInfo()
    }

    /**
     * Check if an update is available by comparing current version with provided latest version
     */
    fun isUpdateAvailable(latestVersion: String): Boolean {
        val currentVersion = getCurrentVersion()
        return currentVersion != latestVersion
    }

    /**
     * Download the latest model from GitHub releases
     *
     * @return DownloadResult with success or detailed failure information
     */
    open suspend fun downloadModelWithResult(modelVersion: String): DownloadResult = withContext(Dispatchers.IO) {
        try {
            performDownload(modelVersion)
        } catch (e: Exception) {
            Log.e(TAG, "Error downloading model", e)
            DownloadResult.Failure(mapDownloadErrorToUserMessage(e), e)
        }
    }

    /**
     * Download the latest model from GitHub releases (legacy method for compatibility)
     *
     * @return true if download successful, false otherwise
     */
    open suspend fun downloadModel(modelVersion: String): Boolean = withContext(Dispatchers.IO) {
        return@withContext when (downloadModelWithResult(modelVersion)) {
            is DownloadResult.Success -> true
            is DownloadResult.Failure -> false
        }
    }

    private suspend fun performDownload(modelVersion: String): DownloadResult = withContext(Dispatchers.IO) {
        Log.d(TAG, "Starting download of model version: $modelVersion")

        val url = URL(getModelUrl(modelVersion))
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
            connection.disconnect()
            val errorMessage = when (responseCode) {
                HttpURLConnection.HTTP_NOT_FOUND -> "Model version not found. It may have been removed or the version number is incorrect."
                HttpURLConnection.HTTP_FORBIDDEN -> "Access denied. You may have exceeded download limits."
                HttpURLConnection.HTTP_UNAVAILABLE -> "Download service is temporarily unavailable. Please try again later."
                else -> "Download failed with server error. Please try again later."
            }
            return@withContext DownloadResult.Failure(errorMessage)
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
            tempFile.delete()
            return@withContext DownloadResult.Failure("Downloaded file appears to be corrupted or incomplete. Please try again.")
        }

        // Move temp file to final location
        val finalFile = File(modelsDir, CURRENT_MODEL_FILE)
        if (finalFile.exists()) {
            finalFile.delete()
        }

        if (!tempFile.renameTo(finalFile)) {
            tempFile.delete()
            return@withContext DownloadResult.Failure("Unable to save the downloaded model. Check available storage space.")
        }

        // Update version file
        val versionFile = File(modelsDir, VERSION_FILE)
        versionFile.writeText(modelVersion)

        Log.d(TAG, "Model successfully downloaded and saved: ${finalFile.absolutePath}")
        DownloadResult.Success
    }

    /**
     * Maps technical download exceptions to user-friendly error messages
     */
    private fun mapDownloadErrorToUserMessage(exception: Exception): String {
        return when (exception) {
            is UnknownHostException -> "No internet connection available. Please check your network and try again."
            is SocketTimeoutException -> "Download timed out. Please check your connection and try again."
            is ConnectException -> "Unable to connect to download servers. Please try again later."
            is FileNotFoundException -> "The requested model version was not found. It may have been removed."
            is IOException -> {
                when {
                    exception.message?.contains("No space left") == true -> "Not enough storage space available. Please free up space and try again."
                    exception.message?.contains("Permission denied") == true -> "Unable to save file due to permission restrictions."
                    else -> "Download failed due to network or storage issues. Please try again."
                }
            }
            is SecurityException -> "Unable to save file due to permission restrictions."
            else -> "An unexpected error occurred during download. Please try again later."
        }
    }

    /**
     * Delete downloaded model and revert to bundled model
     */
    open fun deleteDownloadedModel() {
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
        val downloadedModelExists: Boolean,
        val downloadedModelSize: Long? = null
    )

    open fun getModelInfo(): ModelInfo {
        val modelFile = File(modelsDir, CURRENT_MODEL_FILE)
        return ModelInfo(
            currentVersion = getCurrentVersion(),
            downloadedModelExists = modelFile.exists(),
            downloadedModelSize = if (modelFile.exists()) modelFile.length() else null
        )
    }
}

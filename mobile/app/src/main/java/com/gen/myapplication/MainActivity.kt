package com.gen.myapplication

import android.Manifest
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.gen.myapplication.ui.theme.MyApplicationTheme
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import com.google.accompanist.permissions.shouldShowRationale
import io.ktor.client.engine.okhttp.OkHttp
import kotlinx.coroutines.launch

/**
 * Represents the different states of the model update dialog
 */
sealed class ModelUpdateState {
    data class Loading(val currentVersion: String) : ModelUpdateState()
    data class UpToDate(val currentVersion: String, val latestVersion: String) : ModelUpdateState()
    data class UpdateAvailable(val currentVersion: String, val latestVersion: String) : ModelUpdateState()
    data class Downloading(val currentVersion: String, val latestVersion: String) : ModelUpdateState()
    data class DownloadSuccess(val currentVersion: String, val newVersion: String) : ModelUpdateState()
    data class DownloadFailed(val currentVersion: String, val latestVersion: String, val error: String) : ModelUpdateState()
    data class NetworkError(val currentVersion: String, val error: String) : ModelUpdateState()
}

class MainActivity : ComponentActivity() {
    @OptIn(ExperimentalPermissionsApi::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MyApplicationTheme {
                CameraScreen()
            }
        }
    }
}

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun CameraScreen() {
    val cameraPermissionState = rememberPermissionState(Manifest.permission.CAMERA)
    var classificationResult by remember {
        mutableStateOf<BeansDiseaseClassifier.ClassificationResult?>(
            null
        )
    }
    var showModelUpdateDialog by remember { mutableStateOf(false) }

    when {
        cameraPermissionState.status.isGranted -> {
            Box(modifier = Modifier.fillMaxSize()) {
                CameraPreview(
                    modifier = Modifier.fillMaxSize(),
                    onClassificationResult = { result ->
                        classificationResult = result
                    }
                )

                // Model update button in top-right
                TextButton(
                    onClick = { showModelUpdateDialog = true },
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .padding(16.dp)
                ) {
                    Text("📥 Model Update", color = Color.White)
                }

                classificationResult?.let { result ->
                    ClassificationOverlay(
                        result = result,
                        modifier = Modifier
                            .align(Alignment.BottomCenter)
                            .padding(16.dp)
                    )
                }

                if (showModelUpdateDialog) {
                    ModelUpdateDialog(
                        onDismiss = { showModelUpdateDialog = false }
                    )
                }
            }
        }

        cameraPermissionState.status.shouldShowRationale -> {
            PermissionRationaleContent {
                cameraPermissionState.launchPermissionRequest()
            }
        }

        else -> {
            LaunchedEffect(Unit) {
                cameraPermissionState.launchPermissionRequest()
            }
        }
    }
}

@Composable
private fun PermissionRationaleContent(onRequestPermission: () -> Unit) {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                text = "Camera permission is required to use this feature. Please grant the permission.",
                modifier = Modifier.padding(16.dp)
            )
            Button(onClick = onRequestPermission) {
                Text("Grant Permission")
            }
        }
    }
}

@Composable
private fun ClassificationOverlay(
    result: BeansDiseaseClassifier.ClassificationResult,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                text = "Bean Disease Classification",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )

            Text(
                text = "Disease: ${
                    result.label.replace("_", " ").replaceFirstChar { it.uppercase() }
                }",
                style = MaterialTheme.typography.bodyLarge,
                modifier = Modifier.padding(top = 8.dp)
            )

            Text(
                text = "Confidence: ${String.format("%.1f%%", result.confidence * 100)}",
                style = MaterialTheme.typography.bodyLarge,
                color = if (result.confidence > 0.7f) Color.Black else Color.Red,
                modifier = Modifier.padding(top = 4.dp)
            )
        }
    }
}

@Composable
private fun ModelUpdateDialog(
    onDismiss: () -> Unit
) {
    val context = androidx.compose.ui.platform.LocalContext.current
    val gitHubApi = remember { GitHubApi(OkHttp.create()) }
    val modelDownloader = remember { ModelDownloader(context, gitHubApi) }
    val coroutineScope = rememberCoroutineScope()

    val modelInfo = remember { modelDownloader.getModelInfo() }
    var updateState by remember { mutableStateOf<ModelUpdateState>(ModelUpdateState.Loading(modelInfo.currentVersion)) }

    LaunchedEffect(modelDownloader) {
        val latestVersionResult = modelDownloader.getLatestVersion()
        updateState = if (latestVersionResult.isSuccess) {
            val latestVersion = latestVersionResult.getOrNull()!!
            if (modelDownloader.isUpdateAvailable(latestVersion)) {
                ModelUpdateState.UpdateAvailable(modelInfo.currentVersion, latestVersion)
            } else {
                ModelUpdateState.UpToDate(modelInfo.currentVersion, latestVersion)
            }
        } else {
            val error = latestVersionResult.exceptionOrNull()
            val userFriendlyMessage = when (error) {
                is GitHubApiError -> error.message ?: "Unable to check for updates"
                else -> "Unable to check for updates. Please try again later."
            }
            ModelUpdateState.NetworkError(modelInfo.currentVersion, userFriendlyMessage)
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black.copy(alpha = 0.5f)),
        contentAlignment = Alignment.Center
    ) {
        Card(
            modifier = Modifier
                .padding(32.dp)
                .fillMaxWidth(),
            shape = RoundedCornerShape(16.dp)
        ) {
            Column(
                modifier = Modifier.padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                ModelUpdateHeader()

                Spacer(modifier = Modifier.height(16.dp))

                ModelVersionInfo(updateState, modelInfo)

                Spacer(modifier = Modifier.height(16.dp))

                ModelUpdateContent(updateState)

                Spacer(modifier = Modifier.height(24.dp))

                ModelUpdateActions(
                    updateState = updateState,
                    onDownload = { latestVersion ->
                        updateState = ModelUpdateState.Downloading(updateState.getCurrentVersion(), latestVersion)
                        coroutineScope.launch {
                            val downloadResult = modelDownloader.downloadModelWithResult(latestVersion)
                            updateState = when (downloadResult) {
                                is DownloadResult.Success -> {
                                    ModelUpdateState.DownloadSuccess(updateState.getCurrentVersion(), latestVersion)
                                }
                                is DownloadResult.Failure -> {
                                    ModelUpdateState.DownloadFailed(
                                        updateState.getCurrentVersion(),
                                        latestVersion,
                                        downloadResult.userMessage
                                    )
                                }
                            }
                        }
                    },
                    onRetry = {
                        updateState = ModelUpdateState.Loading(updateState.getCurrentVersion())
                        coroutineScope.launch {
                            val latestVersionResult = modelDownloader.getLatestVersion()
                            updateState = if (latestVersionResult.isSuccess) {
                                val latestVersion = latestVersionResult.getOrNull()!!
                                if (modelDownloader.isUpdateAvailable(latestVersion)) {
                                    ModelUpdateState.UpdateAvailable(modelInfo.currentVersion, latestVersion)
                                } else {
                                    ModelUpdateState.UpToDate(modelInfo.currentVersion, latestVersion)
                                }
                            } else {
                                val error = latestVersionResult.exceptionOrNull()
                                val userFriendlyMessage = when (error) {
                                    is GitHubApiError -> error.message ?: "Unable to check for updates"
                                    else -> "Unable to check for updates. Please try again later."
                                }
                                ModelUpdateState.NetworkError(modelInfo.currentVersion, userFriendlyMessage)
                            }
                        }
                    },
                    onDismiss = onDismiss
                )
            }
        }
    }
}

/**
 * Extension function to get current version from any state
 */
private fun ModelUpdateState.getCurrentVersion(): String = when (this) {
    is ModelUpdateState.Loading -> currentVersion
    is ModelUpdateState.UpToDate -> currentVersion
    is ModelUpdateState.UpdateAvailable -> currentVersion
    is ModelUpdateState.Downloading -> currentVersion
    is ModelUpdateState.DownloadSuccess -> currentVersion
    is ModelUpdateState.DownloadFailed -> currentVersion
    is ModelUpdateState.NetworkError -> currentVersion
}

/**
 * Extension function to get latest version from states that have it
 */
private fun ModelUpdateState.getLatestVersion(): String? = when (this) {
    is ModelUpdateState.UpToDate -> latestVersion
    is ModelUpdateState.UpdateAvailable -> latestVersion
    is ModelUpdateState.Downloading -> latestVersion
    is ModelUpdateState.DownloadFailed -> latestVersion
    else -> null
}

@Composable
private fun ModelUpdateHeader() {
    Text(
        text = "Model Update",
        style = MaterialTheme.typography.headlineSmall,
        fontWeight = FontWeight.Bold
    )
}

@Composable
private fun ModelVersionInfo(
    updateState: ModelUpdateState,
    modelInfo: ModelDownloader.ModelInfo
) {
    Text(
        text = "Current Version: ${updateState.getCurrentVersion()}",
        style = MaterialTheme.typography.bodyMedium
    )

    val latestVersionText = when (updateState) {
        is ModelUpdateState.Loading -> "Fetching..."
        is ModelUpdateState.NetworkError -> "Could not fetch"
        else -> updateState.getLatestVersion() ?: "Unknown"
    }

    Text(
        text = "Latest Version: $latestVersionText",
        style = MaterialTheme.typography.bodyMedium
    )

    if (modelInfo.downloadedModelSize != null) {
        Text(
            text = "Downloaded Model: ${modelInfo.downloadedModelSize / 1024 / 1024}MB",
            style = MaterialTheme.typography.bodySmall,
            color = Color.Gray
        )
    }
}

@Composable
private fun ModelUpdateContent(updateState: ModelUpdateState) {
    when (updateState) {
        is ModelUpdateState.Loading -> {
            Text(
                text = "Checking for updates...",
                style = MaterialTheme.typography.bodyMedium,
                textAlign = TextAlign.Center
            )
        }
        is ModelUpdateState.UpToDate -> {
            Text(
                text = "✓ You have the latest model",
                style = MaterialTheme.typography.bodyMedium,
                color = Color(0xFF4CAF50)
            )
        }
        is ModelUpdateState.UpdateAvailable -> {
            Text(
                text = "⬆️ Update available!",
                style = MaterialTheme.typography.bodyMedium,
                color = Color(0xFF2196F3)
            )
        }
        is ModelUpdateState.Downloading -> {
            CircularProgressIndicator()
            Text(
                text = "Downloading model...",
                style = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.padding(top = 8.dp)
            )
        }
        is ModelUpdateState.DownloadSuccess -> {
            Text(
                text = "✓ Model downloaded successfully!\nRestart the app to use the new model.",
                style = MaterialTheme.typography.bodyMedium,
                textAlign = TextAlign.Center,
                color = Color(0xFF4CAF50)
            )
        }
        is ModelUpdateState.DownloadFailed -> {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = "⚠️ Download Failed",
                    style = MaterialTheme.typography.titleMedium,
                    color = Color.Red,
                    fontWeight = FontWeight.Bold
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = updateState.error,
                    style = MaterialTheme.typography.bodyMedium,
                    textAlign = TextAlign.Center,
                    color = Color.Red
                )
            }
        }
        is ModelUpdateState.NetworkError -> {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = "🌐 Connection Issue",
                    style = MaterialTheme.typography.titleMedium,
                    color = Color(0xFFFF9800),
                    fontWeight = FontWeight.Bold
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = updateState.error,
                    style = MaterialTheme.typography.bodyMedium,
                    textAlign = TextAlign.Center,
                    color = Color(0xFFFF9800)
                )
            }
        }
    }
}

@Composable
private fun ModelUpdateActions(
    updateState: ModelUpdateState,
    onDownload: (String) -> Unit,
    onRetry: () -> Unit = {},
    onDismiss: () -> Unit
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceEvenly
    ) {
        // Show download button only when update is available
        if (updateState is ModelUpdateState.UpdateAvailable) {
            Button(
                onClick = { onDownload(updateState.latestVersion) }
            ) {
                Text("Download Update")
            }
        }

        // Show retry button for failed states
        if (updateState is ModelUpdateState.DownloadFailed) {
            Button(
                onClick = { onDownload(updateState.latestVersion) }
            ) {
                Text("Retry Download")
            }
        }

        if (updateState is ModelUpdateState.NetworkError) {
            Button(
                onClick = onRetry
            ) {
                Text("Retry")
            }
        }

        // Show close/restart button based on state
        if (updateState !is ModelUpdateState.Downloading) {
            Button(onClick = onDismiss) {
                Text(
                    when (updateState) {
                        is ModelUpdateState.DownloadSuccess -> "Restart App"
                        else -> "Close"
                    }
                )
            }
        }
    }
}


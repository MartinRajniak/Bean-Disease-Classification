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
import androidx.compose.foundation.layout.width
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
import kotlinx.coroutines.launch

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
    var classificationResult by remember { mutableStateOf<BeansDiseaseClassifier.ClassificationResult?>(null) }
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
                text = "Disease: ${result.label.replace("_", " ").replaceFirstChar { it.uppercase() }}",
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
    val modelDownloader = remember { ModelDownloader(context) }
    val coroutineScope = rememberCoroutineScope()

    var modelInfo by remember { mutableStateOf(modelDownloader.getModelInfo()) }
    var isDownloading by remember { mutableStateOf(false) }
    var downloadStatus by remember { mutableStateOf<String?>(null) }

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
                Text(
                    text = "Model Update",
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.Bold
                )

                Spacer(modifier = Modifier.height(16.dp))

                Text(
                    text = "Current Version: ${modelInfo.currentVersion}",
                    style = MaterialTheme.typography.bodyMedium
                )

                Text(
                    text = "Latest Version: ${modelInfo.latestVersion}",
                    style = MaterialTheme.typography.bodyMedium
                )

                if (modelInfo.downloadedModelSize != null) {
                    Text(
                        text = "Downloaded Model: ${modelInfo.downloadedModelSize!! / 1024 / 1024}MB",
                        style = MaterialTheme.typography.bodySmall,
                        color = Color.Gray
                    )
                }

                Spacer(modifier = Modifier.height(16.dp))

                if (isDownloading) {
                    CircularProgressIndicator()
                    Text(
                        text = "Downloading model...",
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.padding(top = 8.dp)
                    )
                } else if (downloadStatus != null) {
                    Text(
                        text = downloadStatus!!,
                        style = MaterialTheme.typography.bodyMedium,
                        textAlign = TextAlign.Center,
                        color = if (downloadStatus!!.contains("success")) Color(0xFF4CAF50) else Color.Red
                    )
                } else if (modelInfo.isUpdateAvailable) {
                    Text(
                        text = "A new model version is available!",
                        style = MaterialTheme.typography.bodyMedium,
                        textAlign = TextAlign.Center
                    )
                } else {
                    Text(
                        text = "✓ You have the latest model",
                        style = MaterialTheme.typography.bodyMedium,
                        color = Color(0xFF4CAF50)
                    )
                }

                Spacer(modifier = Modifier.height(24.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceEvenly
                ) {
                    if (modelInfo.isUpdateAvailable && !isDownloading) {
                        Button(
                            onClick = {
                                isDownloading = true
                                downloadStatus = null
                                coroutineScope.launch {
                                    val success = modelDownloader.downloadLatestModel()
                                    isDownloading = false
                                    downloadStatus = if (success) {
                                        modelInfo = modelDownloader.getModelInfo()
                                        "✓ Model downloaded successfully!\nRestart the app to use the new model."
                                    } else {
                                        "✗ Download failed. Check your internet connection and try again."
                                    }
                                }
                            }
                        ) {
                            Text("Download Update")
                        }
                    }

                    if (!isDownloading) {
                        Button(onClick = onDismiss) {
                            Text(if (downloadStatus?.contains("success") == true) "Restart App" else "Close")
                        }
                    }
                }
            }
        }
    }
}


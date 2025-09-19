package com.gen.myapplication

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.util.concurrent.Executors
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine

@Composable
fun CameraPreview(
    modifier: Modifier = Modifier,
    onClassificationResult: ((BeansDiseaseClassifier.ClassificationResult?) -> Unit)? = null
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val previewView = remember { PreviewView(context) }
    val classifier = remember { BeansDiseaseClassifier(context) }

    LaunchedEffect(previewView) {
        classifier.initialize()
        val cameraProvider = getCameraProvider(context)
        bindCameraUseCases(cameraProvider, previewView, lifecycleOwner, classifier, onClassificationResult)
    }

    DisposableEffect(Unit) {
        onDispose {
            classifier.close()
        }
    }

    AndroidView(
        factory = { previewView },
        modifier = modifier.fillMaxSize()
    )
}

private suspend fun getCameraProvider(context: Context): ProcessCameraProvider =
    suspendCoroutine { continuation ->
        ProcessCameraProvider.getInstance(context).also { cameraProvider ->
            cameraProvider.addListener({
                continuation.resume(cameraProvider.get())
            }, ContextCompat.getMainExecutor(context))
        }
    }

private fun bindCameraUseCases(
    cameraProvider: ProcessCameraProvider,
    previewView: PreviewView,
    lifecycleOwner: LifecycleOwner,
    classifier: BeansDiseaseClassifier,
    onClassificationResult: ((BeansDiseaseClassifier.ClassificationResult?) -> Unit)?
) {
    val preview = Preview.Builder().build().apply {
        setSurfaceProvider(previewView.surfaceProvider)
    }

    val imageAnalysis = ImageAnalysis.Builder()
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .build()

    val executor = Executors.newSingleThreadExecutor()

    // Add throttling to prevent inference queue buildup
    var isProcessing = false

    imageAnalysis.setAnalyzer(executor) { imageProxy ->
        if (!isProcessing) {
            isProcessing = true
            processImage(imageProxy, classifier, onClassificationResult) {
                isProcessing = false
            }
        } else {
            imageProxy.close() // Skip this frame if still processing
        }
    }

    try {
        cameraProvider.unbindAll()
        cameraProvider.bindToLifecycle(
            lifecycleOwner,
            CameraSelector.DEFAULT_BACK_CAMERA,
            preview,
            imageAnalysis
        )
    } catch (exc: Exception) {
        // Handle camera binding errors
    }
}

private fun processImage(
    imageProxy: ImageProxy,
    classifier: BeansDiseaseClassifier,
    onResult: ((BeansDiseaseClassifier.ClassificationResult?) -> Unit)?,
    onComplete: () -> Unit
) {
    try {
        val bitmap = imageProxyToBitmap(imageProxy)
        val result = classifier.classifyImage(bitmap)
        onResult?.invoke(result)
    } catch (e: Exception) {
        onResult?.invoke(null)
    } finally {
        imageProxy.close()
        onComplete()
    }
}

private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
    return when (imageProxy.format) {
        ImageFormat.YUV_420_888 -> {
            // Use faster YUV conversion with minimal compression for better performance
            val yBuffer = imageProxy.planes[0].buffer
            val uBuffer = imageProxy.planes[1].buffer
            val vBuffer = imageProxy.planes[2].buffer

            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()

            val nv21 = ByteArray(ySize + uSize + vSize)

            yBuffer.get(nv21, 0, ySize)
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)

            val yuvImage = YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
            val out = ByteArrayOutputStream()
            // Use 95% quality instead of 100% for faster processing
            yuvImage.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), 95, out)
            val imageBytes = out.toByteArray()
            BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
        }
        else -> {
            val buffer: ByteBuffer = imageProxy.planes[0].buffer
            val bytes = ByteArray(buffer.remaining())
            buffer.get(bytes)
            BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        }
    }
}
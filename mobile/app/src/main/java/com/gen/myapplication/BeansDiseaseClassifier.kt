package com.gen.myapplication

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.flex.FlexDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.IOException
import java.nio.MappedByteBuffer

class BeansDiseaseClassifier(private val context: Context) {

    private var interpreter: Interpreter? = null
    private var imageProcessor: ImageProcessor? = null
    private var labels: List<String> = emptyList()

    companion object {
        private const val TAG = "BeansDiseaseClassifier"
        private const val MODEL_FILENAME = "beans_disease_classification_transfer_learning.tflite"
        private const val LABELS_FILENAME = "labels.txt"
        private const val INPUT_SIZE = 224 // Model expects 224x224 input as per preprocessing
    }

    fun initialize(): Boolean {
        return try {
            val model = loadModelFile()

            val flexDelegate = FlexDelegate()
            val options = Interpreter.Options()
            options.addDelegate(flexDelegate)

            interpreter = Interpreter(model, options)

            imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(INPUT_SIZE, INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
                .build()

            loadLabels()

            Log.d(TAG, "Model initialized successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing model", e)
            false
        }
    }

    private fun loadModelFile(): MappedByteBuffer {
        return FileUtil.loadMappedFile(context, MODEL_FILENAME)
    }

    private fun loadLabels() {
        try {
            labels = FileUtil.loadLabels(context, LABELS_FILENAME)
            Log.d(TAG, "Loaded ${labels.size} labels")
        } catch (e: IOException) {
            Log.w(TAG, "Labels file not found, using default labels")
            // Default labels for bean diseases (common classifications)
            labels = listOf(
                "angular_leaf_spot",
                "bean_rust",
                "healthy"
            )
        }
    }

    fun classifyImage(bitmap: Bitmap): ClassificationResult? {
        val interpreter = this.interpreter ?: return null
        val imageProcessor = this.imageProcessor ?: return null

        return try {
            val startTime = System.currentTimeMillis()
            Log.d(TAG, "🚀 Starting classification for image ${bitmap.width}x${bitmap.height}")

            // Step 1: Convert to TensorImage
            val step1Start = System.currentTimeMillis()
            val tensorImage = TensorImage.fromBitmap(bitmap)
            val step1End = System.currentTimeMillis()
            Log.d(TAG, "⏱️ Step 1 - TensorImage creation: ${step1End - step1Start}ms")

            // Step 2: Resize image
            val step2Start = System.currentTimeMillis()
            val resizedImage = imageProcessor.process(tensorImage)
            val resizedBitmap = resizedImage.bitmap
            val step2End = System.currentTimeMillis()
            Log.d(TAG, "⏱️ Step 2 - Image resize: ${step2End - step2Start}ms")

            // Step 3: Create input array and extract pixels
            val step3Start = System.currentTimeMillis()
            val input = Array(1) { Array(INPUT_SIZE) { Array(INPUT_SIZE) { FloatArray(3) } } }
            val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
            resizedBitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
            val step3End = System.currentTimeMillis()
            Log.d(TAG, "⏱️ Step 3 - Array creation & pixel extraction: ${step3End - step3Start}ms")

            // Step 4: Preprocessing (pixel normalization)
            val step4Start = System.currentTimeMillis()
            for (i in 0 until INPUT_SIZE) {
                for (j in 0 until INPUT_SIZE) {
                    val pixelValue = pixels[i * INPUT_SIZE + j]

                    // Extract RGB values (Android bitmap is ARGB format)
                    val red = (pixelValue shr 16) and 0xFF
                    val green = (pixelValue shr 8) and 0xFF
                    val blue = pixelValue and 0xFF

                    // Apply Xception preprocessing exactly as in Python: (pixel/127.5) - 1.0
                    input[0][i][j][0] = (red / 127.5f) - 1.0f     // Red channel
                    input[0][i][j][1] = (green / 127.5f) - 1.0f   // Green channel
                    input[0][i][j][2] = (blue / 127.5f) - 1.0f    // Blue channel
                }
            }
            val step4End = System.currentTimeMillis()
            Log.d(TAG, "⏱️ Step 4 - Pixel preprocessing: ${step4End - step4Start}ms")

            // Step 5: Model inference
            val step5Start = System.currentTimeMillis()
            val outputArray = Array(1) { FloatArray(labels.size) }
            interpreter.run(input, outputArray)
            val step5End = System.currentTimeMillis()
            Log.d(TAG, "⏱️ Step 5 - Model inference: ${step5End - step5Start}ms")

            // Process results
            val results = mutableMapOf<String, Float>()
            for (i in labels.indices) {
                results[labels[i]] = outputArray[0][i]
            }

            // Find the top result
            val topResult = results.maxByOrNull { it.value }

            topResult?.let {
                Log.d(TAG, "Classification result: ${it.key} with confidence ${it.value}")
                ClassificationResult(
                    label = it.key,
                    confidence = it.value,
                    allResults = results
                )
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error during classification", e)
            null
        }
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }


    data class ClassificationResult(
        val label: String,
        val confidence: Float,
        val allResults: Map<String, Float>
    )
}
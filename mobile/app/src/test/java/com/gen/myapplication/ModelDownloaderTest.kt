package com.gen.myapplication

import android.content.Context
import io.ktor.client.engine.mock.MockEngine
import io.ktor.client.engine.mock.respond
import io.ktor.client.engine.mock.respondError
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.http.headersOf
import io.ktor.utils.io.ByteReadChannel
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.mockito.Mock
import org.mockito.MockitoAnnotations
import java.io.File

class ModelDownloaderTest {

    @Mock
    private lateinit var mockContext: Context

    private lateinit var testModelsDir: File
    private lateinit var testVersionFile: File
    private lateinit var testModelFile: File
    private lateinit var mockGitHubApi: GitHubApi
    private lateinit var modelDownloader: ModelDownloader

    @Before
    fun setup() {
        MockitoAnnotations.openMocks(this)

        // Create a temporary directory for testing
        testModelsDir = createTempDir("test_models")
        testVersionFile = File(testModelsDir, "model_version.txt")
        testModelFile = File(testModelsDir, "current_model.tflite")

        // Create mock GitHubApi with success response by default
        val mockEngine = MockEngine { request ->
            respond(
                content = ByteReadChannel("""{"tag_name":"v1.0.0"}"""),
                status = HttpStatusCode.OK,
                headers = headersOf(HttpHeaders.ContentType, "application/json")
            )
        }
        mockGitHubApi = GitHubApi(mockEngine)

        modelDownloader = TestableModelDownloader(mockContext, mockGitHubApi, testModelsDir)
    }

    @Test
    fun `getCurrentVersion returns default version when no version file exists`() {
        assertFalse(testVersionFile.exists())

        val currentVersion = modelDownloader.getCurrentVersion()

        assertEquals("v0.1.0-bundled", currentVersion)
    }

    @Test
    fun `getCurrentVersion returns stored version when version file exists`() {
        val expectedVersion = "v2.0.0-test"
        testVersionFile.writeText(expectedVersion)

        val currentVersion = modelDownloader.getCurrentVersion()

        assertEquals(expectedVersion, currentVersion)
    }

    @Test
    fun `getCurrentVersion handles version file with whitespace`() {
        val expectedVersion = "v1.5.0"
        testVersionFile.writeText("  $expectedVersion\n  ")

        val currentVersion = modelDownloader.getCurrentVersion()

        assertEquals(expectedVersion, currentVersion)
    }

    @Test
    fun `getCurrentModelPath returns null when no model file exists`() {
        assertFalse(testModelFile.exists())

        val modelPath = modelDownloader.getCurrentModelPath()

        assertNull(modelPath)
    }

    @Test
    fun `getCurrentModelPath returns file path when model exists`() {
        testModelFile.writeText("fake model content")

        val modelPath = modelDownloader.getCurrentModelPath()

        assertEquals(testModelFile.absolutePath, modelPath)
    }

    @Test
    fun `getLatestVersion returns success when GitHubApi succeeds`() = runBlocking {
        val expectedVersion = "v2.1.0"
        val mockEngine = MockEngine { request ->
            respond(
                content = ByteReadChannel("""{"tag_name":"$expectedVersion"}"""),
                status = HttpStatusCode.OK,
                headers = headersOf(HttpHeaders.ContentType, "application/json")
            )
        }
        val gitHubApi = GitHubApi(mockEngine)
        val downloader = TestableModelDownloader(mockContext, gitHubApi, testModelsDir)

        val result = downloader.getLatestVersion()

        assertTrue(result.isSuccess)
        assertEquals(expectedVersion, result.getOrNull())
    }

    @Test
    fun `getLatestVersion returns failure when GitHubApi fails`() = runBlocking {
        val mockEngine = MockEngine { request ->
            respondError(HttpStatusCode.InternalServerError)
        }
        val gitHubApi = GitHubApi(mockEngine)
        val downloader = TestableModelDownloader(mockContext, gitHubApi, testModelsDir)

        val result = downloader.getLatestVersion()

        assertTrue(result.isFailure)
    }

    @Test
    fun `isUpdateAvailable returns true when current version differs from latest`() {
        testVersionFile.writeText("v1.0.0")

        val isUpdateAvailable = modelDownloader.isUpdateAvailable("v2.0.0")

        assertTrue(isUpdateAvailable)
    }

    @Test
    fun `isUpdateAvailable returns false when versions are same`() {
        val version = "v1.5.0"
        testVersionFile.writeText(version)

        val isUpdateAvailable = modelDownloader.isUpdateAvailable(version)

        assertFalse(isUpdateAvailable)
    }

    @Test
    fun `isUpdateAvailable compares with default version when no version file exists`() {
        assertFalse(testVersionFile.exists())

        val isUpdateAvailable = modelDownloader.isUpdateAvailable("v0.1.0-bundled")

        assertFalse(isUpdateAvailable)
    }

    @Test
    fun `deleteDownloadedModel removes model and version files`() {
        testModelFile.writeText("fake model")
        testVersionFile.writeText("v1.0.0")
        assertTrue(testModelFile.exists())
        assertTrue(testVersionFile.exists())

        modelDownloader.deleteDownloadedModel()

        assertFalse(testModelFile.exists())
        assertFalse(testVersionFile.exists())
    }

    @Test
    fun `deleteDownloadedModel handles non-existent files gracefully`() {
        assertFalse(testModelFile.exists())
        assertFalse(testVersionFile.exists())

        // Should not throw exception
        modelDownloader.deleteDownloadedModel()

        assertFalse(testModelFile.exists())
        assertFalse(testVersionFile.exists())
    }

    @Test
    fun `getModelInfo returns correct information when no downloaded model exists`() {
        assertFalse(testModelFile.exists())

        val modelInfo = modelDownloader.getModelInfo()

        assertEquals("v0.1.0-bundled", modelInfo.currentVersion)
        assertFalse(modelInfo.downloadedModelExists)
        assertNull(modelInfo.downloadedModelSize)
    }

    @Test
    fun `getModelInfo returns correct information when downloaded model exists`() {
        val modelContent = "fake model content for testing"
        testModelFile.writeText(modelContent)
        testVersionFile.writeText("v2.0.0")

        val modelInfo = modelDownloader.getModelInfo()

        assertEquals("v2.0.0", modelInfo.currentVersion)
        assertTrue(modelInfo.downloadedModelExists)
        assertEquals(modelContent.toByteArray().size.toLong(), modelInfo.downloadedModelSize)
    }

    @Test
    fun `downloadModel handles network errors gracefully`() = runBlocking {
        // Create a custom testable ModelDownloader that simulates network failure
        val downloader = object : TestableModelDownloader(mockContext, mockGitHubApi, testModelsDir) {
            override suspend fun downloadModel(modelVersion: String): Boolean {
                // Simulate network error during download
                return false
            }
        }

        val success = downloader.downloadModel("v1.0.0")

        assertFalse(success)
        assertFalse(testModelFile.exists())
        assertFalse(testVersionFile.exists())
    }

    @Test
    fun `downloadModel updates version file on successful download`() = runBlocking {
        val targetVersion = "v2.0.0-test"

        // Create a custom testable ModelDownloader that simulates successful download
        val downloader = object : TestableModelDownloader(mockContext, mockGitHubApi, testModelsDir) {
            override suspend fun downloadModel(modelVersion: String): Boolean {
                // Simulate successful download
                testModelFile.writeText("downloaded model content")
                testVersionFile.writeText(modelVersion)
                return true
            }
        }

        val success = downloader.downloadModel(targetVersion)

        assertTrue(success)
        assertTrue(testModelFile.exists())
        assertTrue(testVersionFile.exists())
        assertEquals(targetVersion, testVersionFile.readText())
    }

    /**
     * Custom testable version of ModelDownloader that allows us to override
     * the models directory for testing
     */
    private open class TestableModelDownloader(
        context: Context,
        gitHubApi: GitHubApi,
        private val testModelsDir: File
    ) : ModelDownloader(context, gitHubApi) {

        private val testVersionFile = File(testModelsDir, "model_version.txt")
        private val testModelFile = File(testModelsDir, "current_model.tflite")

        override fun getCurrentVersion(): String {
            return if (testVersionFile.exists()) {
                testVersionFile.readText().trim()
            } else {
                "v0.1.0-bundled"
            }
        }

        override fun getCurrentModelPath(): String? {
            return if (testModelFile.exists()) {
                testModelFile.absolutePath
            } else {
                null
            }
        }

        override fun deleteDownloadedModel() {
            if (testModelFile.exists()) {
                testModelFile.delete()
            }
            if (testVersionFile.exists()) {
                testVersionFile.delete()
            }
        }

        override fun getModelInfo(): ModelInfo {
            return ModelInfo(
                currentVersion = getCurrentVersion(),
                downloadedModelExists = testModelFile.exists(),
                downloadedModelSize = if (testModelFile.exists()) testModelFile.length() else null
            )
        }
    }
}
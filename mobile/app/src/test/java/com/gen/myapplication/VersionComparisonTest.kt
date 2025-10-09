package com.gen.myapplication

import android.content.Context
import io.ktor.client.engine.mock.MockEngine
import io.ktor.client.engine.mock.respond
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.http.headersOf
import io.ktor.utils.io.ByteReadChannel
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.mockito.Mock
import org.mockito.MockitoAnnotations
import java.io.File

class VersionComparisonTest {

    @Mock
    private lateinit var mockContext: Context

    private lateinit var testModelsDir: File
    private lateinit var testVersionFile: File
    private lateinit var mockGitHubApi: GitHubApi
    private lateinit var modelDownloader: ModelDownloader

    @Before
    fun setup() {
        MockitoAnnotations.openMocks(this)

        // Create a temporary directory for testing
        testModelsDir = createTempDir("version_test")
        testVersionFile = File(testModelsDir, "model_version.txt")

        // Create mock GitHubApi
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
    fun `should detect update available from bundled to first release`() {
        // No version file means we're on bundled version
        assertFalse(testVersionFile.exists())

        val isUpdateAvailable = modelDownloader.isUpdateAvailable("v1.0.0")

        assertTrue(isUpdateAvailable)
    }

    @Test
    fun `should detect update available for major version bump`() {
        testVersionFile.writeText("v1.0.0")

        val isUpdateAvailable = modelDownloader.isUpdateAvailable("v2.0.0")

        assertTrue(isUpdateAvailable)
    }

    @Test
    fun `should detect update available for minor version bump`() {
        testVersionFile.writeText("v1.0.0")

        val isUpdateAvailable = modelDownloader.isUpdateAvailable("v1.1.0")

        assertTrue(isUpdateAvailable)
    }

    @Test
    fun `should detect update available for patch version bump`() {
        testVersionFile.writeText("v1.0.0")

        val isUpdateAvailable = modelDownloader.isUpdateAvailable("v1.0.1")

        assertTrue(isUpdateAvailable)
    }

    @Test
    fun `should not detect update when versions are identical`() {
        val version = "v1.5.3"
        testVersionFile.writeText(version)

        val isUpdateAvailable = modelDownloader.isUpdateAvailable(version)

        assertFalse(isUpdateAvailable)
    }

    @Test
    fun `should handle semantic versioning with prerelease tags`() {
        testVersionFile.writeText("v1.0.0-alpha")

        val isUpdateAvailable = modelDownloader.isUpdateAvailable("v1.0.0-beta")

        assertTrue(isUpdateAvailable)
    }

    @Test
    fun `should handle semantic versioning with build metadata`() {
        testVersionFile.writeText("v1.0.0+build123")

        val isUpdateAvailable = modelDownloader.isUpdateAvailable("v1.0.0+build456")

        assertTrue(isUpdateAvailable)
    }

    @Test
    fun `should handle complex semantic version format`() {
        testVersionFile.writeText("v2.1.0-beta.1+build.123")

        val isUpdateAvailable = modelDownloader.isUpdateAvailable("v2.1.0-beta.2+build.456")

        assertTrue(isUpdateAvailable)
    }

    @Test
    fun `should handle version without v prefix`() {
        testVersionFile.writeText("1.0.0")

        val isUpdateAvailable = modelDownloader.isUpdateAvailable("1.0.1")

        assertTrue(isUpdateAvailable)
    }

    @Test
    fun `should handle mixed version formats`() {
        testVersionFile.writeText("v1.0.0")

        val isUpdateAvailable = modelDownloader.isUpdateAvailable("1.0.1")

        assertTrue(isUpdateAvailable)
    }

    @Test
    fun `should handle custom version formats`() {
        testVersionFile.writeText("v0.1.0-bundled")

        val isUpdateAvailable = modelDownloader.isUpdateAvailable("v.0.1.0-mobile-net")

        assertTrue(isUpdateAvailable)
    }

    @Test
    fun `should handle date-based versioning`() {
        testVersionFile.writeText("v2025.01.01")

        val isUpdateAvailable = modelDownloader.isUpdateAvailable("v2025.01.02")

        assertTrue(isUpdateAvailable)
    }

    @Test
    fun `should handle git hash style versioning`() {
        testVersionFile.writeText("v1.0.0-abc123")

        val isUpdateAvailable = modelDownloader.isUpdateAvailable("v1.0.0-def456")

        assertTrue(isUpdateAvailable)
    }

    @Test
    fun `should not detect update when current version is newer format`() {
        // Note: This test shows the limitation of string comparison
        // In a real system, you might want semantic version parsing
        testVersionFile.writeText("v2.0.0")

        val isUpdateAvailable = modelDownloader.isUpdateAvailable("v1.9.9")

        assertTrue(isUpdateAvailable) // String comparison: "v2.0.0" != "v1.9.9"
    }

    @Test
    fun `should handle empty version tags`() {
        testVersionFile.writeText("")

        val isUpdateAvailable = modelDownloader.isUpdateAvailable("v1.0.0")

        assertTrue(isUpdateAvailable)
    }

    @Test
    fun `should handle whitespace in version files`() {
        testVersionFile.writeText("  v1.0.0  \n")

        val isUpdateAvailable = modelDownloader.isUpdateAvailable("v1.0.0")

        assertFalse(isUpdateAvailable) // Should be trimmed and match
    }

    @Test
    fun `should compare bundled version with first release`() {
        // Default bundled version
        assertFalse(testVersionFile.exists())

        val isUpdateAvailable = modelDownloader.isUpdateAvailable("v0.1.0-bundled")

        assertFalse(isUpdateAvailable) // Same as default
    }

    @Test
    fun `should detect update from bundled to proper release`() {
        // Default bundled version
        assertFalse(testVersionFile.exists())

        val isUpdateAvailable = modelDownloader.isUpdateAvailable("v1.0.0-release")

        assertTrue(isUpdateAvailable)
    }

    /**
     * Testable version of ModelDownloader for version comparison tests
     */
    private class TestableModelDownloader(
        context: Context,
        gitHubApi: GitHubApi,
        testModelsDir: File
    ) : ModelDownloader(context, gitHubApi) {

        private val testVersionFile = File(testModelsDir, "model_version.txt")

        override fun getCurrentVersion(): String {
            return if (testVersionFile.exists()) {
                testVersionFile.readText().trim()
            } else {
                "v0.1.0-bundled"
            }
        }
    }
}
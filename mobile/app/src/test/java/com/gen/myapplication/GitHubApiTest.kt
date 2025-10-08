package com.gen.myapplication

import io.ktor.client.engine.mock.MockEngine
import io.ktor.client.engine.mock.respond
import io.ktor.client.engine.mock.respondError
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.http.headersOf
import io.ktor.utils.io.ByteReadChannel
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class GitHubApiTest {

    @Test
    fun `should return success with tag name when API returns valid response`() = runBlocking {
        val expected = "v1.0.0"
        val mockEngine = MockEngine { request ->
            respond(
                content = ByteReadChannel("""{"tag_name":"$expected"}"""),
                status = HttpStatusCode.OK,
                headers = headersOf(HttpHeaders.ContentType, "application/json")
            )
        }

        val gitHubApi = GitHubApi(mockEngine)
        val result = gitHubApi.fetchLatestVersionInfo()

        assertTrue(result.isSuccess)
        assertEquals(expected, result.getOrNull())
    }

    @Test
    fun `should handle complex version tag names`() = runBlocking {
        val complexTag = "v2.1.0-beta.3+build.456"
        val mockEngine = MockEngine { request ->
            respond(
                content = ByteReadChannel("""{"tag_name":"$complexTag"}"""),
                status = HttpStatusCode.OK,
                headers = headersOf(HttpHeaders.ContentType, "application/json")
            )
        }

        val gitHubApi = GitHubApi(mockEngine)
        val result = gitHubApi.fetchLatestVersionInfo()

        assertTrue(result.isSuccess)
        assertEquals(complexTag, result.getOrNull())
    }

    @Test
    fun `should ignore extra fields in JSON response`() = runBlocking {
        val expectedTag = "v1.5.0"
        val jsonWithExtraFields = """
            {
                "tag_name": "$expectedTag",
                "name": "Release 1.5.0",
                "draft": false,
                "prerelease": false,
                "created_at": "2025-01-01T00:00:00Z",
                "published_at": "2025-01-01T12:00:00Z",
                "assets": [],
                "body": "Release notes here"
            }
        """.trimIndent()

        val mockEngine = MockEngine { request ->
            respond(
                content = ByteReadChannel(jsonWithExtraFields),
                status = HttpStatusCode.OK,
                headers = headersOf(HttpHeaders.ContentType, "application/json")
            )
        }

        val gitHubApi = GitHubApi(mockEngine)
        val result = gitHubApi.fetchLatestVersionInfo()

        assertTrue(result.isSuccess)
        assertEquals(expectedTag, result.getOrNull())
    }

    @Test
    fun `should return failure when API returns 404`() = runBlocking {
        val mockEngine = MockEngine { request ->
            respondError(HttpStatusCode.NotFound)
        }

        val gitHubApi = GitHubApi(mockEngine)
        val result = gitHubApi.fetchLatestVersionInfo()

        assertTrue(result.isFailure)
    }

    @Test
    fun `should return failure when API returns 500`() = runBlocking {
        val mockEngine = MockEngine { request ->
            respondError(HttpStatusCode.InternalServerError)
        }

        val gitHubApi = GitHubApi(mockEngine)
        val result = gitHubApi.fetchLatestVersionInfo()

        assertTrue(result.isFailure)
    }

    @Test
    fun `should return failure when API returns 403 rate limit`() = runBlocking {
        val mockEngine = MockEngine { request ->
            respondError(HttpStatusCode.Forbidden)
        }

        val gitHubApi = GitHubApi(mockEngine)
        val result = gitHubApi.fetchLatestVersionInfo()

        assertTrue(result.isFailure)
    }

    @Test
    fun `should return failure when JSON is malformed`() = runBlocking {
        val mockEngine = MockEngine { request ->
            respond(
                content = ByteReadChannel("""{"tag_name": "v1.0.0""""), // Missing closing brace
                status = HttpStatusCode.OK,
                headers = headersOf(HttpHeaders.ContentType, "application/json")
            )
        }

        val gitHubApi = GitHubApi(mockEngine)
        val result = gitHubApi.fetchLatestVersionInfo()

        assertTrue(result.isFailure)
    }

    @Test
    fun `should return failure when tag_name field is missing`() = runBlocking {
        val mockEngine = MockEngine { request ->
            respond(
                content = ByteReadChannel("""{"name": "Release 1.0.0"}"""),
                status = HttpStatusCode.OK,
                headers = headersOf(HttpHeaders.ContentType, "application/json")
            )
        }

        val gitHubApi = GitHubApi(mockEngine)
        val result = gitHubApi.fetchLatestVersionInfo()

        assertTrue(result.isFailure)
    }

    @Test
    fun `should return failure when response body is empty`() = runBlocking {
        val mockEngine = MockEngine { request ->
            respond(
                content = ByteReadChannel(""),
                status = HttpStatusCode.OK,
                headers = headersOf(HttpHeaders.ContentType, "application/json")
            )
        }

        val gitHubApi = GitHubApi(mockEngine)
        val result = gitHubApi.fetchLatestVersionInfo()

        assertTrue(result.isFailure)
    }

    @Test
    fun `should return failure when network error occurs`() = runBlocking {
        val mockEngine = MockEngine { request ->
            throw Exception("Network error")
        }

        val gitHubApi = GitHubApi(mockEngine)
        val result = gitHubApi.fetchLatestVersionInfo()

        assertTrue(result.isFailure)
        val exception = result.exceptionOrNull()
        assertTrue("Should return GitHubApiError", exception is GitHubApiError)
        assertTrue("Should be NetworkUnavailable error", exception is GitHubApiError.NetworkUnavailable)
        assertTrue("Should contain user-friendly message",
            exception?.message?.contains("internet connection") == true)
    }

    @Test
    fun `should request correct GitHub API endpoint`() = runBlocking {
        var requestedUrl = ""
        val mockEngine = MockEngine { request ->
            requestedUrl = request.url.toString()
            respond(
                content = ByteReadChannel("""{"tag_name":"v1.0.0"}"""),
                status = HttpStatusCode.OK,
                headers = headersOf(HttpHeaders.ContentType, "application/json")
            )
        }

        val gitHubApi = GitHubApi(mockEngine)
        gitHubApi.fetchLatestVersionInfo()

        assertTrue(
            requestedUrl.contains("api.github.com/repos/MartinRajniak/Bean-Disease-Classification/releases/latest")
        )
    }

    @Test
    fun `should handle empty tag name`() = runBlocking {
        val mockEngine = MockEngine { request ->
            respond(
                content = ByteReadChannel("""{"tag_name":""}"""),
                status = HttpStatusCode.OK,
                headers = headersOf(HttpHeaders.ContentType, "application/json")
            )
        }

        val gitHubApi = GitHubApi(mockEngine)
        val result = gitHubApi.fetchLatestVersionInfo()

        assertTrue(result.isSuccess)
        assertEquals("", result.getOrNull())
    }

    @Test
    fun `should return user-friendly error for general exceptions`() = runBlocking {
        val mockEngine = MockEngine { request ->
            throw IllegalStateException("Some internal error")
        }

        val gitHubApi = GitHubApi(mockEngine)
        val result = gitHubApi.fetchLatestVersionInfo()

        assertTrue(result.isFailure)
        val actualException = result.exceptionOrNull()
        assertTrue("Should return GitHubApiError", actualException is GitHubApiError)
        assertTrue("Should contain user-friendly message",
            actualException?.message?.isNotEmpty() == true)
        // Should preserve technical details for debugging
        assertTrue("Should preserve original cause", actualException?.cause != null)
    }
}
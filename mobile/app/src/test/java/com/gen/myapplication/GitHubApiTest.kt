package com.gen.myapplication

import io.ktor.client.engine.mock.MockEngine
import io.ktor.client.engine.mock.respond
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.http.headersOf
import io.ktor.utils.io.ByteReadChannel
import kotlinx.coroutines.runBlocking
import org.junit.Assert
import org.junit.Test

class GitHubApiTest {

    @Test
    fun `happy case`() = runBlocking {
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

        Assert.assertTrue(result.isSuccess)
        Assert.assertEquals(expected, result.getOrNull())
    }

}
package com.gen.myapplication

import android.util.Log
import io.ktor.client.HttpClient
import io.ktor.client.call.body
import io.ktor.client.engine.HttpClientEngine
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.client.request.get
import io.ktor.serialization.kotlinx.json.json
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

class GitHubApi(
    engine: HttpClientEngine
) {
    private val client = HttpClient(engine) {
        install(ContentNegotiation) {
            json(
                Json {
                    ignoreUnknownKeys = true
                }
            )
        }
    }

    suspend fun fetchLatestVersionInfo(): Result<String> {
        try {
            val response = client.get(LATEST_VERSION_URL).body<GitHubRepoRelease>()
            return Result.success(response.tag)
        } catch (e: Exception) {
            Log.e("GitHubApi", "Failed to fetch latest version info", e)
            return Result.failure(e)
        }
    }

    companion object {
        private const val LATEST_VERSION_URL =
            "https://api.github.com/repos/MartinRajniak/Bean-Disease-Classification/releases/latest"
    }
}

@Serializable
data class GitHubRepoRelease(
    @SerialName("tag_name")
    val tag: String
)
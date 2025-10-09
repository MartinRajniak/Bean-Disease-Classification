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
import kotlinx.serialization.SerializationException
import kotlinx.serialization.json.Json
import java.io.IOException
import java.net.ConnectException
import java.net.SocketTimeoutException
import java.net.UnknownHostException

/**
 * User-friendly error types for GitHub API failures
 */
sealed class GitHubApiError(message: String, cause: Throwable? = null) : Exception(message, cause) {
    class NetworkUnavailable(cause: Throwable? = null) : GitHubApiError(
        "No internet connection available. Please check your network and try again.",
        cause
    )

    class ServerUnavailable(cause: Throwable? = null) : GitHubApiError(
        "GitHub servers are currently unavailable. Please try again later.",
        cause
    )

    class RateLimitExceeded(cause: Throwable? = null) : GitHubApiError(
        "Too many requests. Please wait a few minutes before trying again.",
        cause
    )

    class RepositoryNotFound(cause: Throwable? = null) : GitHubApiError(
        "Model repository not found. The app may need to be updated.",
        cause
    )

    class InvalidResponse(cause: Throwable? = null) : GitHubApiError(
        "Received invalid data from server. Please try again.",
        cause
    )

    class RequestTimeout(cause: Throwable? = null) : GitHubApiError(
        "Request timed out. Please check your connection and try again.",
        cause
    )

    class Unknown(cause: Throwable? = null) : GitHubApiError(
        "An unexpected error occurred. Please try again later.",
        cause
    )
}

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
            val userFriendlyError = mapToUserFriendlyError(e)
            Log.e("GitHubApi", "Failed to fetch latest version info: ${userFriendlyError.message}", e)
            return Result.failure(userFriendlyError)
        }
    }

    /**
     * Maps technical exceptions to user-friendly error messages
     */
    private fun mapToUserFriendlyError(exception: Exception): GitHubApiError {
        return when (exception) {
            is UnknownHostException, is ConnectException -> {
                GitHubApiError.NetworkUnavailable(exception)
            }
            is SocketTimeoutException -> {
                GitHubApiError.RequestTimeout(exception)
            }
            is SerializationException -> {
                GitHubApiError.InvalidResponse(exception)
            }
            is IOException -> {
                // Check if it's an HTTP error by looking at the message
                when {
                    exception.message?.contains("404") == true -> GitHubApiError.RepositoryNotFound(exception)
                    exception.message?.contains("403") == true -> GitHubApiError.RateLimitExceeded(exception)
                    exception.message?.contains("5") == true -> GitHubApiError.ServerUnavailable(exception)
                    else -> GitHubApiError.NetworkUnavailable(exception)
                }
            }
            else -> {
                // Check if the exception message contains HTTP status indicators
                val message = exception.message?.lowercase() ?: ""
                when {
                    message.contains("timeout") -> GitHubApiError.RequestTimeout(exception)
                    message.contains("network") || message.contains("connection") -> GitHubApiError.NetworkUnavailable(exception)
                    message.contains("404") || message.contains("not found") -> GitHubApiError.RepositoryNotFound(exception)
                    message.contains("403") || message.contains("rate limit") -> GitHubApiError.RateLimitExceeded(exception)
                    message.contains("50") || message.contains("server") -> GitHubApiError.ServerUnavailable(exception)
                    else -> GitHubApiError.Unknown(exception)
                }
            }
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
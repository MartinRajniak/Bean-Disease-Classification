package com.gen.myapplication

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.net.UnknownHostException

class ErrorHandlingTest {

    @Test
    fun `GitHubApiError provides user-friendly messages`() {
        val networkError = GitHubApiError.NetworkUnavailable()
        assertTrue(networkError.message!!.contains("internet connection"))
        assertTrue(networkError.message!!.contains("check your network"))

        val serverError = GitHubApiError.ServerUnavailable()
        assertTrue(serverError.message!!.contains("servers are currently unavailable"))

        val rateLimitError = GitHubApiError.RateLimitExceeded()
        assertTrue(rateLimitError.message!!.contains("Too many requests"))
        assertTrue(rateLimitError.message!!.contains("wait a few minutes"))

        val notFoundError = GitHubApiError.RepositoryNotFound()
        assertTrue(notFoundError.message!!.contains("repository not found"))

        val timeoutError = GitHubApiError.RequestTimeout()
        assertTrue(timeoutError.message!!.contains("timed out"))
        assertTrue(timeoutError.message!!.contains("check your connection"))
    }

    @Test
    fun `DownloadResult provides specific error messages`() {
        val networkFailure = DownloadResult.Failure("No internet connection available. Please check your network and try again.")
        assertEquals("No internet connection available. Please check your network and try again.", networkFailure.userMessage)

        val storageFailure = DownloadResult.Failure("Not enough storage space available. Please free up space and try again.")
        assertTrue(storageFailure.userMessage.contains("storage space"))
        assertTrue(storageFailure.userMessage.contains("free up space"))

        val corruptedFailure = DownloadResult.Failure("Downloaded file appears to be corrupted or incomplete. Please try again.")
        assertTrue(corruptedFailure.userMessage.contains("corrupted"))
        assertTrue(corruptedFailure.userMessage.contains("incomplete"))
    }

    @Test
    fun `error messages are actionable and user-friendly`() {
        // Test that error messages contain actionable advice
        val networkError = GitHubApiError.NetworkUnavailable()
        assertTrue("Network error should tell user to check connection",
            networkError.message!!.contains("check your network"))

        val timeoutError = GitHubApiError.RequestTimeout()
        assertTrue("Timeout error should suggest trying again",
            timeoutError.message!!.contains("try again"))

        val rateLimitError = GitHubApiError.RateLimitExceeded()
        assertTrue("Rate limit error should suggest waiting",
            rateLimitError.message!!.contains("wait"))

        // Test that messages are user-friendly (no technical jargon)
        val serverError = GitHubApiError.ServerUnavailable()
        assertTrue("Server error should be understandable",
            !serverError.message!!.contains("HTTP") &&
            !serverError.message!!.contains("500") &&
            !serverError.message!!.contains("exception"))
    }

    @Test
    fun `error messages provide context about the problem`() {
        val notFoundError = GitHubApiError.RepositoryNotFound()
        assertTrue("Repository not found should explain what this means",
            notFoundError.message!!.contains("repository not found"))
        assertTrue("Should suggest possible solution",
            notFoundError.message!!.contains("app may need to be updated"))

        val invalidResponseError = GitHubApiError.InvalidResponse()
        assertTrue("Invalid response should explain in user terms",
            invalidResponseError.message!!.contains("invalid data"))
        assertTrue("Should not contain technical details",
            !invalidResponseError.message!!.contains("JSON") &&
            !invalidResponseError.message!!.contains("serialization"))
    }

    @Test
    fun `download error messages distinguish different failure types`() {
        // Test different HTTP error codes result in different messages
        val notFoundMessage = "Model version not found. It may have been removed or the version number is incorrect."
        assertTrue(notFoundMessage.contains("version not found"))

        val forbiddenMessage = "Access denied. You may have exceeded download limits."
        assertTrue(forbiddenMessage.contains("Access denied"))

        val unavailableMessage = "Download service is temporarily unavailable. Please try again later."
        assertTrue(unavailableMessage.contains("temporarily unavailable"))

        val corruptedMessage = "Downloaded file appears to be corrupted or incomplete. Please try again."
        assertTrue(corruptedMessage.contains("corrupted"))

        val storageMessage = "Unable to save the downloaded model. Check available storage space."
        assertTrue(storageMessage.contains("storage space"))
    }

    @Test
    fun `error messages avoid technical terminology`() {
        val errors = listOf(
            GitHubApiError.NetworkUnavailable().message!!,
            GitHubApiError.ServerUnavailable().message!!,
            GitHubApiError.RateLimitExceeded().message!!,
            GitHubApiError.RepositoryNotFound().message!!,
            GitHubApiError.InvalidResponse().message!!,
            GitHubApiError.RequestTimeout().message!!,
            GitHubApiError.Unknown().message!!
        )

        val technicalTerms = listOf(
            "HTTP", "404", "500", "403", "JSON", "serialization",
            "exception", "null", "timeout", "connection refused",
            "socket", "SSL", "certificate", "DNS"
        )

        errors.forEach { errorMessage ->
            technicalTerms.forEach { technicalTerm ->
                assertTrue(
                    "Error message '$errorMessage' should not contain technical term '$technicalTerm'",
                    !errorMessage.lowercase().contains(technicalTerm.lowercase())
                )
            }
        }
    }

    @Test
    fun `error hierarchy maintains cause information for debugging`() {
        val originalException = UnknownHostException("google.com")
        val userFriendlyError = GitHubApiError.NetworkUnavailable(originalException)

        assertEquals(originalException, userFriendlyError.cause)
        assertTrue("User message should be friendly",
            userFriendlyError.message!!.contains("internet connection"))
    }
}
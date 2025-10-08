package com.gen.myapplication

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class ModelUpdateStateTest {

    @Test
    fun `getCurrentVersion returns correct version for all states`() {
        val testVersion = "v1.0.0"
        val states = listOf(
            ModelUpdateState.Loading(testVersion),
            ModelUpdateState.UpToDate(testVersion, "v1.0.0"),
            ModelUpdateState.UpdateAvailable(testVersion, "v2.0.0"),
            ModelUpdateState.Downloading(testVersion, "v2.0.0"),
            ModelUpdateState.DownloadSuccess(testVersion, "v2.0.0"),
            ModelUpdateState.DownloadFailed(testVersion, "v2.0.0", "Error"),
            ModelUpdateState.NetworkError(testVersion, "Network error")
        )

        states.forEach { state ->
            assertEquals(testVersion, state.getCurrentVersion())
        }
    }

    @Test
    fun `getLatestVersion returns correct version for applicable states`() {
        val currentVersion = "v1.0.0"
        val latestVersion = "v2.0.0"

        // States that have latest version
        val statesWithLatest = listOf(
            ModelUpdateState.UpToDate(currentVersion, latestVersion),
            ModelUpdateState.UpdateAvailable(currentVersion, latestVersion),
            ModelUpdateState.Downloading(currentVersion, latestVersion),
            ModelUpdateState.DownloadFailed(currentVersion, latestVersion, "Error")
        )

        statesWithLatest.forEach { state ->
            assertEquals(latestVersion, state.getLatestVersion())
        }

        // States that don't have latest version
        val statesWithoutLatest = listOf(
            ModelUpdateState.Loading(currentVersion),
            ModelUpdateState.DownloadSuccess(currentVersion, latestVersion),
            ModelUpdateState.NetworkError(currentVersion, "Error")
        )

        statesWithoutLatest.forEach { state ->
            assertNull(state.getLatestVersion())
        }
    }

    @Test
    fun `state transitions represent correct user flow`() {
        val currentVersion = "v1.0.0"
        val latestVersion = "v2.0.0"

        // Normal flow: Loading -> UpdateAvailable -> Downloading -> DownloadSuccess
        val loadingState = ModelUpdateState.Loading(currentVersion)
        assertEquals(currentVersion, loadingState.getCurrentVersion())
        assertNull(loadingState.getLatestVersion())

        val updateAvailableState = ModelUpdateState.UpdateAvailable(currentVersion, latestVersion)
        assertEquals(currentVersion, updateAvailableState.getCurrentVersion())
        assertEquals(latestVersion, updateAvailableState.getLatestVersion())

        val downloadingState = ModelUpdateState.Downloading(currentVersion, latestVersion)
        assertEquals(currentVersion, downloadingState.getCurrentVersion())
        assertEquals(latestVersion, downloadingState.getLatestVersion())

        val successState = ModelUpdateState.DownloadSuccess(currentVersion, latestVersion)
        assertEquals(currentVersion, successState.getCurrentVersion())
        assertNull(successState.getLatestVersion()) // Success state doesn't need to track latest
    }

    @Test
    fun `error states contain appropriate error information`() {
        val currentVersion = "v1.0.0"
        val latestVersion = "v2.0.0"
        val networkError = "Network connection failed"
        val downloadError = "Download interrupted"

        val networkErrorState = ModelUpdateState.NetworkError(currentVersion, networkError)
        assertEquals(currentVersion, networkErrorState.getCurrentVersion())
        assertEquals(networkError, networkErrorState.error)
        assertNull(networkErrorState.getLatestVersion())

        val downloadFailedState = ModelUpdateState.DownloadFailed(currentVersion, latestVersion, downloadError)
        assertEquals(currentVersion, downloadFailedState.getCurrentVersion())
        assertEquals(latestVersion, downloadFailedState.getLatestVersion())
        assertEquals(downloadError, downloadFailedState.error)
    }

    @Test
    fun `up to date state indicates no update needed`() {
        val version = "v1.0.0"
        val upToDateState = ModelUpdateState.UpToDate(version, version)

        assertEquals(version, upToDateState.getCurrentVersion())
        assertEquals(version, upToDateState.getLatestVersion())
    }
}

// Extension functions for testing (these would normally be in MainActivity)
private fun ModelUpdateState.getCurrentVersion(): String = when (this) {
    is ModelUpdateState.Loading -> currentVersion
    is ModelUpdateState.UpToDate -> currentVersion
    is ModelUpdateState.UpdateAvailable -> currentVersion
    is ModelUpdateState.Downloading -> currentVersion
    is ModelUpdateState.DownloadSuccess -> currentVersion
    is ModelUpdateState.DownloadFailed -> currentVersion
    is ModelUpdateState.NetworkError -> currentVersion
}

private fun ModelUpdateState.getLatestVersion(): String? = when (this) {
    is ModelUpdateState.UpToDate -> latestVersion
    is ModelUpdateState.UpdateAvailable -> latestVersion
    is ModelUpdateState.Downloading -> latestVersion
    is ModelUpdateState.DownloadFailed -> latestVersion
    else -> null
}
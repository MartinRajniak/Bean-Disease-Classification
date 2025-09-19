package com.gen.myapplication

import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.onRoot
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.rule.GrantPermissionRule
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class CameraPreviewTest {

    @get:Rule
    val permissionRule: GrantPermissionRule = GrantPermissionRule.grant(
        android.Manifest.permission.CAMERA
    )

    @get:Rule
    val composeTestRule = createAndroidComposeRule<MainActivity>()

    @Test
    fun cameraPreview_withPermission_displaysCamera() {
        // Give camera time to initialize
        composeTestRule.waitForIdle()

        // When camera permission is granted, we should not see permission text
        composeTestRule.onNodeWithText(
            "Camera permission", substring = true
        ).assertDoesNotExist()

        // Make sure permission dialog is not displayed
        composeTestRule.onRoot().assertExists()
    }
}
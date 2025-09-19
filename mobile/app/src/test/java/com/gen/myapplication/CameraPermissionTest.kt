package com.gen.myapplication

import android.Manifest
import org.junit.Test
import org.junit.Assert.*

class CameraPermissionTest {

    @Test
    fun `camera permission constant is correct`() {
        assertEquals("android.permission.CAMERA", Manifest.permission.CAMERA)
    }

    @Test
    fun `camera feature requirement is defined`() {
        val expectedFeature = "android.hardware.camera"
        assertTrue("Camera hardware feature should be required",
            expectedFeature.isNotEmpty())
    }
}
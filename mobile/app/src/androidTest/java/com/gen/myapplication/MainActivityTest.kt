package com.gen.myapplication

import androidx.test.ext.junit.rules.ActivityScenarioRule
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
@LargeTest
class MainActivityTest {

    @get:Rule
    val activityRule = ActivityScenarioRule(MainActivity::class.java)

    @Test
    fun mainActivity_launches_withoutCrashing() {
        // Test that the main activity launches successfully
        // This is a smoke test to ensure basic app functionality
        activityRule.scenario.onActivity { activity ->
            // Activity should be created successfully
            assert(activity != null)
        }
    }

    @Test
    fun app_handles_camera_initialization() {
        // Test that camera initialization doesn't crash the app
        activityRule.scenario.onActivity { activity ->
            // Give some time for camera initialization
            Thread.sleep(1000)
            // App should still be running
            assert(!activity.isFinishing)
        }
    }
}
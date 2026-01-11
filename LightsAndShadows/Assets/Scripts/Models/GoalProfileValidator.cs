using System;
using UnityEngine;

public static class GoalProfileValidator
{
    public static void Validate(ref DroneGoalProfile[] profiles, ref int goalProfileIndex, string context)
    {
        try
        {
            if (profiles == null || profiles.Length == 0)
            {
                Debug.LogWarning($"[GoalProfileValidator] {context} missing goal profiles. Applying default profile.");
                profiles = new[] { DroneGoalProfile.Default };
                goalProfileIndex = 0;
                return;
            }

            goalProfileIndex = Mathf.Clamp(goalProfileIndex, 0, profiles.Length - 1);

            if (!IsProfileValid(profiles[goalProfileIndex]))
            {
                Debug.LogWarning($"[GoalProfileValidator] {context} active goal profile is invalid. Replacing with default.");
                profiles[goalProfileIndex] = DroneGoalProfile.Default;
            }
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[GoalProfileValidator] {context} validation failed: {ex.Message}");
            profiles = new[] { DroneGoalProfile.Default };
            goalProfileIndex = 0;
        }
    }

    private static bool IsProfileValid(DroneGoalProfile profile)
    {
        if (float.IsNaN(profile.heightWeight) || float.IsInfinity(profile.heightWeight))
        {
            return false;
        }
        if (float.IsNaN(profile.distanceWeight) || float.IsInfinity(profile.distanceWeight))
        {
            return false;
        }
        if (float.IsNaN(profile.yawWeight) || float.IsInfinity(profile.yawWeight))
        {
            return false;
        }
        if (float.IsNaN(profile.stabilityWeight) || float.IsInfinity(profile.stabilityWeight))
        {
            return false;
        }
        if (float.IsNaN(profile.speedWeight) || float.IsInfinity(profile.speedWeight))
        {
            return false;
        }

        float totalWeight = Mathf.Abs(profile.heightWeight)
            + Mathf.Abs(profile.distanceWeight)
            + Mathf.Abs(profile.yawWeight)
            + Mathf.Abs(profile.stabilityWeight)
            + Mathf.Abs(profile.speedWeight);

        return totalWeight > Mathf.Epsilon;
    }
}

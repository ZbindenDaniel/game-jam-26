using System;

[Serializable]
public struct DroneGoalProfile
{
    public string profileName;
    public float heightWeight;
    public float distanceWeight;
    public float yawWeight;
    public float stabilityWeight;
    public float speedWeight;

    public static DroneGoalProfile Default => new DroneGoalProfile
    {
        profileName = "Default",
        heightWeight = 1f,
        distanceWeight = 1f,
        yawWeight = 1f,
        stabilityWeight = 0f,
        speedWeight = 0f
    };
}

using System;

[Serializable]
public class RunMetric
{
    public int runIndex;
    public float averageError;
    public float avgHeightError;
    public float avgDistanceError;
    public float avgYawError;
    public float avgSpeed;
    public float avgAngularVelocity;
    public float invertedTime;
    public float invertRatio;
    public float maxHeightError;
    public float maxDistanceError;
    public float maxYawError;
    public float maxTiltDeg;
    public int collisionCount;
    public float avgImpactSeverity;
    public float maxImpactSeverity;
    public float timeAlive;
    public int goalProfileIndex;
    public string goalProfileName;
    public float goalHeightWeight;
    public float goalDistanceWeight;
    public float goalYawWeight;
    public float goalStabilityWeight;
    public float goalSpeedWeight;
    public float heightKp, heightKi, heightKd;
    public float distanceKp, distanceKi, distanceKd;
    public float pitchKp, pitchKi, pitchKd;
    public float yawKp, yawKi, yawKd;
    public float rollKp, rollKi, rollKd;
}

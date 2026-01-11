using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;

public class MetricsRecorder : MonoBehaviour
{
    [Header("Episode Settings")]
    [SerializeField] private float runDuration = 10f;
    [SerializeField] private Text display;
    [SerializeField] private bool persistMetrics = true;
    [SerializeField] private string metricsFilename = "drone_metrics.json";
    [Tooltip("Log collisions and impact severity for debugging damage signals.")]
    [SerializeField] private bool logCollisionMetrics = false;

    [Header("Goal Profiles")]
    [Tooltip("Index into the goal profile array used to weight composite metrics.")]
    [SerializeField] private int goalProfileIndex = 0;
    [Tooltip("Profiles defining how composite metrics are weighted.")]
    [SerializeField] private DroneGoalProfile[] goalProfiles;
    [Tooltip("Log the active profile name/weights once per episode.")]
    [SerializeField] private bool logGoalProfilePerEpisode = true;

    [Header("Dependencies")]
    [SerializeField] private Rigidbody rb;
    [SerializeField] private BehaviourSelector behaviourSelector;

    private float sumSqHeightError;
    private float sumSqDistanceError;
    private float sumSqYawError;
    private float sumInvertedTime;
    private float sumSpeed;
    private float sumAngularVelocity;
    private float maxAbsHeightError;
    private float maxAbsDistanceError;
    private float maxAbsYawError;
    private float maxAbsTiltDeg;
    private int collisionCount;
    private float sumImpactSeverity;
    private float maxImpactSeverity;
    private int runCount;
    private float lastAvgError;
    private readonly List<RunMetric> metrics = new List<RunMetric>();
    private bool loggedGoalProfileThisEpisode;

    public float RunDuration => runDuration;

    private void Awake()
    {
        if (rb == null)
        {
            rb = GetComponent<Rigidbody>();
        }

        InitializeGoalProfiles();
    }

    public void AccumulateFrame(float dt, float heightError, float distanceError, float yawErrorDeg, float tiltDeg, bool isInverted)
    {
        sumSqHeightError += heightError * heightError * dt;
        sumSqDistanceError += distanceError * distanceError * dt;
        sumSqYawError += yawErrorDeg * yawErrorDeg * dt;

        if (Mathf.Abs(heightError) > maxAbsHeightError)
        {
            maxAbsHeightError = Mathf.Abs(heightError);
        }
        if (Mathf.Abs(distanceError) > maxAbsDistanceError)
        {
            maxAbsDistanceError = Mathf.Abs(distanceError);
        }
        if (Mathf.Abs(yawErrorDeg) > maxAbsYawError)
        {
            maxAbsYawError = Mathf.Abs(yawErrorDeg);
        }
        if (tiltDeg > maxAbsTiltDeg)
        {
            maxAbsTiltDeg = tiltDeg;
        }
        if (isInverted)
        {
            sumInvertedTime += dt;
        }

        try
        {
            if (rb != null)
            {
                sumSpeed += rb.velocity.magnitude * dt;
                sumAngularVelocity += rb.angularVelocity.magnitude * dt;
            }
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[PlayerDrone] Metric aggregation failed: {ex.Message}");
        }
    }

    public void RecordMetrics(float simulationTimer)
    {
        float avgHeight = simulationTimer > 0f ? sumSqHeightError / simulationTimer : 0f;
        float avgDist = simulationTimer > 0f ? sumSqDistanceError / simulationTimer : 0f;
        float avgYaw = simulationTimer > 0f ? sumSqYawError / simulationTimer : 0f;
        float avgSpeed = simulationTimer > 0f ? sumSpeed / simulationTimer : 0f;
        float avgAngularVelocity = simulationTimer > 0f ? sumAngularVelocity / simulationTimer : 0f;
        float avgImpactSeverity = collisionCount > 0 ? sumImpactSeverity / collisionCount : 0f;
        float invertRatio = simulationTimer > 0f ? sumInvertedTime / simulationTimer : 0f;
        int activeProfileIndex = GetActiveGoalProfileIndex();
        DroneGoalProfile profile = GetActiveGoalProfile();
        LogGoalProfileIfNeeded(profile);

        float composite = (avgHeight * profile.heightWeight)
            + (avgDist * profile.distanceWeight)
            + (avgYaw * profile.yawWeight)
            + (invertRatio * profile.stabilityWeight)
            + (avgSpeed * profile.speedWeight);

        AutoTunePID heightPID = behaviourSelector != null ? behaviourSelector.HeightPID : null;
        AutoTunePID distancePID = behaviourSelector != null ? behaviourSelector.DistancePID : null;
        AutoTunePID pitchPID = behaviourSelector != null ? behaviourSelector.PitchPID : null;
        AutoTunePID yawPID = behaviourSelector != null ? behaviourSelector.YawPID : null;
        AutoTunePID rollPID = behaviourSelector != null ? behaviourSelector.RollPID : null;

        var rm = new RunMetric
        {
            runIndex = runCount,
            averageError = composite,
            avgHeightError = avgHeight,
            avgDistanceError = avgDist,
            avgYawError = avgYaw,
            avgSpeed = avgSpeed,
            avgAngularVelocity = avgAngularVelocity,
            invertedTime = sumInvertedTime,
            invertRatio = invertRatio,
            maxHeightError = maxAbsHeightError,
            maxDistanceError = maxAbsDistanceError,
            maxYawError = maxAbsYawError,
            maxTiltDeg = maxAbsTiltDeg,
            collisionCount = collisionCount,
            avgImpactSeverity = avgImpactSeverity,
            maxImpactSeverity = maxImpactSeverity,
            timeAlive = simulationTimer,
            goalProfileIndex = activeProfileIndex,
            goalProfileName = profile.profileName,
            goalHeightWeight = profile.heightWeight,
            goalDistanceWeight = profile.distanceWeight,
            goalYawWeight = profile.yawWeight,
            goalStabilityWeight = profile.stabilityWeight,
            goalSpeedWeight = profile.speedWeight,
            heightKp = heightPID != null ? heightPID.controller.Kp : 0f,
            heightKi = heightPID != null ? heightPID.controller.Ki : 0f,
            heightKd = heightPID != null ? heightPID.controller.Kd : 0f,
            distanceKp = distancePID != null ? distancePID.controller.Kp : 0f,
            distanceKi = distancePID != null ? distancePID.controller.Ki : 0f,
            distanceKd = distancePID != null ? distancePID.controller.Kd : 0f,
            pitchKp = pitchPID != null ? pitchPID.controller.Kp : 0f,
            pitchKi = pitchPID != null ? pitchPID.controller.Ki : 0f,
            pitchKd = pitchPID != null ? pitchPID.controller.Kd : 0f,
            yawKp = yawPID != null ? yawPID.controller.Kp : 0f,
            yawKi = yawPID != null ? yawPID.controller.Ki : 0f,
            yawKd = yawPID != null ? yawPID.controller.Kd : 0f,
            rollKp = rollPID != null ? rollPID.controller.Kp : 0f,
            rollKi = rollPID != null ? rollPID.controller.Ki : 0f,
            rollKd = rollPID != null ? rollPID.controller.Kd : 0f
        };
        metrics.Add(rm);

        if (display != null)
        {
            float delta = runCount > 0 ? composite - lastAvgError : 0f;
            display.text =
                $"Runs: {runCount + 1}\n" +
                $"AvgErr: {composite:F3}\n" +
                $"ΔErr: {delta:F3}\n" +
                $"AvgH: {avgHeight:F3} AvgD: {avgDist:F3} AvgYaw: {avgYaw:F3}\n" +
                $"MaxH: {maxAbsHeightError:F3} MaxD: {maxAbsDistanceError:F3} MaxYaw: {maxAbsYawError:F3}\n" +
                $"Invert%: {invertRatio * 100f:F2}%";
        }
        runCount++;
        lastAvgError = composite;
    }

    public void ResetEpisode()
    {
        sumSqHeightError = 0f;
        sumSqDistanceError = 0f;
        sumSqYawError = 0f;
        sumInvertedTime = 0f;
        sumSpeed = 0f;
        sumAngularVelocity = 0f;
        maxAbsHeightError = 0f;
        maxAbsDistanceError = 0f;
        maxAbsYawError = 0f;
        maxAbsTiltDeg = 0f;
        collisionCount = 0;
        sumImpactSeverity = 0f;
        maxImpactSeverity = 0f;
        loggedGoalProfileThisEpisode = false;
    }

    private void OnCollisionEnter(Collision collision)
    {
        try
        {
            if (collision == null)
            {
                return;
            }

            float impactSeverity = collision.relativeVelocity.magnitude;
            collisionCount++;
            sumImpactSeverity += impactSeverity;
            if (impactSeverity > maxImpactSeverity)
            {
                maxImpactSeverity = impactSeverity;
            }

            if (logCollisionMetrics)
            {
                Debug.Log($"[PlayerDrone] Collision recorded on {name}: impact={impactSeverity:F2}, count={collisionCount}");
            }
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[PlayerDrone] Collision handling failed: {ex.Message}");
        }
    }

    private void OnApplicationQuit()
    {
        SaveMetricsToFile();
    }

    private void OnDisable()
    {
        SaveMetricsToFile();
    }

    public void SaveMetricsToFile()
    {
        if (!persistMetrics || metrics.Count == 0)
        {
            return;
        }
        try
        {
            var wrapper = new MetricsWrapper
            {
                runs = metrics,
                goalProfiles = goalProfiles != null ? new List<DroneGoalProfile>(goalProfiles) : new List<DroneGoalProfile>()
            };
            string json = JsonUtility.ToJson(wrapper, true);
            string path = Path.Combine(Application.persistentDataPath, metricsFilename);
            File.WriteAllText(path, json);
            Debug.Log($"[PlayerDrone] Metrics saved to {path}");
        }
        catch (Exception e)
        {
            Debug.LogError($"[PlayerDrone] Failed to write metrics: {e.Message}");
        }
    }

    private void InitializeGoalProfiles()
    {
        if (goalProfiles == null || goalProfiles.Length == 0)
        {
            goalProfiles = new[]
            {
                DroneGoalProfile.Default
            };
        }

        goalProfileIndex = Mathf.Clamp(goalProfileIndex, 0, goalProfiles.Length - 1);
    }

    private int GetActiveGoalProfileIndex()
    {
        if (goalProfiles == null || goalProfiles.Length == 0)
        {
            return 0;
        }

        return Mathf.Clamp(goalProfileIndex, 0, goalProfiles.Length - 1);
    }

    private DroneGoalProfile GetActiveGoalProfile()
    {
        if (goalProfiles == null || goalProfiles.Length == 0)
        {
            return DroneGoalProfile.Default;
        }

        return goalProfiles[GetActiveGoalProfileIndex()];
    }

    private void LogGoalProfileIfNeeded(DroneGoalProfile profile)
    {
        if (!logGoalProfilePerEpisode || loggedGoalProfileThisEpisode)
        {
            return;
        }

        try
        {
            string profileName = string.IsNullOrWhiteSpace(profile.profileName) ? "Unnamed" : profile.profileName;
            Debug.Log($"[PlayerDrone] Goal profile '{profileName}' weights: height={profile.heightWeight:F2}, distance={profile.distanceWeight:F2}, yaw={profile.yawWeight:F2}, stability={profile.stabilityWeight:F2}, speed={profile.speedWeight:F2}.");
            loggedGoalProfileThisEpisode = true;
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[PlayerDrone] Goal profile logging failed: {ex.Message}");
        }
    }
}

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

[Serializable]
public class MetricsWrapper
{
    public List<RunMetric> runs;
    public List<DroneGoalProfile> goalProfiles;
}

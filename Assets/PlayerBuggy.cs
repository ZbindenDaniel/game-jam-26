using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;
using System.IO;

/// <summary>
/// PlayerBuggy implements a simple wheeled vehicle controller using PID loops
/// for acceleration, steering and obstacle avoidance.  The behaviour loosely
/// follows the structure of PlayerDrone but is adapted for a six wheel car
/// using WheelCollider components.  A forward looking raycast cone is used
/// to detect nearby obstacles; a PID controller scales the influence of
/// avoidance on the desired direction to the target.  Distance and heading
/// errors are accumulated during each episode for auto‑tuning of the gains.
/// Episodes reset the buggy to its initial pose and optionally persist
/// performance metrics.
/// </summary>
[RequireComponent(typeof(Rigidbody))]
public class PlayerBuggy : MonoBehaviour
{
    // ----------------------------------------------------------------------
    // Wheel configuration
    // ----------------------------------------------------------------------
    [Header("Wheel Colliders (Rear for motor, Front for steering)")]
    /// <summary>
    /// Wheel colliders providing traction to the rear axle.  Four wheels are
    /// expected for propulsion.  Assign these in the inspector.
    /// </summary>
    public WheelCollider[] rearWheels;
    /// <summary>
    /// Wheel colliders on the front axle responsible for steering the buggy.
    /// Two wheels are expected.  Assign these in the inspector.
    /// </summary>
    public WheelCollider[] frontWheels;

    // ----------------------------------------------------------------------
    // Motor and steering limits
    // ----------------------------------------------------------------------
    [Header("Drive Parameters")]
    /// <summary>Maximum motor torque applied to each rear wheel (N·m).</summary>
    public float maxMotorTorque = 150f;
    /// <summary>Maximum steering angle in degrees for the front wheels.</summary>
    public float maxSteerAngle = 25f;

    // ----------------------------------------------------------------------
    // Targeting and avoidance parameters
    // ----------------------------------------------------------------------
    [Header("Target Settings")]
    /// <summary>The transform of the waypoint the buggy should approach.</summary>
    public Transform currentWaypoint;
    /// <summary>The desired horizontal distance to maintain from the waypoint.
    /// If zero the buggy will try to reach the waypoint directly.</summary>
    public float desiredDistance = 0f;

    [Header("Field of View for Obstacle Avoidance")]
    /// <summary>Half angle of the obstacle detection cone in degrees.</summary>
    public float viewAngle = 30f;
    /// <summary>Maximum distance to search for obstacles.</summary>
    public float viewDistance = 20f;
    /// <summary>Physics layer mask used to identify obstacles.  Configure this to
    /// include static geometry and exclude the buggy itself.</summary>
    public LayerMask obstacleLayers = ~0;

    // ----------------------------------------------------------------------
    // Controllers
    // ----------------------------------------------------------------------
    [Header("PID Controllers")]
    /// <summary>Controls the forward/backward acceleration of the buggy based on
    /// distance error to the target.</summary>
    public AutoTunePID accelerationPID = new AutoTunePID();
    /// <summary>Controls the steering angle based on the angular error between
    /// the buggy’s forward vector and the desired direction.</summary>
    public AutoTunePID steeringPID = new AutoTunePID();
    /// <summary>Controls the weighting of obstacle avoidance relative to the
    /// direct path towards the target.  Larger outputs from this PID increase
    /// the influence of the avoidance vector.</summary>
    public AutoTunePID obstaclePID = new AutoTunePID();

    // ----------------------------------------------------------------------
    // Auto tuning parameters
    // ----------------------------------------------------------------------
    [Header("Auto Tuning Settings")]
    /// <summary>If true, gradient‑descent tuning of the PID gains will be
    /// applied at the end of each episode.</summary>
    public bool enableAutoTuning = false;
    /// <summary>Learning rate for gradient descent.  Smaller values result in
    /// slower but steadier tuning.</summary>
    public float learningRate = 0.001f;

    // ----------------------------------------------------------------------
    // Episode management and metrics
    // ----------------------------------------------------------------------
    [Header("Episode Settings")]
    /// <summary>Duration of an episode in seconds.  After this time the
    /// buggy is reset and metrics are recorded.</summary>
    public float runDuration = 10f;
    /// <summary>Optional text object for displaying statistics on screen.</summary>
    public Text display;
    /// <summary>If true, metrics are written to disk when the application
    /// quits or the component is disabled.</summary>
    public bool persistMetrics = true;
    /// <summary>Name of the file used to persist metrics.  The file is
    /// created in Application.persistentDataPath.</summary>
    public string metricsFilename = "buggy_metrics.json";

    // Internal state
    private Rigidbody rb;
    private float simulationTimer;
    private Vector3 initialPosition;
    private Quaternion initialRotation;
    // Metrics accumulation
    private float sumSqDistanceError;
    private float sumSqAngleError;
    private float sumSqObstacleError;
    private float sumInvertedTime;
    private int crashCount;
    private float maxAbsDistanceError;
    private float maxAbsAngleError;
    private float maxAbsObstacleError;
    private int runCount;
    private float lastAvgError;
    private readonly List<CarRunMetric> metrics = new List<CarRunMetric>();

    // Flag used to avoid counting multiple collision events in one frame
    private bool collidedThisFrame;

    void Awake()
    {
        rb = GetComponent<Rigidbody>();
        initialPosition = transform.position;
        initialRotation = transform.rotation;
    }

    void FixedUpdate()
    {
        float dt = Time.fixedDeltaTime;
        simulationTimer += dt;
        collidedThisFrame = false;

        // Determine the current target position.  If no waypoint is assigned
        // the buggy remains stationary at its initial position.
        Vector3 targetPos = currentWaypoint != null ? currentWaypoint.position : transform.position;

        // Compute horizontal vector to the target in the XZ plane
        Vector3 toTarget = targetPos - transform.position;
        Vector3 toTargetXZ = Vector3.ProjectOnPlane(toTarget, Vector3.up);
        float distanceError = toTargetXZ.magnitude - desiredDistance;

        // Construct the obstacle avoidance vector and corresponding error
        Vector3 obstacleVec;
        float obstacleError;
        ComputeObstacleVectorAndError(out obstacleVec, out obstacleError);

        // Determine desired direction: toward the target plus avoidance influence
        Vector3 targetDir = toTargetXZ.sqrMagnitude > 0.0001f ? toTargetXZ.normalized : Vector3.zero;
        // Evaluate the obstacle PID to determine weighting.  Zero error when no
        // obstacle is hit yields zero influence; positive error increases the
        // avoidance term.  Negative weighting is not permitted.
        float obstacleWeight = obstaclePID.controller.Update(obstacleError, dt);
        obstacleWeight = Mathf.Clamp(obstacleWeight, 0f, 1f);
        Vector3 avoidanceDir = obstacleVec.sqrMagnitude > 0.0001f ? obstacleVec.normalized : Vector3.zero;
        Vector3 desiredDir = (targetDir + avoidanceDir * obstacleWeight);
        // Normalise to avoid scaling of the steering command by combined magnitude
        Vector3 finalDir = desiredDir.sqrMagnitude > 0.0001f ? desiredDir.normalized : Vector3.zero;

        // Compute the heading error between the buggy's forward direction and the desired direction
        Vector3 forwardXZ = Vector3.ProjectOnPlane(transform.forward, Vector3.up);
        float angleErrorDeg = Vector3.SignedAngle(forwardXZ, finalDir, Vector3.up);
        float angleErrorRad = angleErrorDeg * Mathf.Deg2Rad;

        // Update the steering and acceleration PIDs
        float steerCmd = steeringPID.controller.Update(angleErrorRad, dt);
        // Clamp to [-1, 1] to prevent overly large steering angles
        steerCmd = Mathf.Clamp(steerCmd, -1f, 1f);
        float accelCmd = accelerationPID.controller.Update(distanceError, dt);
        // Negative acceleration allows for reversing; clamp to [-1, 1]
        accelCmd = Mathf.Clamp(accelCmd, -1f, 1f);

        // Apply steering to front wheels
        if (frontWheels != null)
        {
            foreach (var wheel in frontWheels)
            {
                if (wheel != null)
                {
                    wheel.steerAngle = steerCmd * maxSteerAngle;
                }
            }
        }

        // Apply motor torque to rear wheels
        if (rearWheels != null)
        {
            foreach (var wheel in rearWheels)
            {
                if (wheel != null)
                {
                    wheel.motorTorque = accelCmd * maxMotorTorque;
                }
            }
        }

        // Accumulate metrics
        sumSqDistanceError += distanceError * distanceError * dt;
        sumSqAngleError += angleErrorRad * angleErrorRad * dt;
        sumSqObstacleError += obstacleError * obstacleError * dt;
        if (Mathf.Abs(distanceError) > maxAbsDistanceError) maxAbsDistanceError = Mathf.Abs(distanceError);
        if (Mathf.Abs(angleErrorRad) > maxAbsAngleError) maxAbsAngleError = Mathf.Abs(angleErrorRad);
        if (Mathf.Abs(obstacleError) > maxAbsObstacleError) maxAbsObstacleError = Mathf.Abs(obstacleError);
        // Inversion detection: treat upside down as a penalty
        if (Vector3.Dot(transform.up, Vector3.up) < 0f)
        {
            sumInvertedTime += dt;
        }

        // Gradient accumulation for auto tuning
        if (enableAutoTuning)
        {
            if (accelerationPID.optimize) accelerationPID.AccumulateGradients(distanceError, dt);
            if (steeringPID.optimize) steeringPID.AccumulateGradients(angleErrorRad, dt);
            if (obstaclePID.optimize) obstaclePID.AccumulateGradients(obstacleError, dt);
        }

        // End of episode: record metrics and reset
        if (simulationTimer >= runDuration)
        {
            RecordMetrics();
            ApplyAutoTuning();
            ResetEpisode();
        }
    }

    /// <summary>
    /// Casts a set of rays in a forward‑oriented cone to detect obstacles.  The
    /// resulting vector sums contributions from each hit ray pointing away from
    /// the obstacle.  The obstacle error is the normalised inverse distance
    /// (1 - distance/viewDistance) for the closest hit.  If no obstacle is
    /// detected obstacleError is zero and obstacleVec is zero.
    /// </summary>
    /// <param name="obstacleVec">The summed avoidance vector pointing away from obstacles.</param>
    /// <param name="obstacleError">A scalar error proportional to the proximity of the closest obstacle.</param>
    private void ComputeObstacleVectorAndError(out Vector3 obstacleVec, out float obstacleError)
    {
        obstacleVec = Vector3.zero;
        obstacleError = 0f;
        float closest = float.PositiveInfinity;
        // Cast a fixed number of rays uniformly spanning the view angle
        int rayCount = 5;
        for (int i = 0; i < rayCount; i++)
        {
            float t = rayCount > 1 ? i / (float)(rayCount - 1) : 0.5f;
            float angle = Mathf.Lerp(-viewAngle, viewAngle, t);
            // Rotate the forward vector around Y by the given angle
            Vector3 dir = Quaternion.Euler(0f, angle, 0f) * transform.forward;
            Ray ray = new Ray(transform.position + Vector3.up * 0.5f, dir);
            if (Physics.Raycast(ray, out RaycastHit hit, viewDistance, obstacleLayers, QueryTriggerInteraction.Ignore))
            {
                // Weight inversely with distance so closer obstacles contribute more
                float weight = 1f - (hit.distance / viewDistance);
                obstacleVec += -dir * weight;
                if (hit.distance < closest)
                {
                    closest = hit.distance;
                }
            }
        }
        if (closest < float.PositiveInfinity)
        {
            obstacleError = 1f - (closest / viewDistance);
        }
    }

    /// <summary>
    /// Records the aggregated metrics at the end of an episode.  Average
    /// errors are computed by dividing the accumulated squared errors by the
    /// simulation time.  A composite metric equal to the sum of the
    /// individual averages is used as a convenient scalar for comparison.
    /// </summary>
    private void RecordMetrics()
    {
        float duration = simulationTimer > 0f ? simulationTimer : 1f;
        float avgDist = sumSqDistanceError / duration;
        float avgAng = sumSqAngleError / duration;
        float avgObs = sumSqObstacleError / duration;
        float composite = avgDist + avgAng + avgObs;
        float invertRatio = sumInvertedTime / duration;
        var rm = new CarRunMetric
        {
            runIndex = runCount,
            averageError = composite,
            avgDistanceError = avgDist,
            avgAngleError = avgAng,
            avgObstacleError = avgObs,
            invertedTime = sumInvertedTime,
            invertRatio = invertRatio,
            crashCount = crashCount,
            maxDistanceError = maxAbsDistanceError,
            maxAngleError = maxAbsAngleError,
            maxObstacleError = maxAbsObstacleError,
            accelerationKp = accelerationPID.controller.Kp,
            accelerationKi = accelerationPID.controller.Ki,
            accelerationKd = accelerationPID.controller.Kd,
            steeringKp = steeringPID.controller.Kp,
            steeringKi = steeringPID.controller.Ki,
            steeringKd = steeringPID.controller.Kd,
            obstacleKp = obstaclePID.controller.Kp,
            obstacleKi = obstaclePID.controller.Ki,
            obstacleKd = obstaclePID.controller.Kd
        };
        metrics.Add(rm);
        // Update display if assigned
        if (display != null)
        {
            float delta = runCount > 0 ? composite - lastAvgError : 0f;
            display.text =
                $"Runs: {runCount + 1}\n" +
                $"AvgErr: {composite:F3}\n" +
                $"ΔErr: {delta:F3}\n" +
                $"AvgDist: {avgDist:F3} AvgAng: {avgAng:F3} AvgObs: {avgObs:F3}\n" +
                $"MaxDist: {maxAbsDistanceError:F3} MaxAng: {maxAbsAngleError:F3} MaxObs: {maxAbsObstacleError:F3}\n" +
                $"Invert%: {invertRatio * 100f:F2}% Crash: {crashCount}";
        }
        runCount++;
        lastAvgError = composite;
    }

    /// <summary>
    /// Applies gradient descent to each PID controller based on the
    /// accumulated gradients during the episode.  Gains are incremented by
    /// learningRate multiplied by the average gradient.  Negative gains are
    /// clamped to zero.  All gradient accumulators are reset afterwards.
    /// </summary>
    private void ApplyAutoTuning()
    {
        if (!enableAutoTuning || simulationTimer <= 0f) return;
        accelerationPID.ApplyTuning(simulationTimer, learningRate);
        steeringPID.ApplyTuning(simulationTimer, learningRate);
        obstaclePID.ApplyTuning(simulationTimer, learningRate);
    }

    /// <summary>
    /// Resets the simulation state for a new episode.  Errors and counters are
    /// cleared, the buggy is repositioned to its initial pose and the PIDs
    /// internal state is reset to prevent integral wind‑up.  The waypoint
    /// remains at its assigned position.
    /// </summary>
    private void ResetEpisode()
    {
        simulationTimer = 0f;
        sumSqDistanceError = 0f;
        sumSqAngleError = 0f;
        sumSqObstacleError = 0f;
        sumInvertedTime = 0f;
        crashCount = 0;
        maxAbsDistanceError = 0f;
        maxAbsAngleError = 0f;
        maxAbsObstacleError = 0f;
        transform.position = initialPosition;
        transform.rotation = initialRotation;
        if (rb != null)
        {
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
        }
        // Reset wheel torques and steer angles
        if (rearWheels != null)
        {
            foreach (var wheel in rearWheels)
            {
                if (wheel != null)
                {
                    wheel.motorTorque = 0f;
                }
            }
        }
        if (frontWheels != null)
        {
            foreach (var wheel in frontWheels)
            {
                if (wheel != null)
                {
                    wheel.steerAngle = 0f;
                }
            }
        }
        // Reset PIDs and gradient accumulators
        accelerationPID.ResetTuning();
        steeringPID.ResetTuning();
        obstaclePID.ResetTuning();
    }

    /// <summary>
    /// Records a crash when the buggy collides with an obstacle.  Multiple
    /// collisions within a single FixedUpdate are counted once.  Collisions
    /// with the terrain or other benign objects can be filtered by layers.
    /// </summary>
    /// <param name="collision">Collision data</param>
    void OnCollisionEnter(Collision collision)
    {
        if (collidedThisFrame) return;
        // Ignore collisions with the buggy’s own colliders by comparing
        // root transforms.  Only count collisions with external objects.
        if (collision != null && collision.transform != transform)
        {
            // Optionally filter by layer if needed
            crashCount++;
            collidedThisFrame = true;
        }
    }

    void OnApplicationQuit()
    {
        SaveMetricsToFile();
    }

    void OnDisable()
    {
        SaveMetricsToFile();
    }

    /// <summary>
    /// Writes the collected run metrics to a JSON file in the persistent data
    /// path if persistence is enabled.  A wrapper class is used because
    /// Unity’s JsonUtility cannot serialise bare lists.  Errors are
    /// silently logged.
    /// </summary>
    private void SaveMetricsToFile()
    {
        if (!persistMetrics || metrics.Count == 0) return;
        try
        {
            var wrapper = new CarMetricsWrapper { runs = metrics };
            string json = JsonUtility.ToJson(wrapper, true);
            string path = Path.Combine(Application.persistentDataPath, metricsFilename);
            File.WriteAllText(path, json);
            Debug.Log("Buggy metrics saved to " + path);
        }
        catch (System.Exception e)
        {
            Debug.LogError("Failed to write buggy metrics: " + e.Message);
        }
    }
}

/// <summary>
/// Data structure containing aggregate statistics for a single buggy run.  The
/// average of each error type is recorded along with the cumulative time
/// inverted and number of collisions.  PID gains at the end of the run are
/// stored for analysis.
/// </summary>
[System.Serializable]
public class CarRunMetric
{
    public int runIndex;
    public float averageError;
    public float avgDistanceError;
    public float avgAngleError;
    public float avgObstacleError;
    public float invertedTime;
    public float invertRatio;
    public int crashCount;
    public float maxDistanceError;
    public float maxAngleError;
    public float maxObstacleError;
    public float accelerationKp, accelerationKi, accelerationKd;
    public float steeringKp, steeringKi, steeringKd;
    public float obstacleKp, obstacleKi, obstacleKd;
}

/// <summary>
/// Wrapper class used to serialise a list of CarRunMetric objects.  Unity’s
/// JsonUtility cannot serialise bare lists so this wrapper is required to
/// persist the metrics to disk.
/// </summary>
[System.Serializable]
public class CarMetricsWrapper
{
    public List<CarRunMetric> runs;
}
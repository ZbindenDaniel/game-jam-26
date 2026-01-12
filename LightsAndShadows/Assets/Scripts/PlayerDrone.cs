using UnityEngine;
using System;


/// <summary>
/// A physics‑based controller for a quad‑drone simulation.  The controller
/// provides independent control loops for height, distance, pitch, yaw and
/// roll.  Optional yaw thrusters decouple heading from lift and an idle
/// path generator keeps the drone moving in a circular path when no
/// waypoint is assigned.  A field of view cone is exposed for
/// detection logic.  Auto tuning of the PID gains can be enabled on a
/// per‑controller basis via the AutoTunePID wrappers.  During each
/// episode metrics are gathered and can be persisted to a JSON file for
/// analysis.
/// </summary>
[RequireComponent(typeof(Rigidbody))]
public class PlayerDrone : MonoBehaviour
{
    // ------------------------------------------------------------------------
    // Thruster configuration
    // ------------------------------------------------------------------------
    [Header("Lift Thrusters (Upwards)")]
    public Transform thrusterFL;
    public Transform thrusterFR;
    public Transform thrusterBL;
    public Transform thrusterBR;

    [Header("Yaw Thrusters (Sideways)")]
    /// <summary>
    /// Left and right thrusters responsible for generating pure yaw torque.  These
    /// thrusters are assumed to produce force along the ±Right axis and are
    /// positioned symmetrically around the drone’s centre of mass.  They
    /// contribute no vertical thrust and therefore do not interfere with height
    /// and pitch control.
    /// </summary>
    public Transform yawThrusterLeft;
    public Transform yawThrusterRight;

    // ------------------------------------------------------------------------
    // Thrust limits and control parameters
    // ------------------------------------------------------------------------
    [Header("Thrust Limits")]
    public float maxLiftThrustPerThruster = 15f; // N
    public float maxYawThrust = 5f;              // N used for yaw torque
    public float maxPitchAngleDeg = 20f;         // maximum commanded pitch in degrees

    [Range(0f, 1f)]
    public float thrustSmoothing = 0.2f;

    // ------------------------------------------------------------------------
    // Subsystem references
    // ------------------------------------------------------------------------
    [Header("Subsystems")]
    [SerializeField] private EnergyController energyController;
    [SerializeField] private BehaviourSelector behaviourSelector;
    [SerializeField] private MetricsRecorder metricsRecorder;

    // ------------------------------------------------------------------------
    // Target definitions
    // ------------------------------------------------------------------------
    [Header("Target")]
    /// <summary>
    /// Desired drone altitude (world Y) for height control.  This setting does not
    /// adjust the waypoint’s Y position; the waypoint remains at its own Y (e.g. the
    /// ground) while the drone maintains this altitude.
    /// </summary>
    public float targetAltitude = 3f;
    /// <summary>The desired distance from the waypoint in the XZ plane.
    /// If zero the drone will attempt to hover directly above the waypoint.</summary>
    public float desiredDistance = 0f;

    // ------------------------------------------------------------------------
    // Idle path parameters
    // ------------------------------------------------------------------------
    [Header("Idle Path Settings")]
    /// <summary>If true and no waypoint is assigned, the drone will
    /// follow a circular idle path centred on idleCenter.</summary>
    public bool idleMode = true;
    /// <summary>The centre point around which the drone moves when in
    /// idle mode.  Defaults to the drone’s initial position.</summary>
    public Vector3 idleCenter;
    /// <summary>The radius of the idle path circle in metres.</summary>
    public float idleRadius = 5f;
    /// <summary>The angular speed (radians per second) of the idle path.
    /// Positive values make the path counter‑clockwise when looking down
    /// the Y axis.</summary>
    public float idleAngularSpeed = 1f;
    /// <summary>Amplitude of Perlin noise deviations added to the idle
    /// path.  Set to zero for a perfect circle.</summary>
    public float idleDeviationAmplitude = 0f;
    /// <summary>Frequency of the Perlin noise used for idle deviations.</summary>
    public float idleNoiseFrequency = 0.5f;

    // ------------------------------------------------------------------------
    // Avoidance settings
    // ------------------------------------------------------------------------
    [Header("Avoidance Settings")]
    public float maxAvoidanceClimb = 3f;
    public float avoidanceDotThreshold = -0.1f;
    public float avoidanceMinMagnitude = 0.5f;
    public float avoidanceLogInterval = 1f;
    public float maxTargetInfluence = 6f;

    // Internal runtime state
    private Rigidbody rb;
    private float simulationTimer = 0f;
    private Vector3 initialPosition;
    private Quaternion initialRotation;
    private Vector3 waypointStartPos;
    // Idle path generator instance used when no waypoint is assigned
    private CircularPathGenerator idlePath;
    private readonly float[] liftOutputs = new float[4];
    private float baseMaxLiftThrustPerThruster;
    private float baseMaxYawThrust;
    private float defaultAltitude;
    private float dynamicAltitudeOffset;
    private float lastAvoidanceLogTime = float.NegativeInfinity;
    private float lastAvoidancePidLogTime = float.NegativeInfinity;
    private float lastTargetLogTime = float.NegativeInfinity;

    // ------------------------------------------------------------------------
    // Unity lifecycle methods
    // ------------------------------------------------------------------------
    void Awake()
    {
        rb = GetComponent<Rigidbody>();
        if (energyController == null)
        {
            energyController = GetComponent<EnergyController>();
        }
        if (behaviourSelector == null)
        {
            behaviourSelector = GetComponent<BehaviourSelector>();
        }
        if (metricsRecorder == null)
        {
            metricsRecorder = GetComponent<MetricsRecorder>();
        }
        if (energyController == null)
        {
            Debug.LogWarning($"[PlayerDrone] EnergyController missing on {gameObject.name}.");
        }
        if (behaviourSelector == null)
        {
            Debug.LogWarning($"[PlayerDrone] BehaviourSelector missing on {gameObject.name}.");
        }
        if (metricsRecorder == null)
        {
            Debug.LogWarning($"[PlayerDrone] MetricsRecorder missing on {gameObject.name}.");
        }
        initialPosition = transform.position;
        initialRotation = transform.rotation;
        defaultAltitude = targetAltitude;
        baseMaxLiftThrustPerThruster = maxLiftThrustPerThruster;
        baseMaxYawThrust = maxYawThrust;
        Transform waypoint = behaviourSelector != null ? behaviourSelector.CurrentWaypoint : null;
        if (waypoint != null)
        {
            waypointStartPos = waypoint.position;
        }
        // If idle centre not specified, default to the drone's initial position
        if (idleCenter == Vector3.zero)
        {
            idleCenter = initialPosition;
        }
        // Initialise the idle path generator if idle mode is enabled
        if (idleMode)
        {
            idlePath = new CircularPathGenerator(idleCenter, idleRadius, idleAngularSpeed, idleDeviationAmplitude, idleNoiseFrequency);
        }
    }

    private float lastShotTime = 0f;

    void FixedUpdate()
    {
        // Always progress the simulation timer even if no waypoint is assigned.
        float dt = Time.fixedDeltaTime;
        simulationTimer += dt;
        if (energyController != null)
        {
            energyController.Tick(dt);
        }
        if (behaviourSelector != null)
        {
            behaviourSelector.TickBehaviour(dt);
        }

        // Determine the current target.  If a waypoint is assigned use its
        // position.  Otherwise, if idle mode is enabled, follow the idle
        // circular path.  If neither is available, remain stationary.
        Vector3 targetPos;
        Transform currentWaypoint = behaviourSelector != null ? behaviourSelector.CurrentWaypoint : null;
        if (currentWaypoint != null)
        {
            targetPos = currentWaypoint.position;
        }
        else if (idleMode && idlePath != null)
        {
            targetPos = idlePath.Update(dt);
        }
        else
        {
            targetPos = new Vector3(13f,2f,18f); // hover in place
        }
        Debug.DrawLine(transform.position, targetPos);

        // Field of view check: determine if the current target lies within
        // the drone's view cone.  This can later be used to trigger behaviour
        // such as targeting or aggression.  Currently the result is
        // computed but not acted upon.
        bool targetInView = IsTargetInFieldOfView(targetPos);

        if (targetInView && lastShotTime >= .3f)
        {
            // var weaponCtrl = gameObject.GetComponentInChildren<WeaponCtrl>();
            // if (weaponCtrl != null)
            // {
            //     weaponCtrl.DoShoot();
            //     lastShotTime = 0f;
            // }
            // If the target is not in view, we could choose to hover in place
            // or implement alternative behaviour.  For now, we proceed with
            // control towards the last known target position.
        }

        lastShotTime += dt;

        // Compute control errors
        Vector3 toTarget = targetPos - transform.position;
        Vector3 obstacleVector = Vector3.zero;
        try
        {
            obstacleVector = behaviourSelector != null ? behaviourSelector.CalculateObstacleVector(transform) : Vector3.zero;
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[PlayerDrone] Obstacle vector calculation failed: {ex.Message}");
        }
        float obstacleMagnitude = obstacleVector.magnitude;
        Vector3 normalizedObstacle = obstacleMagnitude > Mathf.Epsilon ? obstacleVector / obstacleMagnitude : Vector3.zero;
        AutoTunePID obstacleWeightPID = behaviourSelector != null ? behaviourSelector.ObstacleWeightPID : null;
        AutoTunePID obstacleYawPID = behaviourSelector != null ? behaviourSelector.ObstacleYawPID : null;
        AutoTunePID obstaclePitchPID = behaviourSelector != null ? behaviourSelector.ObstaclePitchPID : null;
        float obstacleWeightError = obstacleMagnitude;
        float obstacleYawError = normalizedObstacle.sqrMagnitude > Mathf.Epsilon
            ? Vector3.SignedAngle(transform.forward, normalizedObstacle, Vector3.up) * Mathf.Deg2Rad
            : 0f;
        float obstaclePitchError = normalizedObstacle.sqrMagnitude > Mathf.Epsilon
            ? Vector3.SignedAngle(transform.forward, normalizedObstacle, transform.right) * Mathf.Deg2Rad
            : 0f;
        float obstacleWeightOutput = obstacleWeightPID != null ? obstacleWeightPID.controller.Update(obstacleWeightError, dt) : 0f;
        float obstacleYawOutput = obstacleYawPID != null ? obstacleYawPID.controller.Update(obstacleYawError, dt) : 0f;
        float obstaclePitchOutput = obstaclePitchPID != null ? obstaclePitchPID.controller.Update(obstaclePitchError, dt) : 0f;
        float obstacleWeightScale = Mathf.Clamp(obstacleWeightOutput, -1f, 1f);
        Vector3 adjustedObstacle = Vector3.zero;
        if (normalizedObstacle.sqrMagnitude > Mathf.Epsilon)
        {
            Quaternion yawRotation = Quaternion.AngleAxis(obstacleYawOutput * Mathf.Rad2Deg, Vector3.up);
            Quaternion pitchRotation = Quaternion.AngleAxis(obstaclePitchOutput * Mathf.Rad2Deg, transform.right);
            adjustedObstacle = yawRotation * pitchRotation * normalizedObstacle * obstacleMagnitude * obstacleWeightScale;
        }
        Vector3 targetDir = Vector3.zero;
        bool targetDirValid = true;
        try
        {
            targetDir = toTarget.sqrMagnitude > Mathf.Epsilon ? toTarget.normalized : Vector3.zero;
            if (float.IsNaN(targetDir.x) || float.IsNaN(targetDir.y) || float.IsNaN(targetDir.z))
            {
                targetDir = Vector3.zero;
                targetDirValid = false;
            }
        }
        catch (Exception ex)
        {
            targetDir = Vector3.zero;
            targetDirValid = false;
            Debug.LogWarning($"[PlayerDrone] Target direction normalization failed: {ex.Message}");
        }
        float waypointMagnitude = toTarget.magnitude;
        float targetInfluenceMagnitude = Mathf.Max(avoidanceMinMagnitude, Mathf.Min(waypointMagnitude, maxTargetInfluence));
        Vector3 targetInfluence = targetDir * targetInfluenceMagnitude;
        Vector3 desiredMovement = (targetInfluence + adjustedObstacle) * 0.5f;
        float movementDot = 1f;
        if (targetInfluence.sqrMagnitude > Mathf.Epsilon && adjustedObstacle.sqrMagnitude > Mathf.Epsilon)
        {
            movementDot = Vector3.Dot(targetInfluence.normalized, adjustedObstacle.normalized);
        }
        if (!Mathf.Approximately(defaultAltitude, targetAltitude))
        {
            defaultAltitude = targetAltitude;
        }
        try
        {
            if (Time.time - lastTargetLogTime >= avoidanceLogInterval)
            {
                if (!targetDirValid)
                {
                    Debug.LogWarning("[PlayerDrone] Target direction invalid; using zero vector.");
                    lastTargetLogTime = Time.time;
                }
                else if (waypointMagnitude > maxTargetInfluence)
                {
                    Debug.Log($"[PlayerDrone] Target influence clamped. Dist={waypointMagnitude:F2}, Clamp={maxTargetInfluence:F2}");
                    lastTargetLogTime = Time.time;
                }
            }
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[PlayerDrone] Target influence logging failed: {ex.Message}");
        }
        bool avoidanceTriggered = movementDot < avoidanceDotThreshold || desiredMovement.magnitude < avoidanceMinMagnitude;
        float desiredMagnitude = Mathf.Max(avoidanceMinMagnitude, (waypointMagnitude + obstacleMagnitude) * 0.5f);
        float missingMagnitude = Mathf.Max(0f, desiredMagnitude - desiredMovement.magnitude);
        dynamicAltitudeOffset = 0f;
        if (avoidanceTriggered)
        {
            dynamicAltitudeOffset = Mathf.Clamp(missingMagnitude + obstacleMagnitude * 0.5f, 0f, maxAvoidanceClimb);
            try
            {
                if (Time.time - lastAvoidanceLogTime >= avoidanceLogInterval)
                {
                    Debug.Log($"[PlayerDrone] Vertical avoidance triggered. Offset={dynamicAltitudeOffset:F2}, Dot={movementDot:F2}, ObstMag={obstacleMagnitude:F2}");
                    lastAvoidanceLogTime = Time.time;
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[PlayerDrone] Avoidance logging failed: {ex.Message}");
            }
        }
        else if (adjustedObstacle.sqrMagnitude > Mathf.Epsilon)
        {
            try
            {
                if (Time.time - lastTargetLogTime >= avoidanceLogInterval)
                {
                    Debug.Log($"[PlayerDrone] Avoidance suppressed. Dot={movementDot:F2}, TargetMag={targetInfluenceMagnitude:F2}, ObstMag={obstacleMagnitude:F2}");
                    lastTargetLogTime = Time.time;
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[PlayerDrone] Avoidance suppression logging failed: {ex.Message}");
            }
        }
        try
        {
            if (Time.time - lastAvoidancePidLogTime >= avoidanceLogInterval)
            {
                Debug.Log($"[PlayerDrone] Avoidance PID outputs: weight={obstacleWeightOutput:F2}, yaw={obstacleYawOutput:F2}, pitch={obstaclePitchOutput:F2}, vector={adjustedObstacle}");
                lastAvoidancePidLogTime = Time.time;
            }
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[PlayerDrone] Avoidance PID logging failed: {ex.Message}");
        }
        // Height error along Y
        float heightError = (defaultAltitude + dynamicAltitudeOffset) - transform.position.y;
        // Horizontal distance error in XZ plane
        Vector3 toTargetXZ = Vector3.ProjectOnPlane(toTarget, Vector3.up);
        float distanceError = toTargetXZ.magnitude - desiredDistance;
        // Yaw error as signed angle between forward and target direction (degrees)
        Vector3 forwardXZ = Vector3.ProjectOnPlane(transform.forward, Vector3.up);
        float yawErrorDeg = Vector3.SignedAngle(forwardXZ, toTargetXZ, Vector3.up);

        AutoTunePID heightPID = behaviourSelector != null ? behaviourSelector.HeightPID : null;
        AutoTunePID distancePID = behaviourSelector != null ? behaviourSelector.DistancePID : null;
        AutoTunePID pitchPID = behaviourSelector != null ? behaviourSelector.PitchPID : null;
        AutoTunePID yawPID = behaviourSelector != null ? behaviourSelector.YawPID : null;
        AutoTunePID rollPID = behaviourSelector != null ? behaviourSelector.RollPID : null;
        if (heightPID == null || distancePID == null || pitchPID == null || yawPID == null || rollPID == null || obstacleWeightPID == null || obstacleYawPID == null || obstaclePitchPID == null)
        {
            return;
        }

        // Update PID controllers via AutoTune wrappers
        float liftCmd = Mathf.Clamp01(heightPID.controller.Update(heightError, dt));
        // Distance loop: generate desired pitch command from horizontal distance error
        float forwardCmd = distancePID.controller.Update(distanceError, dt);
        // Convert forward command to desired pitch angle and clamp
        float desiredPitchRad = Mathf.Clamp(forwardCmd, -maxPitchAngleDeg * Mathf.Deg2Rad, maxPitchAngleDeg * Mathf.Deg2Rad);
        // Current pitch in radians.  Use arcsin of negative forward.y: positive when nose down, negative when nose up.
        float currentPitchRad = Mathf.Asin(Mathf.Clamp(-transform.forward.y, -1f, 1f));
        float pitchError = desiredPitchRad - currentPitchRad;
        float pitchCmd = pitchPID.controller.Update(pitchError, dt);
        float yawCmd = yawPID.controller.Update(yawErrorDeg * Mathf.Deg2Rad, dt);
        // Roll control: compute roll error (drone should remain level)
        float rollErrorRad = Mathf.Asin(Mathf.Clamp(transform.right.y, -1f, 1f));
        float rollCmd = rollPID.controller.Update(rollErrorRad, dt);
        float tiltDeg = Mathf.Max(Mathf.Abs(currentPitchRad), Mathf.Abs(rollErrorRad)) * Mathf.Rad2Deg;
        // Clamp outputs to avoid saturating thrusters and introducing jitter
        float pitchOut = Mathf.Clamp(pitchCmd, -1f, 1f);
        float rollOut = Mathf.Clamp(rollCmd, -1f, 1f);

        // Mix lift thrusters: adjust front/back with pitch and left/right with roll
        // Thruster order: 0=FL, 1=FR, 2=BL, 3=BR
        float fl = liftCmd - pitchOut + rollOut;
        float fr = liftCmd - pitchOut - rollOut;
        float bl = liftCmd + pitchOut + rollOut;
        float br = liftCmd + pitchOut - rollOut;
        float[] targets = { fl, fr, bl, br };
        // Apply smoothing and clamp
        float energyRatio = energyController != null ? energyController.EnergyRatio : 1f;
        float minEnergyMultiplier = energyController != null ? energyController.MinEnergyThrustMultiplier : 1f;
        float energyThrustMultiplier = Mathf.Lerp(minEnergyMultiplier, 1f, energyRatio);
        float maxLiftThrust = baseMaxLiftThrustPerThruster * energyThrustMultiplier;
        for (int i = 0; i < 4; i++)
        {
            float clamped = Mathf.Clamp01(targets[i]);
            liftOutputs[i] = Mathf.Lerp(liftOutputs[i], clamped, thrustSmoothing);
            // Apply thrust along the thruster’s local up direction
            Transform thruster = (i == 0 ? thrusterFL : i == 1 ? thrusterFR : i == 2 ? thrusterBL : thrusterBR);
            Vector3 liftForce = thruster.up * liftOutputs[i] * maxLiftThrust;
            rb.AddForceAtPosition(liftForce, thruster.position, ForceMode.Force);
        }

        // Apply yaw torque via side thrusters.  The thrusters are rotated so that their
        // local up axis points sideways; using .up here produces a horizontal force to
        // generate a torque about the vertical (Y) axis.  Positive yawCmd pushes in
        // opposite directions on left and right thrusters.
        float maxYaw = baseMaxYawThrust * energyThrustMultiplier;
        float yawForce = Mathf.Clamp(yawCmd, -1f, 1f) * maxYaw;
        if (yawThrusterLeft != null)
        {
            rb.AddForceAtPosition(yawThrusterLeft.up * yawForce, yawThrusterLeft.position, ForceMode.Force);
        }
        if (yawThrusterRight != null)
        {
            rb.AddForceAtPosition(-yawThrusterRight.up * yawForce, yawThrusterRight.position, ForceMode.Force);
        }

        if (metricsRecorder != null)
        {
            bool isInverted = Vector3.Dot(transform.up, Vector3.up) < 0f;
            metricsRecorder.AccumulateFrame(dt, heightError, distanceError, yawErrorDeg, tiltDeg, isInverted);
        }

        if (behaviourSelector != null)
        {
            behaviourSelector.AccumulateGradients(heightError, distanceError, pitchError, yawErrorDeg * Mathf.Deg2Rad, rollErrorRad, obstacleWeightError, obstacleYawError, obstaclePitchError, dt);
        }

        // End of episode: record metrics and reset
        float runDuration = metricsRecorder != null ? metricsRecorder.RunDuration : 0f;
        if (runDuration > 0f && simulationTimer >= runDuration)
        {
            // End of episode: record metrics and tune controllers
            if (metricsRecorder != null)
            {
                metricsRecorder.RecordMetrics(simulationTimer);
            }
            if (behaviourSelector != null)
            {
                behaviourSelector.ApplyAutoTuning(simulationTimer);
                behaviourSelector.ApplyBehaviourTraining();
            }
            // Reset for the next episode
            ResetEpisode();
        }
    }

    /// <summary>
    /// Resets the episode: clears errors, resets physical state and path variables,
    /// and randomises noise offsets so that each new episode explores slightly
    /// different conditions.
    /// </summary>
    private void ResetEpisode()
    {
        simulationTimer = 0f;
        if (metricsRecorder != null)
        {
            metricsRecorder.ResetEpisode();
        }
        // Reset physical state
        transform.position = initialPosition;
        transform.rotation = initialRotation;
        if (rb != null)
        {
            // Reset velocities to stop all motion.  Unity's Rigidbody uses
            // 'velocity' for linear motion; there is no 'linearVelocity'
            // property.  Using 'velocity' ensures the physical state is
            // correctly cleared between episodes and prevents any residual
            // momentum from affecting tuning.
            rb.linearVelocity = Vector3.zero;
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
        }
        // Reset lift outputs
        for (int i = 0; i < liftOutputs.Length; i++)
        {
            liftOutputs[i] = 0f;
        }
        // Reset idle path generator state.  If idle mode is enabled, reset
        // the circular path generator to start a new orbit with new noise offsets.
        if (idleMode && idlePath != null)
        {
            idlePath.Reset();
        }
        // If a waypoint is assigned, return it to its original position for the next
        // episode.  Only XZ are reset because the waypoint manages its own Y.
        Transform currentWaypoint = behaviourSelector != null ? behaviourSelector.CurrentWaypoint : null;
        if (currentWaypoint != null)
        {
            currentWaypoint.position = waypointStartPos;
        }
        if (behaviourSelector != null)
        {
            behaviourSelector.ResetControllers();
            behaviourSelector.ResetForEpisode();
        }
    }

    /// <summary>
    /// Determines whether a target position falls within the drone's
    /// field of view cone.  The view cone is defined by a half‑angle
    /// (viewAngle) and a maximum distance (viewDistance).  If the target
    /// lies farther than viewDistance or outside the angular cone,
    /// this returns false.  The full three‑dimensional vector between
    /// the drone and the target is used.
    /// </summary>
    /// <param name="targetPos">World position of the potential target.</param>
    /// <returns>True if the target is within view; otherwise false.</returns>
    private bool IsTargetInFieldOfView(Vector3 targetPos)
    {
        float viewDistance = behaviourSelector != null ? behaviourSelector.ViewDistance : 0f;
        float viewAngle = behaviourSelector != null ? behaviourSelector.ViewAngle : 0f;
        Vector3 toTarget = targetPos - transform.position;
        float distance = toTarget.magnitude;
        if (distance > viewDistance) return false;
        // Compute angle between forward direction and the vector to target
        // Use Vector3.Angle which returns degrees between 0 and 180
        float angle = Vector3.Angle(transform.forward, toTarget);
        return angle <= viewAngle;
    }

}

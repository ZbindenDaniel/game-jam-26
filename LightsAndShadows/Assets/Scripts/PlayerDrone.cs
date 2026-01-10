using UnityEngine;
using UnityEngine.UI;
using System;
using System.Collections.Generic;
using System.IO;


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
    // Energy and lighting settings
    // ------------------------------------------------------------------------
    [Header("Energy Settings")]
    public float maxEnergy = 100f;
    public float energy = 100f;
    public float energyDrainRate = 5f;
    public float energyRechargeRate = 10f;
    [Range(0f, 1f)]
    public float minEnergyThrustMultiplier = 0.2f;
    public Light sunLight;
    public LayerMask lightOccluderLayers = ~0;
    public float lightCheckOffset = 0.1f;
    public bool logEnergyChanges = false;
    public float energyLogInterval = 1f;

    // ------------------------------------------------------------------------
    // Target definitions
    // ------------------------------------------------------------------------
    [Header("Target")]
    public Transform currentWaypoint;
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
    // Field of view and idle path parameters
    // ------------------------------------------------------------------------
    [Header("Field of View")]
    /// <summary>The half angle of the view cone in degrees.  Targets
    /// outside this angle are considered unseen.</summary>
    public float viewAngle = 15f;
    /// <summary>The maximum distance at which the drone can see its
    /// waypoint.  Beyond this range the target is considered out of
    /// view.</summary>
    public float viewDistance = 45f;

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
    // Controllers
    // ------------------------------------------------------------------------
    [Header("Controllers")]
    //
    // Height, distance, pitch, yaw and roll controllers are now instantiated
    // privately.  Multiple parameter sets for each controller are exposed
    // below.  A behaviour‑selection neural network chooses between these
    // sets at a slower rate than the physics loop.  This supports
    // contextual behaviours such as attack, cruise and avoidance.
    private AutoTunePID heightPID;
    private AutoTunePID distancePID;
    private AutoTunePID pitchPID;
    private AutoTunePID yawPID;
    private AutoTunePID rollPID;

    // ------------------------------------------------------------------------
    // PID optimisation flags
    // ------------------------------------------------------------------------
    [Header("PID Optimisation Flags")]
    /// <summary>
    /// Enable auto‑tuning of the height PID gains.  When true, gradient
    /// descent will adjust the gains at the end of each episode.  When false,
    /// gains remain fixed.
    /// </summary>
    public bool optimiseHeightPID = false;
    /// <summary>Enable auto‑tuning of the distance PID gains.</summary>
    public bool optimiseDistancePID = false;
    /// <summary>Enable auto‑tuning of the pitch PID gains.</summary>
    public bool optimisePitchPID = false;
    /// <summary>Enable auto‑tuning of the yaw PID gains.</summary>
    public bool optimiseYawPID = false;
    /// <summary>Enable auto‑tuning of the roll PID gains.</summary>
    public bool optimiseRollPID = false;

    // ------------------------------------------------------------------------
    // Automatic tuning settings
    // ------------------------------------------------------------------------
    [Header("Auto Tuning Settings")]
    /// <summary>Enable gradient‑descent tuning of PID gains at the end of each run.</summary>
    public bool enableAutoTuning = false;
    /// <summary>Learning rate for gradient descent.  Smaller values lead to slower but more stable tuning.</summary>
    public float learningRate = 0.001f;


    // ------------------------------------------------------------------------
    // Episode and metric tracking
    // ------------------------------------------------------------------------
    [Header("Episode Settings")]
    /// <summary>
    /// Duration of one simulation episode in seconds.  After this interval
    /// the metrics are recorded, the drone is reset and a new episode begins.
    /// </summary>
    public float runDuration = 10f;
    /// <summary>Optional text component for displaying run statistics.</summary>
    public Text display;
    /// <summary>If true, metrics will be written to a JSON file upon application
    /// quit (OnApplicationQuit).  The file is created in Application.persistentDataPath
    /// with the name specified in metricsFilename.</summary>
    public bool persistMetrics = true;
    /// <summary>Name of the JSON file to write metrics to.</summary>
    public string metricsFilename = "drone_metrics.json";
    [Tooltip("Log collisions and impact severity for debugging damage signals.")]
    public bool logCollisionMetrics = false;

    // ------------------------------------------------------------------------
    // PID parameter sets and behaviour selector
    // ------------------------------------------------------------------------
    [Header("PID Parameter Sets")]
    /// <summary>
    /// Alternative parameter sets for the height PID.  Each element defines
    /// Kp, Ki and Kd gains.  A behaviour‑selection neural network chooses
    /// between these sets at runtime.  If empty, default values will be
    /// initialised in Awake().
    /// </summary>
    public PIDGains[] heightParamSets;
    /// <summary>Alternative parameter sets for the distance PID.</summary>
    public PIDGains[] distanceParamSets;
    /// <summary>Alternative parameter sets for the pitch PID.</summary>
    public PIDGains[] pitchParamSets;
    /// <summary>Alternative parameter sets for the yaw PID.</summary>
    public PIDGains[] yawParamSets;
    /// <summary>Alternative parameter sets for the roll PID.</summary>
    public PIDGains[] rollParamSets;

    [Header("Behaviour Selection Settings")]
    /// <summary>
    /// Interval in seconds between behaviour network evaluations.  At each
    /// interval the neural network is run to select new parameter sets for
    /// each PID.  Keeping this value above the physics timestep avoids
    /// excessive switching.
    /// </summary>
    public int behaviourtrainingIndex = 0;
    public float behaviourUpdateInterval = 0.5f;
    [Header("Behaviour Input Debugging")]
    public bool logBehaviourInputs = false;
    public float behaviourInputLogInterval = 1f;
    // Timer accumulating deltaTime until the next behaviour update
    private float behaviourTimer = 0f;
    private float lastBehaviourInputLogTime = float.NegativeInfinity;
    // Behaviour selector neural network instance.  It takes a vector of
    // high‑level state inputs and outputs discrete indices into the parameter
    // set arrays for each PID.
    public BehaviourNN behaviourNN;
    /// <summary>Layer mask used to identify obstacles for behaviour selection.
    /// This is separate from the field of view used for targeting.  Configure
    /// this to include objects that should contribute to the obstacles count.
    /// </summary>
    public LayerMask obstacleLayers = ~0;
    [Header("Avoidance Settings")]
    public float maxAvoidanceClimb = 3f;
    public float avoidanceDotThreshold = -0.1f;
    public float avoidanceMinMagnitude = 0.5f;
    public float avoidanceLogInterval = 1f;

    // Currently selected parameter set indices for each PID.  These
    // variables store which set was chosen during the last behaviour
    // evaluation.  They are used to update the parameter arrays when
    // auto tuning adjusts the gains.
    private int currentHeightSetIndex = 0;
    private int currentDistanceSetIndex = 0;
    private int currentPitchSetIndex = 0;
    private int currentYawSetIndex = 0;
    private int currentRollSetIndex = 0;

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
    private float lastEnergyLogTime = float.NegativeInfinity;
    private float energyRatio = 1f;
    private CanTakeShootingDamage damageReceiver;

    // Metric accumulation per episode
    private float sumSqHeightError;
    private float sumSqDistanceError;
    private float sumSqYawError;
    // Additional metrics: time spent inverted and maximum absolute errors for diagnostics
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
    private float defaultAltitude;
    private float dynamicAltitudeOffset;
    private float lastAvoidanceLogTime = float.NegativeInfinity;
    private bool loggedGoalProfileThisEpisode;

    // ------------------------------------------------------------------------
    // Goal profile settings
    // ------------------------------------------------------------------------
    [Header("Goal Profiles")]
    [Tooltip("Index into the goal profile array used to weight composite metrics.")]
    public int goalProfileIndex = 0;
    [Tooltip("Profiles defining how composite metrics are weighted.")]
    public DroneGoalProfile[] goalProfiles;
    [Tooltip("Log the active profile name/weights once per episode.")]
    public bool logGoalProfilePerEpisode = true;

    // ------------------------------------------------------------------------
    // Behaviour neural network training settings
    // ------------------------------------------------------------------------
    [Header("Behaviour NN Training")]
    /// <summary>
    /// Enable online training of the behaviour selection neural network.
    /// When true, the network collects input/target pairs during episodes
    /// and performs gradient descent updates after a specified number of
    /// episodes.  This allows the behaviour selector to adapt its
    /// weights based on observed performance without requiring offline
    /// data collection.
    /// </summary>
    public bool enableBehaviourTraining = false;
    /// <summary>Number of episodes to accumulate training data before
    /// applying an update to the neural network.  Setting this to 1
    /// will train after every episode; larger values batch updates.
    /// </summary>
    public int behaviourTrainingInterval = 1;
    /// <summary>Learning rate used when training the behaviour neural
    /// network.  Smaller values lead to more gradual weight updates.</summary>
    public float behaviourLearningRate = 0.01f;
    /// <summary>Number of epochs (full passes over the collected
    /// training data) performed each time training is invoked.</summary>
    public int behaviourEpochs = 1;
    // Accumulated training examples for the behaviour network: inputs
    // and the selected parameter indices that acted as targets.
    private readonly List<float[]> behaviourTrainingInputs = new List<float[]>();
    private readonly List<int> behaviourTrainingTargets = new List<int>();
    // Counter of episodes since the last behaviour training update
    private int episodesSinceBehaviourTraining = 0;

    // ------------------------------------------------------------------------
    // Unity lifecycle methods
    // ------------------------------------------------------------------------
    void Awake()
    {
        rb = GetComponent<Rigidbody>();
        damageReceiver = GetComponent<CanTakeShootingDamage>();
        initialPosition = transform.position;
        initialRotation = transform.rotation;
        defaultAltitude = targetAltitude;
        baseMaxLiftThrustPerThruster = maxLiftThrustPerThruster;
        baseMaxYawThrust = maxYawThrust;
        if (damageReceiver != null)
        {
            maxEnergy = damageReceiver.MaxEnergy;
            energy = damageReceiver.Energy;
        }
        if (currentWaypoint != null)
        {
            waypointStartPos = currentWaypoint.position;
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
        InitializeGoalProfiles();

        // Initialise private PID controllers for height, distance, pitch, yaw and roll.  These
        // replace the public declarations and are not exposed in the inspector.  They
        // maintain the same default gains as defined in PIDController.
        heightPID = new AutoTunePID();
        distancePID = new AutoTunePID();
        pitchPID = new AutoTunePID();
        yawPID = new AutoTunePID();
        rollPID = new AutoTunePID();

        // Assign optimise flags from inspector to each PID.  These control
        // whether gradient descent tuning is applied at the end of each
        // episode.  Without these assignments the default value of false
        // would prevent any learning.
        heightPID.optimize = optimiseHeightPID;
        distancePID.optimize = optimiseDistancePID;
        pitchPID.optimize = optimisePitchPID;
        yawPID.optimize = optimiseYawPID;
        rollPID.optimize = optimiseRollPID;
        // If no parameter sets have been provided in the inspector, create three
        // sets derived from the default gains for each controller.  The first set
        // uses the default gains, the second halves Kp and Kd, and the third
        // doubles Kp and Kd.  Integral gains are left unchanged because
        // integral action is typically disabled by default.
        int defaultSetCount = 3;
        if (heightParamSets == null || heightParamSets.Length == 0)
        {
            heightParamSets = new PIDGains[defaultSetCount];
            var def = new PIDGains(heightPID.controller.Kp, heightPID.controller.Ki, heightPID.controller.Kd);
            heightParamSets[0] = def;
            heightParamSets[1] = new PIDGains(def.Kp * 0.5f, def.Ki, def.Kd * 0.5f);
            heightParamSets[2] = new PIDGains(def.Kp * 2f, def.Ki, def.Kd * 2f);
        }
        if (distanceParamSets == null || distanceParamSets.Length == 0)
        {
            distanceParamSets = new PIDGains[defaultSetCount];
            var def = new PIDGains(distancePID.controller.Kp, distancePID.controller.Ki, distancePID.controller.Kd);
            distanceParamSets[0] = def;
            distanceParamSets[1] = new PIDGains(def.Kp * 0.5f, def.Ki, def.Kd * 0.5f);
            distanceParamSets[2] = new PIDGains(def.Kp * 2f, def.Ki, def.Kd * 2f);
        }
        if (pitchParamSets == null || pitchParamSets.Length == 0)
        {
            pitchParamSets = new PIDGains[defaultSetCount];
            var def = new PIDGains(pitchPID.controller.Kp, pitchPID.controller.Ki, pitchPID.controller.Kd);
            pitchParamSets[0] = def;
            pitchParamSets[1] = new PIDGains(def.Kp * 0.5f, def.Ki, def.Kd * 0.5f);
            pitchParamSets[2] = new PIDGains(def.Kp * 2f, def.Ki, def.Kd * 2f);
        }
        if (yawParamSets == null || yawParamSets.Length == 0)
        {
            yawParamSets = new PIDGains[defaultSetCount];
            var def = new PIDGains(yawPID.controller.Kp, yawPID.controller.Ki, yawPID.controller.Kd);
            yawParamSets[0] = def;
            yawParamSets[1] = new PIDGains(def.Kp * 0.5f, def.Ki, def.Kd * 0.5f);
            yawParamSets[2] = new PIDGains(def.Kp * 2f, def.Ki, def.Kd * 2f);
        }
        if (rollParamSets == null || rollParamSets.Length == 0)
        {
            rollParamSets = new PIDGains[defaultSetCount];
            var def = new PIDGains(rollPID.controller.Kp, rollPID.controller.Ki, rollPID.controller.Kd);
            rollParamSets[0] = def;
            rollParamSets[1] = new PIDGains(def.Kp * 0.5f, def.Ki, def.Kd * 0.5f);
            rollParamSets[2] = new PIDGains(def.Kp * 2f, def.Ki, def.Kd * 2f);
        }
        int behaviourParamCount = InitializeBehaviourNetwork();
        if (behaviourNN != null)
        {
            try
            {
                int inputCount = GetBehaviourInputs().Length;
                behaviourNN.Initialize(inputCount, behaviourParamCount);
            }
            catch (System.Exception e)
            {
                Debug.LogWarning($"BehaviourNN initialization failed on {gameObject.name}: {e.Message}");
            }
        }
        LoadTrainingState();
    }

private float lastShotTime = 0f;

    void FixedUpdate()
    {
        // Always progress the simulation timer even if no waypoint is assigned.
        float dt = Time.fixedDeltaTime;
        simulationTimer += dt;
        UpdateEnergy(dt);
        // Update the behaviour selector at a slower rate than the physics update.
        behaviourTimer += dt;
        if (behaviourTimer >= behaviourUpdateInterval)
        {
            ApplyBehaviourParameterSets();
            behaviourTimer = 0f;
        }

        // Determine the current target.  If a waypoint is assigned use its
        // position.  Otherwise, if idle mode is enabled, follow the idle
        // circular path.  If neither is available, remain stationary.
        Vector3 targetPos;
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
        Vector3 obstacleVector = CalculateObstacleVector();
        Vector3 desiredMovement = (toTarget + obstacleVector) * 0.5f;
        float obstacleMagnitude = obstacleVector.magnitude;
        float waypointMagnitude = toTarget.magnitude;
        float movementDot = 1f;
        if (toTarget.sqrMagnitude > Mathf.Epsilon && obstacleVector.sqrMagnitude > Mathf.Epsilon)
        {
            movementDot = Vector3.Dot(toTarget.normalized, obstacleVector.normalized);
        }
        if (!Mathf.Approximately(defaultAltitude, targetAltitude))
        {
            defaultAltitude = targetAltitude;
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
        // Height error along Y
        float heightError = (defaultAltitude + dynamicAltitudeOffset) - transform.position.y;
        // Horizontal distance error in XZ plane
        Vector3 toTargetXZ = Vector3.ProjectOnPlane(toTarget, Vector3.up);
        float distanceError = toTargetXZ.magnitude - desiredDistance;
        // Yaw error as signed angle between forward and target direction (degrees)
        Vector3 forwardXZ = Vector3.ProjectOnPlane(transform.forward, Vector3.up);
        float yawErrorDeg = Vector3.SignedAngle(forwardXZ, toTargetXZ, Vector3.up);

        // Track extreme errors for diagnostics
        if (Mathf.Abs(heightError) > maxAbsHeightError) maxAbsHeightError = Mathf.Abs(heightError);
        if (Mathf.Abs(distanceError) > maxAbsDistanceError) maxAbsDistanceError = Mathf.Abs(distanceError);
        if (Mathf.Abs(yawErrorDeg) > maxAbsYawError) maxAbsYawError = Mathf.Abs(yawErrorDeg);
        // Accumulate time the drone spends inverted (up vector pointing downward). Use dot product to detect orientation.
        if (Vector3.Dot(transform.up, Vector3.up) < 0f)
        {
            sumInvertedTime += dt;
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
        if (tiltDeg > maxAbsTiltDeg)
        {
            maxAbsTiltDeg = tiltDeg;
        }
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
        float energyThrustMultiplier = Mathf.Lerp(minEnergyThrustMultiplier, 1f, energyRatio);
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

        // Accumulate squared errors for metrics
        sumSqHeightError += heightError * heightError * dt;
        sumSqDistanceError += distanceError * distanceError * dt;
        sumSqYawError += yawErrorDeg * yawErrorDeg * dt;
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

        // If auto tuning is enabled, accumulate gradient information for each PID via the AutoTune wrappers
        if (enableAutoTuning)
        {
            if (heightPID.optimize) heightPID.AccumulateGradients(heightError, dt);
            if (distancePID.optimize) distancePID.AccumulateGradients(distanceError, dt);
            if (pitchPID.optimize) pitchPID.AccumulateGradients(pitchError, dt);
            if (yawPID.optimize) yawPID.AccumulateGradients(yawErrorDeg * Mathf.Deg2Rad, dt);
            if (rollPID.optimize) rollPID.AccumulateGradients(rollErrorRad, dt);
        }

        // End of episode: record metrics and reset
        if (simulationTimer >= runDuration)
        {
            // End of episode: record metrics and tune controllers
            RecordMetrics();
            // Tune PID gains via gradient descent
            ApplyAutoTuning();
            // Train behaviour neural network if enough episodes have elapsed
            ApplyBehaviourTraining();
            // Reset for the next episode
            ResetEpisode();
        }
    }

    private Vector3 CalculateObstacleVector()
    {
        Collider[] nearby = null;
        try
        {
            nearby = Physics.OverlapSphere(transform.position, viewDistance, obstacleLayers, QueryTriggerInteraction.Ignore);
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[PlayerDrone] Obstacle query failed: {ex.Message}");
            return Vector3.zero;
        }

        if (nearby == null || nearby.Length == 0)
        {
            return Vector3.zero;
        }

        Vector3 obstacleVector = Vector3.zero;
        foreach (Collider collider in nearby)
        {
            if (collider == null)
            {
                continue;
            }
            if (collider.attachedRigidbody == rb || collider.transform.IsChildOf(transform))
            {
                continue;
            }

            Vector3 toObstacle = transform.position - collider.bounds.center;
            float distance = toObstacle.magnitude;
            if (distance <= Mathf.Epsilon)
            {
                continue;
            }

            float weight = 1f / distance;
            obstacleVector += toObstacle.normalized * weight;
        }

        return obstacleVector;
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


    /// <summary>
    /// Records the metrics for the current episode, displays them and adds them to
    /// the metrics list.  The average composite error is computed as a weighted
    /// sum using the active goal profile.
    /// </summary>
    private void RecordMetrics()
    {
        float avgHeight = simulationTimer > 0f ? sumSqHeightError / simulationTimer : 0f;
        float avgDist = simulationTimer > 0f ? sumSqDistanceError / simulationTimer : 0f;
        float avgYaw = simulationTimer > 0f ? sumSqYawError / simulationTimer : 0f;
        float avgSpeed = simulationTimer > 0f ? sumSpeed / simulationTimer : 0f;
        float avgAngularVelocity = simulationTimer > 0f ? sumAngularVelocity / simulationTimer : 0f;
        float avgImpactSeverity = collisionCount > 0 ? sumImpactSeverity / collisionCount : 0f;
        // Average proportion of time the drone spent inverted during the episode
        float invertRatio = simulationTimer > 0f ? sumInvertedTime / simulationTimer : 0f;
        int activeProfileIndex = GetActiveGoalProfileIndex();
        DroneGoalProfile profile = GetActiveGoalProfile();
        LogGoalProfileIfNeeded(profile);
        float composite = (avgHeight * profile.heightWeight)
            + (avgDist * profile.distanceWeight)
            + (avgYaw * profile.yawWeight)
            + (invertRatio * profile.stabilityWeight)
            + (avgSpeed * profile.speedWeight);
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
            heightKp = heightPID.controller.Kp,
            heightKi = heightPID.controller.Ki,
            heightKd = heightPID.controller.Kd,
            distanceKp = distancePID.controller.Kp,
            distanceKi = distancePID.controller.Ki,
            distanceKd = distancePID.controller.Kd,
            pitchKp = pitchPID.controller.Kp,
            pitchKi = pitchPID.controller.Ki,
            pitchKd = pitchPID.controller.Kd,
            yawKp = yawPID.controller.Kp,
            yawKi = yawPID.controller.Ki,
            yawKd = yawPID.controller.Kd,
            // Record roll PID gains
            rollKp = rollPID.controller.Kp,
            rollKi = rollPID.controller.Ki,
            rollKd = rollPID.controller.Kd
        };
        metrics.Add(rm);
        // Update display text
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

    /// <summary>
    /// Resets the episode: clears errors, resets physical state and path variables,
    /// and randomises noise offsets so that each new episode explores slightly
    /// different conditions.
    /// </summary>
    private void ResetEpisode()
    {
        simulationTimer = 0f;
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
        if (currentWaypoint != null)
        {
            currentWaypoint.position = waypointStartPos;
        }
        // Reset PIDs internal state and gradient accumulators.  Use the
        // AutoTunePID ResetTuning method to clear integral and derivative state
        // as well as gradient sums.  This prevents carry‑over from previous
        // episodes and avoids integral wind‑up.
        heightPID.ResetTuning();
        distancePID.ResetTuning();
        pitchPID.ResetTuning();
        yawPID.ResetTuning();
        rollPID.ResetTuning();

        // Reset behaviour timer and apply the current parameter sets.  This
        // ensures that on a new episode the controller gains reflect the
        // most recent behaviour selection rather than stale values.
        behaviourTimer = 0f;
        ApplyBehaviourParameterSets();
    }

    /// <summary>
    /// When the application quits, write the collected metrics to a JSON file if
    /// persistence is enabled.  The file is stored under the persistent data
    /// path so it survives play sessions.  This method will only execute in
    /// builds or when entering play mode in the editor.
    /// </summary>
    private void OnApplicationQuit()
    {
        SaveMetricsToFile();
        SaveTrainingState();
    }

    /// <summary>
    /// OnDisable is invoked when the component is disabled or the object is
    /// destroyed.  Use this hook to persist metrics if the application is
    /// stopping without quitting completely (e.g. exiting play mode in the
    /// editor).  It calls the same persistence routine as OnApplicationQuit.
    /// </summary>
    private void OnDisable()
    {
        SaveMetricsToFile();
        SaveTrainingState();
    }

    private void UpdateEnergy(float dt)
    {
        if (damageReceiver != null)
        {
            maxEnergy = damageReceiver.MaxEnergy;
            energy = damageReceiver.Energy;
        }

        bool inSunlight = IsInSunlight();
        float energyDelta = inSunlight ? energyRechargeRate : -energyDrainRate;
        energy = Mathf.Clamp(energy + energyDelta * dt, 0f, maxEnergy);
        energyRatio = maxEnergy > Mathf.Epsilon ? Mathf.Clamp01(energy / maxEnergy) : 0f;

        if (damageReceiver != null)
        {
            damageReceiver.Energy = energy;
        }

        if (logEnergyChanges && Time.time - lastEnergyLogTime >= energyLogInterval)
        {
            try
            {
                Debug.Log($"[PlayerDrone] Energy={energy:F1}/{maxEnergy:F1} (sunlit={inSunlight})");
                lastEnergyLogTime = Time.time;
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[PlayerDrone] Energy logging failed: {ex.Message}");
            }
        }
    }

    private bool IsInSunlight()
    {
        if (sunLight == null || !sunLight.enabled)
        {
            return false;
        }

        Vector3 origin = transform.position + Vector3.up * lightCheckOffset;
        Vector3 direction;
        float maxDistance;
        if (sunLight.type == LightType.Directional)
        {
            direction = -sunLight.transform.forward;
            maxDistance = Mathf.Infinity;
        }
        else
        {
            Vector3 toLight = sunLight.transform.position - origin;
            maxDistance = toLight.magnitude;
            if (maxDistance <= Mathf.Epsilon)
            {
                return true;
            }
            direction = toLight / maxDistance;
        }

        try
        {
            RaycastHit[] hits = Physics.RaycastAll(origin, direction, maxDistance, lightOccluderLayers, QueryTriggerInteraction.Ignore);
            if (hits == null || hits.Length == 0)
            {
                return true;
            }

            Array.Sort(hits, (a, b) => a.distance.CompareTo(b.distance));
            foreach (RaycastHit hit in hits)
            {
                if (hit.collider == null)
                {
                    continue;
                }

                if (hit.collider.attachedRigidbody == rb || hit.collider.transform.IsChildOf(transform))
                {
                    continue;
                }

                return false;
            }
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[PlayerDrone] Light check failed: {ex.Message}");
            return false;
        }

        return true;
    }

    private int InitializeBehaviourNetwork()
    {
        int paramCount = Mathf.Min(heightParamSets.Length,
                                   Mathf.Min(distanceParamSets.Length,
                                             Mathf.Min(pitchParamSets.Length,
                                                       Mathf.Min(yawParamSets.Length, rollParamSets.Length))));
        if (paramCount <= 0)
        {
            behaviourNN = null;
            return 0;
        }
        if (behaviourNN == null)
        {
            Debug.LogWarning($"BehaviourNN not assigned on {gameObject.name}; persistence will skip weight loading.");
        }
        return paramCount;
    }

    private void LoadTrainingState()
    {
        string droneId = gameObject.name;
        if (PersistenceManager.TryLoadDroneIndices(droneId, out int heightIndex, out int distanceIndex, out int pitchIndex, out int yawIndex, out int rollIndex))
        {
            ApplyPersistedParameterSets(heightIndex, distanceIndex, pitchIndex, yawIndex, rollIndex);
        }
    }

    private void SaveTrainingState()
    {
        string droneId = gameObject.name;
        PersistenceManager.TrySaveDroneIndices(droneId, currentHeightSetIndex, currentDistanceSetIndex, currentPitchSetIndex, currentYawSetIndex, currentRollSetIndex);
    }

    private void ApplyPersistedParameterSets(int heightIndex, int distanceIndex, int pitchIndex, int yawIndex, int rollIndex)
    {
        ApplyPIDIndex(heightParamSets, heightPID, ref currentHeightSetIndex, heightIndex);
        ApplyPIDIndex(distanceParamSets, distancePID, ref currentDistanceSetIndex, distanceIndex);
        ApplyPIDIndex(pitchParamSets, pitchPID, ref currentPitchSetIndex, pitchIndex);
        ApplyPIDIndex(yawParamSets, yawPID, ref currentYawSetIndex, yawIndex);
        ApplyPIDIndex(rollParamSets, rollPID, ref currentRollSetIndex, rollIndex);
    }

    private void ApplyPIDIndex(PIDGains[] sets, AutoTunePID pid, ref int currentIndex, int desiredIndex)
    {
        if (sets == null || sets.Length == 0 || pid == null)
        {
            return;
        }
        int idx = Mathf.Clamp(desiredIndex, 0, sets.Length - 1);
        currentIndex = idx;
        var g = sets[idx];
        pid.controller.Kp = g.Kp;
        pid.controller.Ki = g.Ki;
        pid.controller.Kd = g.Kd;
    }

    /// <summary>
    /// Writes the collected metrics to a JSON file if persistence is enabled.
    /// A wrapper type is used because Unity's JsonUtility cannot serialise raw
    /// lists.  If writing fails an error is logged.
    /// </summary>
    private void SaveMetricsToFile()
    {
        if (!persistMetrics || metrics.Count == 0) return;
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
            Debug.Log("Metrics saved to " + path);
        }
        catch (System.Exception e)
        {
            Debug.LogError("Failed to write metrics: " + e.Message);
        }
    }

    /// <summary>
    /// Applies gradient-descent tuning to each PID controller based on the
    /// accumulated gradients during the episode.  Gains are incremented by
    /// learningRate * (gradient / episodeDuration).  Negative gains are
    /// clamped at zero to prevent destabilising the system.  After updating,
    /// all gradient accumulators and internal tracking variables are reset.
    /// </summary>
    private void ApplyAutoTuning()
    {
        if (!enableAutoTuning || simulationTimer <= 0f) return;
        // Delegate tuning to each AutoTunePID wrapper.  Each controller
        // checks its optimise flag internally.
        // Apply tuning to the currently active controllers.  After tuning,
        // propagate the new gains back into the parameter sets so that the
        // behaviour selector operates on up‑to‑date tuned parameters.
        heightPID.ApplyTuning(simulationTimer, learningRate);
        if (heightParamSets != null && heightParamSets.Length > 0)
        {
            int idx = Mathf.Clamp(currentHeightSetIndex, 0, heightParamSets.Length - 1);
            heightParamSets[idx] = new PIDGains(heightPID.controller.Kp, heightPID.controller.Ki, heightPID.controller.Kd);
        }
        distancePID.ApplyTuning(simulationTimer, learningRate);
        if (distanceParamSets != null && distanceParamSets.Length > 0)
        {
            int idx = Mathf.Clamp(currentDistanceSetIndex, 0, distanceParamSets.Length - 1);
            distanceParamSets[idx] = new PIDGains(distancePID.controller.Kp, distancePID.controller.Ki, distancePID.controller.Kd);
        }
        pitchPID.ApplyTuning(simulationTimer, learningRate);
        if (pitchParamSets != null && pitchParamSets.Length > 0)
        {
            int idx = Mathf.Clamp(currentPitchSetIndex, 0, pitchParamSets.Length - 1);
            pitchParamSets[idx] = new PIDGains(pitchPID.controller.Kp, pitchPID.controller.Ki, pitchPID.controller.Kd);
        }
        yawPID.ApplyTuning(simulationTimer, learningRate);
        if (yawParamSets != null && yawParamSets.Length > 0)
        {
            int idx = Mathf.Clamp(currentYawSetIndex, 0, yawParamSets.Length - 1);
            yawParamSets[idx] = new PIDGains(yawPID.controller.Kp, yawPID.controller.Ki, yawPID.controller.Kd);
        }
        rollPID.ApplyTuning(simulationTimer, learningRate);
        if (rollParamSets != null && rollParamSets.Length > 0)
        {
            int idx = Mathf.Clamp(currentRollSetIndex, 0, rollParamSets.Length - 1);
            rollParamSets[idx] = new PIDGains(rollPID.controller.Kp, rollPID.controller.Ki, rollPID.controller.Kd);
        }
    }

    /// <summary>
    /// Applies training to the behaviour neural network when the
    /// configured number of episodes have elapsed.  Training data
    /// collected during episodes (inputs and the selected parameter
    /// indices) are used to perform gradient descent updates on the
    /// network weights.  After training, the accumulated examples and
    /// episode counter are reset.  This is invoked from FixedUpdate
    /// after auto tuning and before resetting the episode.
    /// </summary>
    private void ApplyBehaviourTraining()
    {
        if (!enableBehaviourTraining || behaviourNN == null) return;
        episodesSinceBehaviourTraining++;
        if (episodesSinceBehaviourTraining < Mathf.Max(1, behaviourTrainingInterval))
        {
            return;
        }
        // When sufficient episodes have passed, perform training
        if (behaviourTrainingInputs.Count > 0 && behaviourTrainingTargets.Count > 0)
        {
            try
            {
                // Convert lists to arrays for training
                float[][] inputArray = behaviourTrainingInputs.ToArray();
                int[] targetArray = behaviourTrainingTargets.ToArray();
                // Train the network
                behaviourNN.Train(inputArray, targetArray, behaviourLearningRate, Mathf.Max(1, behaviourEpochs));
            }
            catch (System.Exception e)
            {
                Debug.LogError("BehaviourNN training failed: " + e.Message);
            }
        }
        // Clear training data and reset counter for the next cycle
        behaviourTrainingInputs.Clear();
        behaviourTrainingTargets.Clear();
        episodesSinceBehaviourTraining = 0;
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
        Vector3 toTarget = targetPos - transform.position;
        float distance = toTarget.magnitude;
        if (distance > viewDistance) return false;
        // Compute angle between forward direction and the vector to target
        // Use Vector3.Angle which returns degrees between 0 and 180
        float angle = Vector3.Angle(transform.forward, toTarget);
        return angle <= viewAngle;
    }

    /// <summary>
    /// Applies the parameter sets selected by the behaviour neural network to
    /// each of the PID controllers.  The network outputs an index per PID
    /// corresponding to the desired parameter set.  This method updates the
    /// controller gains accordingly.  It is invoked periodically from
    /// FixedUpdate() and also on startup.
    /// </summary>
    private void ApplyBehaviourParameterSets()
    {
        if (behaviourNN == null) return;
        // Gather high‑level inputs for behaviour selection
        float[] inputs = GetBehaviourInputs();
        int behaviourIndex = behaviourtrainingIndex;
        try
        {
            behaviourIndex = behaviourNN.SelectParamSet(inputs);
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"BehaviourNN selection failed for {name} with inputs length {(inputs == null ? 0 : inputs.Length)}: {e.Message}");
        }
        // Defensive: ensure arrays are non‑null and indices are clamped
        int countHeight = heightParamSets != null ? heightParamSets.Length : 0;
        int countDistance = distanceParamSets != null ? distanceParamSets.Length : 0;
        int countPitch = pitchParamSets != null ? pitchParamSets.Length : 0;
        int countYaw = yawParamSets != null ? yawParamSets.Length : 0;
        int countRoll = rollParamSets != null ? rollParamSets.Length : 0;
        // Height controller
        if (countHeight > 0)
        {
            int idx = Mathf.Clamp(behaviourIndex, 0, countHeight - 1);
            currentHeightSetIndex = idx;
            var g = heightParamSets[idx];
            heightPID.controller.Kp = g.Kp;
            heightPID.controller.Ki = g.Ki;
            heightPID.controller.Kd = g.Kd;
        }
        // Distance controller
        if (countDistance > 0)
        {
            int idx = Mathf.Clamp(behaviourIndex, 0, countDistance - 1);
            currentDistanceSetIndex = idx;
            var g = distanceParamSets[idx];
            distancePID.controller.Kp = g.Kp;
            distancePID.controller.Ki = g.Ki;
            distancePID.controller.Kd = g.Kd;
        }
        // Pitch controller
        if (countPitch > 0)
        {
            int idx = Mathf.Clamp(behaviourIndex, 0, countPitch - 1);
            currentPitchSetIndex = idx;
            var g = pitchParamSets[idx];
            pitchPID.controller.Kp = g.Kp;
            pitchPID.controller.Ki = g.Ki;
            pitchPID.controller.Kd = g.Kd;
        }
        // Yaw controller
        if (countYaw > 0)
        {
            int idx = Mathf.Clamp(behaviourIndex, 0, countYaw - 1);
            currentYawSetIndex = idx;
            var g = yawParamSets[idx];
            yawPID.controller.Kp = g.Kp;
            yawPID.controller.Ki = g.Ki;
            yawPID.controller.Kd = g.Kd;
        }
        // Roll controller
        if (countRoll > 0)
        {
            int idx = Mathf.Clamp(behaviourIndex, 0, countRoll - 1);
            currentRollSetIndex = idx;
            var g = rollParamSets[idx];
            rollPID.controller.Kp = g.Kp;
            rollPID.controller.Ki = g.Ki;
            rollPID.controller.Kd = g.Kd;
        }

        // If behaviour training is enabled, record the input and the target indices
        // so that the neural network can learn to reproduce these selections.  A
        // deep copy of the arrays is taken to avoid mutation of the live
        // references when the next behaviour update overwrites them.
        if (enableBehaviourTraining)
        {
            // Copy inputs and indices into new arrays for storage
            float[] inputCopy = null;
            int indexCopy = behaviourIndex;
            // Get the most recent inputs by computing them again.  Although
            // GetBehaviourInputs() is called earlier in this method, those
            // values may have been garbage collected; recomputing ensures
            // correctness without additional state.
            float[] rawInputs = GetBehaviourInputs();
            if (rawInputs != null)
            {
                inputCopy = new float[rawInputs.Length];
                rawInputs.CopyTo(inputCopy, 0);
            }
            if (inputCopy != null)
            {
                behaviourTrainingInputs.Add(inputCopy);
                behaviourTrainingTargets.Add(indexCopy);
            }
        }
    }

    /// <summary>
    /// Computes the input vector for the behaviour selection neural network.  The
    /// features included here provide a high‑level description of the
    /// environment and the drone's state: current energy, velocity,
    /// height, waypoint active flag, waypoint distance, and obstacle counts
    /// (below, same/above, forward FOV).  Additional inputs can easily be
    /// added here to enrich the behaviour selection.
    /// </summary>
    /// <returns>Array of input values matching the number of inputs expected by the neural network.</returns>
    private float[] GetBehaviourInputs()
    {
        float energyPercent = maxEnergy > Mathf.Epsilon ? Mathf.Clamp01(energy / maxEnergy) : 0f;
        float velocity = rb != null ? rb.linearVelocity.magnitude : 0f;

        // Waypoint active flag: 1 when a waypoint is set, 0 in idle mode
        float waypointActive = currentWaypoint != null ? 1f : 0f;
        float waypointDistance = currentWaypoint != null ? Vector3.Distance(transform.position, currentWaypoint.position) : 0f;

        // Height of the drone as an input feature
        float height = transform.position.y;
        int obstaclesBelow;
        int obstaclesSameOrAbove;
        int obstaclesInFov;
        ComputeObstaclePerception(out obstaclesBelow, out obstaclesSameOrAbove, out obstaclesInFov);
        
        if (logBehaviourInputs && (Time.time - lastBehaviourInputLogTime) >= behaviourInputLogInterval)
        {
            try
            {
                // Debug.Log($"Behaviour inputs ({inputs.Length}): energy={energyPercent:F2} bias={bias:F2} waypointActive={waypointActive:F0} obstacles={obstacleCount:F0} height={height:F2}");
                lastBehaviourInputLogTime = Time.time;
            }
            catch (System.Exception e)
            {
                Debug.LogWarning("Behaviour input logging failed: " + e.Message);
            }
        }
        
        return new float[]
        {
            energyPercent,
            velocity,
            height,
            waypointActive,
            waypointDistance,
            obstaclesBelow,
            obstaclesSameOrAbove,
            obstaclesInFov
        };
    }

    /// <summary>
    /// Counts the number of obstacles (or other players) beneath the drone,
    /// at the same level/above, and within the forward field of view using a
    /// spherical query.  The query is filtered by obstacleLayers.
    /// </summary>
    private void ComputeObstaclePerception(out int below, out int sameOrAbove, out int inFov)
    {
        below = 0;
        sameOrAbove = 0;
        inFov = 0;
        float heightTolerance = 0.5f;
        Collider[] hits;
        try
        {
            hits = Physics.OverlapSphere(transform.position, viewDistance, obstacleLayers, QueryTriggerInteraction.Ignore);
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"Obstacle perception query failed for {name} at {transform.position}: {e.Message}");
            return;
        }

        foreach (Collider hit in hits)
        {
            if (hit == null)
            {
                continue;
            }

            if (hit.attachedRigidbody == rb || hit.transform == transform)
            {
                continue;
            }

            Vector3 targetPos = hit.bounds.center;
            float deltaY = targetPos.y - transform.position.y;
            if (deltaY < -heightTolerance)
            {
                below++;
            }
            else
            {
                sameOrAbove++;
            }

            Vector3 toTarget = targetPos - transform.position;
            if (toTarget.sqrMagnitude > 0.0001f)
            {
                float angle = Vector3.Angle(transform.forward, toTarget);
                if (angle <= viewAngle)
                {
                    inFov++;
                }
            }
        }

        return;
    }
}

/// <summary>
/// Defines a goal profile that weights different metrics when computing a
/// composite objective score.
/// </summary>
[System.Serializable]
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

/// <summary>
/// Holds aggregated information about a single run.  The average errors are
/// stored separately for height, distance and yaw as well as the sum of these
/// for convenience.  The PID gains at the end of the run are also recorded
/// for post‑analysis.  This type is marked as Serializable so it can be
/// converted to JSON by Unity’s JsonUtility.
/// </summary>
[System.Serializable]
public class RunMetric
{
    public int runIndex;
    public float averageError;
    public float avgHeightError;
    public float avgDistanceError;
    public float avgYawError;
    public float avgSpeed;
    public float avgAngularVelocity;
    // Inverse orientation metrics: total time drone was inverted and fraction of episode inverted
    public float invertedTime;
    public float invertRatio;
    // Maximum absolute errors recorded during the episode, useful for diagnosing overshoot and instability
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
    // Roll PID gains at the end of the run
    public float rollKp, rollKi, rollKd;
}

/// <summary>
/// Wrapper class used to serialise a list of RunMetric objects.  Unity's
/// JsonUtility cannot serialise bare lists so this wrapper is required to
/// persist the metrics to disk.
/// </summary>
[System.Serializable]
public class MetricsWrapper
{
    public List<RunMetric> runs;
    public List<DroneGoalProfile> goalProfiles;
}

/// <summary>
/// Holds the proportional, integral and derivative gains for a PID controller.
/// Multiple instances of this struct are used to define alternative parameter
/// sets for each PID in the PlayerDrone.  These sets are selected at
/// runtime by a behaviour‑selection neural network.
/// </summary>
[System.Serializable]
public struct PIDGains
{
    public float Kp;
    public float Ki;
    public float Kd;
    public PIDGains(float kp, float ki, float kd)
    {
        Kp = kp;
        Ki = ki;
        Kd = kd;
    }
}

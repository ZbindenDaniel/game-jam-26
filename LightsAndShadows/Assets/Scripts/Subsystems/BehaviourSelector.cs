using System;
using System.Collections.Generic;
using UnityEngine;

public class BehaviourSelector : MonoBehaviour
{
    [Header("Dependencies")]
    [SerializeField] private Rigidbody rb;
    [SerializeField] private Transform currentWaypoint;
    [SerializeField] private EnergyController energyController;

    [Header("PID Parameter Sets")]
    [SerializeField] private PIDGains[] heightParamSets;
    [SerializeField] private PIDGains[] distanceParamSets;
    [SerializeField] private PIDGains[] pitchParamSets;
    [SerializeField] private PIDGains[] yawParamSets;
    [SerializeField] private PIDGains[] rollParamSets;
    [SerializeField] private PIDGains[] obstacleWeightParamSets;
    [SerializeField] private PIDGains[] obstacleYawParamSets;
    [SerializeField] private PIDGains[] obstaclePitchParamSets;

    [Header("Behaviour Selection Settings")]
    [SerializeField] private int behaviourtrainingIndex = 0;
    [SerializeField] private float behaviourUpdateInterval = 0.5f;
    [SerializeField] private BehaviourNN behaviourNN;
    [SerializeField] private LayerMask obstacleLayers = ~0;
    [SerializeField] private float viewAngle = 15f;
    [SerializeField] private float viewDistance = 45f;

    [Header("Behaviour Input Debugging")]
    [SerializeField] private bool logBehaviourInputs = false;
    [SerializeField] private float behaviourInputLogInterval = 1f;

    [Header("PID Optimisation Flags")]
    [SerializeField] private bool optimiseHeightPID = false;
    [SerializeField] private bool optimiseDistancePID = false;
    [SerializeField] private bool optimisePitchPID = false;
    [SerializeField] private bool optimiseYawPID = false;
    [SerializeField] private bool optimiseRollPID = false;
    [SerializeField] private bool optimiseObstacleWeightPID = false;
    [SerializeField] private bool optimiseObstacleYawPID = false;
    [SerializeField] private bool optimiseObstaclePitchPID = false;

    [Header("Auto Tuning Settings")]
    [SerializeField] private bool enableAutoTuning = false;
    [SerializeField] private float learningRate = 0.001f;

    [Header("Behaviour NN Training")]
    [SerializeField] private bool enableBehaviourTraining = false;
    [SerializeField] private int behaviourTrainingInterval = 1;
    [SerializeField] private float behaviourLearningRate = 0.01f;
    [SerializeField] private int behaviourEpochs = 1;

    [SerializeField] private AutoTunePID heightPID;
    [SerializeField] private AutoTunePID distancePID;
    [SerializeField] private AutoTunePID pitchPID;
    [SerializeField] private AutoTunePID yawPID;
    [SerializeField] private AutoTunePID rollPID;
    [SerializeField] private AutoTunePID obstacleWeightPID;
    [SerializeField] private AutoTunePID obstacleYawPID;
    [SerializeField] private AutoTunePID obstaclePitchPID;

    private float behaviourTimer = 0f;
    private float lastBehaviourInputLogTime = float.NegativeInfinity;
    private int currentHeightSetIndex = 0;
    private int currentDistanceSetIndex = 0;
    private int currentPitchSetIndex = 0;
    private int currentYawSetIndex = 0;
    private int currentRollSetIndex = 0;
    private int currentObstacleWeightSetIndex = 0;
    private int currentObstacleYawSetIndex = 0;
    private int currentObstaclePitchSetIndex = 0;
    private readonly List<float[]> behaviourTrainingInputs = new List<float[]>();
    private readonly List<int> behaviourTrainingTargets = new List<int>();
    private int episodesSinceBehaviourTraining = 0;

    public AutoTunePID HeightPID => heightPID;
    public AutoTunePID DistancePID => distancePID;
    public AutoTunePID PitchPID => pitchPID;
    public AutoTunePID YawPID => yawPID;
    public AutoTunePID RollPID => rollPID;
    public AutoTunePID ObstacleWeightPID => obstacleWeightPID;
    public AutoTunePID ObstacleYawPID => obstacleYawPID;
    public AutoTunePID ObstaclePitchPID => obstaclePitchPID;
    public float BehaviourUpdateInterval => behaviourUpdateInterval;
    public float ViewAngle => viewAngle;
    public float ViewDistance => viewDistance;
    public Transform CurrentWaypoint => currentWaypoint;

    private void Awake()
    {
        if (rb == null)
        {
            rb = GetComponent<Rigidbody>();
        }

        if (energyController == null)
        {
            energyController = GetComponent<EnergyController>();
        }

        if (heightPID == null) heightPID = new AutoTunePID();
        if (distancePID == null) distancePID = new AutoTunePID();
        if (pitchPID == null) pitchPID = new AutoTunePID();
        if (yawPID == null) yawPID = new AutoTunePID();
        if (rollPID == null) rollPID = new AutoTunePID();
        if (obstacleWeightPID == null) obstacleWeightPID = new AutoTunePID();
        if (obstacleYawPID == null) obstacleYawPID = new AutoTunePID();
        if (obstaclePitchPID == null) obstaclePitchPID = new AutoTunePID();

        heightPID.optimize = optimiseHeightPID;
        distancePID.optimize = optimiseDistancePID;
        pitchPID.optimize = optimisePitchPID;
        yawPID.optimize = optimiseYawPID;
        rollPID.optimize = optimiseRollPID;
        obstacleWeightPID.optimize = optimiseObstacleWeightPID;
        obstacleYawPID.optimize = optimiseObstacleYawPID;
        obstaclePitchPID.optimize = optimiseObstaclePitchPID;

        InitializeDefaultParamSets();
        int behaviourParamCount = InitializeBehaviourNetwork();
        if (behaviourNN != null)
        {
            try
            {
                int inputCount = GetBehaviourInputs().Length;
                behaviourNN.Initialize(inputCount, behaviourParamCount);
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[PlayerDrone] BehaviourNN initialization failed on {gameObject.name}: {e.Message}");
            }
        }
        LoadTrainingState();
    }

    public void TickBehaviour(float dt)
    {
        behaviourTimer += dt;
        if (behaviourTimer >= behaviourUpdateInterval)
        {
            ApplyBehaviourParameterSets();
            behaviourTimer = 0f;
        }
    }

    public void AccumulateGradients(float heightError, float distanceError, float pitchError, float yawError, float rollError, float obstacleWeightError, float obstacleYawError, float obstaclePitchError, float dt)
    {
        if (!enableAutoTuning)
        {
            return;
        }

        if (heightPID != null && heightPID.optimize)
        {
            heightPID.AccumulateGradients(heightError, dt);
        }
        if (distancePID != null && distancePID.optimize)
        {
            distancePID.AccumulateGradients(distanceError, dt);
        }
        if (pitchPID != null && pitchPID.optimize)
        {
            pitchPID.AccumulateGradients(pitchError, dt);
        }
        if (yawPID != null && yawPID.optimize)
        {
            yawPID.AccumulateGradients(yawError, dt);
        }
        if (rollPID != null && rollPID.optimize)
        {
            rollPID.AccumulateGradients(rollError, dt);
        }
        if (obstacleWeightPID != null && obstacleWeightPID.optimize)
        {
            obstacleWeightPID.AccumulateGradients(obstacleWeightError, dt);
        }
        if (obstacleYawPID != null && obstacleYawPID.optimize)
        {
            obstacleYawPID.AccumulateGradients(obstacleYawError, dt);
        }
        if (obstaclePitchPID != null && obstaclePitchPID.optimize)
        {
            obstaclePitchPID.AccumulateGradients(obstaclePitchError, dt);
        }
    }

    public void ResetForEpisode()
    {
        behaviourTimer = 0f;
        ApplyBehaviourParameterSets();
    }

    public void ResetControllers()
    {
        heightPID?.ResetTuning();
        distancePID?.ResetTuning();
        pitchPID?.ResetTuning();
        yawPID?.ResetTuning();
        rollPID?.ResetTuning();
        obstacleWeightPID?.ResetTuning();
        obstacleYawPID?.ResetTuning();
        obstaclePitchPID?.ResetTuning();
    }

    public void ApplyAutoTuning(float episodeDuration)
    {
        if (!enableAutoTuning || episodeDuration <= 0f)
        {
            return;
        }

        heightPID.ApplyTuning(episodeDuration, learningRate);
        if (heightParamSets != null && heightParamSets.Length > 0)
        {
            int idx = Mathf.Clamp(currentHeightSetIndex, 0, heightParamSets.Length - 1);
            heightParamSets[idx] = new PIDGains(heightPID.controller.Kp, heightPID.controller.Ki, heightPID.controller.Kd);
        }
        distancePID.ApplyTuning(episodeDuration, learningRate);
        if (distanceParamSets != null && distanceParamSets.Length > 0)
        {
            int idx = Mathf.Clamp(currentDistanceSetIndex, 0, distanceParamSets.Length - 1);
            distanceParamSets[idx] = new PIDGains(distancePID.controller.Kp, distancePID.controller.Ki, distancePID.controller.Kd);
        }
        pitchPID.ApplyTuning(episodeDuration, learningRate);
        if (pitchParamSets != null && pitchParamSets.Length > 0)
        {
            int idx = Mathf.Clamp(currentPitchSetIndex, 0, pitchParamSets.Length - 1);
            pitchParamSets[idx] = new PIDGains(pitchPID.controller.Kp, pitchPID.controller.Ki, pitchPID.controller.Kd);
        }
        yawPID.ApplyTuning(episodeDuration, learningRate);
        if (yawParamSets != null && yawParamSets.Length > 0)
        {
            int idx = Mathf.Clamp(currentYawSetIndex, 0, yawParamSets.Length - 1);
            yawParamSets[idx] = new PIDGains(yawPID.controller.Kp, yawPID.controller.Ki, yawPID.controller.Kd);
        }
        rollPID.ApplyTuning(episodeDuration, learningRate);
        if (rollParamSets != null && rollParamSets.Length > 0)
        {
            int idx = Mathf.Clamp(currentRollSetIndex, 0, rollParamSets.Length - 1);
            rollParamSets[idx] = new PIDGains(rollPID.controller.Kp, rollPID.controller.Ki, rollPID.controller.Kd);
        }
        obstacleWeightPID.ApplyTuning(episodeDuration, learningRate);
        if (obstacleWeightParamSets != null && obstacleWeightParamSets.Length > 0)
        {
            int idx = Mathf.Clamp(currentObstacleWeightSetIndex, 0, obstacleWeightParamSets.Length - 1);
            obstacleWeightParamSets[idx] = new PIDGains(obstacleWeightPID.controller.Kp, obstacleWeightPID.controller.Ki, obstacleWeightPID.controller.Kd);
        }
        obstacleYawPID.ApplyTuning(episodeDuration, learningRate);
        if (obstacleYawParamSets != null && obstacleYawParamSets.Length > 0)
        {
            int idx = Mathf.Clamp(currentObstacleYawSetIndex, 0, obstacleYawParamSets.Length - 1);
            obstacleYawParamSets[idx] = new PIDGains(obstacleYawPID.controller.Kp, obstacleYawPID.controller.Ki, obstacleYawPID.controller.Kd);
        }
        obstaclePitchPID.ApplyTuning(episodeDuration, learningRate);
        if (obstaclePitchParamSets != null && obstaclePitchParamSets.Length > 0)
        {
            int idx = Mathf.Clamp(currentObstaclePitchSetIndex, 0, obstaclePitchParamSets.Length - 1);
            obstaclePitchParamSets[idx] = new PIDGains(obstaclePitchPID.controller.Kp, obstaclePitchPID.controller.Ki, obstaclePitchPID.controller.Kd);
        }
    }

    public void ApplyBehaviourTraining()
    {
        if (!enableBehaviourTraining || behaviourNN == null)
        {
            return;
        }

        episodesSinceBehaviourTraining++;
        if (episodesSinceBehaviourTraining < Mathf.Max(1, behaviourTrainingInterval))
        {
            return;
        }

        if (behaviourTrainingInputs.Count > 0 && behaviourTrainingTargets.Count > 0)
        {
            try
            {
                float[][] inputArray = behaviourTrainingInputs.ToArray();
                int[] targetArray = behaviourTrainingTargets.ToArray();
                behaviourNN.Train(inputArray, targetArray, behaviourLearningRate, Mathf.Max(1, behaviourEpochs));
            }
            catch (Exception e)
            {
                Debug.LogError($"[PlayerDrone] BehaviourNN training failed: {e.Message}");
            }
        }
        behaviourTrainingInputs.Clear();
        behaviourTrainingTargets.Clear();
        episodesSinceBehaviourTraining = 0;
    }

    public Vector3 CalculateObstacleVector(Transform droneTransform)
    {
        if (droneTransform == null)
        {
            return Vector3.zero;
        }

        Collider[] nearby;
        try
        {
            nearby = Physics.OverlapSphere(droneTransform.position, viewDistance, obstacleLayers, QueryTriggerInteraction.Ignore);
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

            if (collider.attachedRigidbody == rb || collider.transform.IsChildOf(droneTransform))
            {
                continue;
            }

            Vector3 toObstacle = droneTransform.position - collider.bounds.center;
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

    private void OnApplicationQuit()
    {
        SaveTrainingState();
    }

    private void OnDisable()
    {
        SaveTrainingState();
    }

    private void InitializeDefaultParamSets()
    {
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
        if (obstacleWeightParamSets == null || obstacleWeightParamSets.Length == 0)
        {
            obstacleWeightParamSets = new PIDGains[defaultSetCount];
            var def = new PIDGains(obstacleWeightPID.controller.Kp, obstacleWeightPID.controller.Ki, obstacleWeightPID.controller.Kd);
            obstacleWeightParamSets[0] = def;
            obstacleWeightParamSets[1] = new PIDGains(def.Kp * 0.5f, def.Ki, def.Kd * 0.5f);
            obstacleWeightParamSets[2] = new PIDGains(def.Kp * 2f, def.Ki, def.Kd * 2f);
        }
        if (obstacleYawParamSets == null || obstacleYawParamSets.Length == 0)
        {
            obstacleYawParamSets = new PIDGains[defaultSetCount];
            var def = new PIDGains(obstacleYawPID.controller.Kp, obstacleYawPID.controller.Ki, obstacleYawPID.controller.Kd);
            obstacleYawParamSets[0] = def;
            obstacleYawParamSets[1] = new PIDGains(def.Kp * 0.5f, def.Ki, def.Kd * 0.5f);
            obstacleYawParamSets[2] = new PIDGains(def.Kp * 2f, def.Ki, def.Kd * 2f);
        }
        if (obstaclePitchParamSets == null || obstaclePitchParamSets.Length == 0)
        {
            obstaclePitchParamSets = new PIDGains[defaultSetCount];
            var def = new PIDGains(obstaclePitchPID.controller.Kp, obstaclePitchPID.controller.Ki, obstaclePitchPID.controller.Kd);
            obstaclePitchParamSets[0] = def;
            obstaclePitchParamSets[1] = new PIDGains(def.Kp * 0.5f, def.Ki, def.Kd * 0.5f);
            obstaclePitchParamSets[2] = new PIDGains(def.Kp * 2f, def.Ki, def.Kd * 2f);
        }
    }

    private int InitializeBehaviourNetwork()
    {
        int paramCount = Mathf.Min(heightParamSets.Length,
            Mathf.Min(distanceParamSets.Length,
                Mathf.Min(pitchParamSets.Length,
                    Mathf.Min(yawParamSets.Length,
                        Mathf.Min(rollParamSets.Length,
                            Mathf.Min(obstacleWeightParamSets.Length,
                                Mathf.Min(obstacleYawParamSets.Length, obstaclePitchParamSets.Length)))))));
        if (paramCount <= 0)
        {
            behaviourNN = null;
            return 0;
        }
        if (behaviourNN == null)
        {
            Debug.LogWarning($"[PlayerDrone] BehaviourNN not assigned on {gameObject.name}; persistence will skip weight loading.");
        }
        return paramCount;
    }

    private void LoadTrainingState()
    {
        string droneId = gameObject.name;
        if (PersistenceManager.TryLoadDroneIndices(droneId, out int heightIndex, out int distanceIndex, out int pitchIndex, out int yawIndex, out int rollIndex, out int obstacleWeightIndex, out int obstacleYawIndex, out int obstaclePitchIndex))
        {
            ApplyPersistedParameterSets(heightIndex, distanceIndex, pitchIndex, yawIndex, rollIndex, obstacleWeightIndex, obstacleYawIndex, obstaclePitchIndex);
        }
    }

    private void SaveTrainingState()
    {
        string droneId = gameObject.name;
        PersistenceManager.TrySaveDroneIndices(droneId, currentHeightSetIndex, currentDistanceSetIndex, currentPitchSetIndex, currentYawSetIndex, currentRollSetIndex, currentObstacleWeightSetIndex, currentObstacleYawSetIndex, currentObstaclePitchSetIndex);
    }

    private void ApplyPersistedParameterSets(int heightIndex, int distanceIndex, int pitchIndex, int yawIndex, int rollIndex, int obstacleWeightIndex, int obstacleYawIndex, int obstaclePitchIndex)
    {
        ApplyPIDIndex(heightParamSets, heightPID, ref currentHeightSetIndex, heightIndex);
        ApplyPIDIndex(distanceParamSets, distancePID, ref currentDistanceSetIndex, distanceIndex);
        ApplyPIDIndex(pitchParamSets, pitchPID, ref currentPitchSetIndex, pitchIndex);
        ApplyPIDIndex(yawParamSets, yawPID, ref currentYawSetIndex, yawIndex);
        ApplyPIDIndex(rollParamSets, rollPID, ref currentRollSetIndex, rollIndex);
        ApplyPIDIndex(obstacleWeightParamSets, obstacleWeightPID, ref currentObstacleWeightSetIndex, obstacleWeightIndex);
        ApplyPIDIndex(obstacleYawParamSets, obstacleYawPID, ref currentObstacleYawSetIndex, obstacleYawIndex);
        ApplyPIDIndex(obstaclePitchParamSets, obstaclePitchPID, ref currentObstaclePitchSetIndex, obstaclePitchIndex);
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

    public void ApplyBehaviourParameterSets()
    {
        if (behaviourNN == null)
        {
            return;
        }

        float[] inputs = GetBehaviourInputs();
        int behaviourIndex = behaviourtrainingIndex;
        try
        {
            behaviourIndex = behaviourNN.SelectParamSet(inputs);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[PlayerDrone] BehaviourNN selection failed for {name} with inputs length {(inputs == null ? 0 : inputs.Length)}: {e.Message}");
        }

        int countHeight = heightParamSets != null ? heightParamSets.Length : 0;
        int countDistance = distanceParamSets != null ? distanceParamSets.Length : 0;
        int countPitch = pitchParamSets != null ? pitchParamSets.Length : 0;
        int countYaw = yawParamSets != null ? yawParamSets.Length : 0;
        int countRoll = rollParamSets != null ? rollParamSets.Length : 0;
        int countObstacleWeight = obstacleWeightParamSets != null ? obstacleWeightParamSets.Length : 0;
        int countObstacleYaw = obstacleYawParamSets != null ? obstacleYawParamSets.Length : 0;
        int countObstaclePitch = obstaclePitchParamSets != null ? obstaclePitchParamSets.Length : 0;
        if (countHeight > 0)
        {
            int idx = Mathf.Clamp(behaviourIndex, 0, countHeight - 1);
            currentHeightSetIndex = idx;
            var g = heightParamSets[idx];
            heightPID.controller.Kp = g.Kp;
            heightPID.controller.Ki = g.Ki;
            heightPID.controller.Kd = g.Kd;
        }
        if (countDistance > 0)
        {
            int idx = Mathf.Clamp(behaviourIndex, 0, countDistance - 1);
            currentDistanceSetIndex = idx;
            var g = distanceParamSets[idx];
            distancePID.controller.Kp = g.Kp;
            distancePID.controller.Ki = g.Ki;
            distancePID.controller.Kd = g.Kd;
        }
        if (countPitch > 0)
        {
            int idx = Mathf.Clamp(behaviourIndex, 0, countPitch - 1);
            currentPitchSetIndex = idx;
            var g = pitchParamSets[idx];
            pitchPID.controller.Kp = g.Kp;
            pitchPID.controller.Ki = g.Ki;
            pitchPID.controller.Kd = g.Kd;
        }
        if (countYaw > 0)
        {
            int idx = Mathf.Clamp(behaviourIndex, 0, countYaw - 1);
            currentYawSetIndex = idx;
            var g = yawParamSets[idx];
            yawPID.controller.Kp = g.Kp;
            yawPID.controller.Ki = g.Ki;
            yawPID.controller.Kd = g.Kd;
        }
        if (countRoll > 0)
        {
            int idx = Mathf.Clamp(behaviourIndex, 0, countRoll - 1);
            currentRollSetIndex = idx;
            var g = rollParamSets[idx];
            rollPID.controller.Kp = g.Kp;
            rollPID.controller.Ki = g.Ki;
            rollPID.controller.Kd = g.Kd;
        }
        if (countObstacleWeight > 0)
        {
            int idx = Mathf.Clamp(behaviourIndex, 0, countObstacleWeight - 1);
            currentObstacleWeightSetIndex = idx;
            var g = obstacleWeightParamSets[idx];
            obstacleWeightPID.controller.Kp = g.Kp;
            obstacleWeightPID.controller.Ki = g.Ki;
            obstacleWeightPID.controller.Kd = g.Kd;
        }
        if (countObstacleYaw > 0)
        {
            int idx = Mathf.Clamp(behaviourIndex, 0, countObstacleYaw - 1);
            currentObstacleYawSetIndex = idx;
            var g = obstacleYawParamSets[idx];
            obstacleYawPID.controller.Kp = g.Kp;
            obstacleYawPID.controller.Ki = g.Ki;
            obstacleYawPID.controller.Kd = g.Kd;
        }
        if (countObstaclePitch > 0)
        {
            int idx = Mathf.Clamp(behaviourIndex, 0, countObstaclePitch - 1);
            currentObstaclePitchSetIndex = idx;
            var g = obstaclePitchParamSets[idx];
            obstaclePitchPID.controller.Kp = g.Kp;
            obstaclePitchPID.controller.Ki = g.Ki;
            obstaclePitchPID.controller.Kd = g.Kd;
        }

        if (enableBehaviourTraining)
        {
            float[] inputCopy = null;
            int indexCopy = behaviourIndex;
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

    public float[] GetBehaviourInputs()
    {
        float maxEnergy = energyController != null ? energyController.MaxEnergy : 0f;
        float energyPercent = maxEnergy > Mathf.Epsilon && energyController != null
            ? Mathf.Clamp01(energyController.Energy / maxEnergy)
            : 0f;
        float velocity = rb != null ? rb.linearVelocity.magnitude : 0f;

        float waypointActive = currentWaypoint != null ? 1f : 0f;
        float waypointDistance = currentWaypoint != null ? Vector3.Distance(transform.position, currentWaypoint.position) : 0f;

        float height = transform.position.y;
        int obstaclesBelow;
        int obstaclesSameOrAbove;
        int obstaclesInFov;
        ComputeObstaclePerception(out obstaclesBelow, out obstaclesSameOrAbove, out obstaclesInFov);

        if (logBehaviourInputs && (Time.time - lastBehaviourInputLogTime) >= behaviourInputLogInterval)
        {
            try
            {
                lastBehaviourInputLogTime = Time.time;
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[PlayerDrone] Behaviour input logging failed: {e.Message}");
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

    public void ComputeObstaclePerception(out int below, out int sameOrAbove, out int inFov)
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
        catch (Exception e)
        {
            Debug.LogWarning($"[PlayerDrone] Obstacle perception query failed for {name} at {transform.position}: {e.Message}");
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
    }
}

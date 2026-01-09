using System.IO;
using UnityEngine;

public static class PersistenceManager
{
    private const string TrainingDataFolder = "Assets/TrainingData";

    [System.Serializable]
    private class DronePersistenceData
    {
        public int heightIndex;
        public int distanceIndex;
        public int pitchIndex;
        public int yawIndex;
        public int rollIndex;
    }

    [System.Serializable]
    private class BehaviourPersistenceData
    {
        public int inputCount;
        public int paramCount;
        public float[] weights;
        public float bias;
    }

    public static bool TryLoadDroneIndices(string droneId, out int heightIndex, out int distanceIndex, out int pitchIndex, out int yawIndex, out int rollIndex)
    {
        heightIndex = 0;
        distanceIndex = 0;
        pitchIndex = 0;
        yawIndex = 0;
        rollIndex = 0;

        string path = GetDronePath(droneId);
        if (!File.Exists(path))
        {
            Debug.Log($"Drone persistence load skipped: no data at {path} for drone {droneId}.");
            return false;
        }

        try
        {
            string json = File.ReadAllText(path);
            DronePersistenceData data = JsonUtility.FromJson<DronePersistenceData>(json);
            if (data == null)
            {
                Debug.LogWarning($"Drone persistence load failed: invalid JSON at {path} for drone {droneId}.");
                return false;
            }

            heightIndex = data.heightIndex;
            distanceIndex = data.distanceIndex;
            pitchIndex = data.pitchIndex;
            yawIndex = data.yawIndex;
            rollIndex = data.rollIndex;

            Debug.Log($"Drone persistence load succeeded at {path} for drone {droneId}.");
            return true;
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Drone persistence load failed at {path} for drone {droneId}: {e.Message}");
            return false;
        }
    }

    public static bool TrySaveDroneIndices(string droneId, int heightIndex, int distanceIndex, int pitchIndex, int yawIndex, int rollIndex)
    {
        string path = GetDronePath(droneId);
        try
        {
            EnsureTrainingFolderExists();
            DronePersistenceData data = new DronePersistenceData
            {
                heightIndex = heightIndex,
                distanceIndex = distanceIndex,
                pitchIndex = pitchIndex,
                yawIndex = yawIndex,
                rollIndex = rollIndex
            };

            string json = JsonUtility.ToJson(data, true);
            File.WriteAllText(path, json);
            Debug.Log($"Drone persistence save succeeded at {path} for drone {droneId}.");
            return true;
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Drone persistence save failed at {path} for drone {droneId}: {e.Message}");
            return false;
        }
    }

    public static bool TryLoadBehaviourWeights(string behaviourId, BehaviourNN behaviour)
    {
        if (behaviour == null)
        {
            Debug.LogWarning($"Behaviour persistence load skipped: missing BehaviourNN for {behaviourId}.");
            return false;
        }

        string path = GetBehaviourPath(behaviourId);
        if (!File.Exists(path))
        {
            Debug.Log($"Behaviour persistence load skipped: no data at {path} for {behaviourId}.");
            return false;
        }

        try
        {
            string json = File.ReadAllText(path);
            BehaviourPersistenceData data = JsonUtility.FromJson<BehaviourPersistenceData>(json);
            if (data == null)
            {
                Debug.LogWarning($"Behaviour persistence load failed: invalid JSON at {path} for {behaviourId}.");
                return false;
            }
            if (data.inputCount != behaviour.InputCount || data.paramCount != behaviour.ParamCount)
            {
                Debug.LogWarning($"Behaviour persistence load discarded: shape mismatch at {path} for {behaviourId} (inputs {data.inputCount} vs {behaviour.InputCount}, params {data.paramCount} vs {behaviour.ParamCount}).");
                return false;
            }
            if (data.weights == null || data.weights.Length != behaviour.InputCount)
            {
                Debug.LogWarning($"Behaviour persistence load discarded: weight length mismatch at {path} for {behaviourId}.");
                return false;
            }

            if (!behaviour.TryApplyWeights(data.weights, data.bias))
            {
                Debug.LogWarning($"Behaviour persistence load discarded: failed to apply weights at {path} for {behaviourId}.");
                return false;
            }

            Debug.Log($"Behaviour persistence load succeeded at {path} for {behaviourId}.");
            return true;
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Behaviour persistence load failed at {path} for {behaviourId}: {e.Message}");
            return false;
        }
    }

    public static bool TrySaveBehaviourWeights(string behaviourId, BehaviourNN behaviour)
    {
        if (behaviour == null)
        {
            Debug.LogWarning($"Behaviour persistence save skipped: missing BehaviourNN for {behaviourId}.");
            return false;
        }

        string path = GetBehaviourPath(behaviourId);
        try
        {
            EnsureTrainingFolderExists();
            BehaviourPersistenceData data = new BehaviourPersistenceData
            {
                inputCount = behaviour.InputCount,
                paramCount = behaviour.ParamCount,
                weights = behaviour.GetWeightsCopy(),
                bias = behaviour.Bias
            };

            string json = JsonUtility.ToJson(data, true);
            File.WriteAllText(path, json);
            Debug.Log($"Behaviour persistence save succeeded at {path} for {behaviourId}.");
            return true;
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Behaviour persistence save failed at {path} for {behaviourId}: {e.Message}");
            return false;
        }
    }

    private static void EnsureTrainingFolderExists()
    {
        if (!Directory.Exists(TrainingDataFolder))
        {
            Directory.CreateDirectory(TrainingDataFolder);
        }
    }

    private static string GetDronePath(string droneId)
    {
        string safeId = string.IsNullOrWhiteSpace(droneId) ? "drone" : SanitizeFileName(droneId);
        return Path.Combine(TrainingDataFolder, $"drone_{safeId}_pid.json");
    }

    private static string GetBehaviourPath(string behaviourId)
    {
        string safeId = string.IsNullOrWhiteSpace(behaviourId) ? "behaviour" : SanitizeFileName(behaviourId);
        return Path.Combine(TrainingDataFolder, $"behaviour_{safeId}_weights.json");
    }

    private static string SanitizeFileName(string input)
    {
        char[] invalid = Path.GetInvalidFileNameChars();
        foreach (char c in invalid)
        {
            input = input.Replace(c, '_');
        }
        return input;
    }
}

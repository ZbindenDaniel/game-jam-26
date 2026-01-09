using System.IO;
using UnityEngine;

public static class PersistenceManager
{
    private const string TrainingDataFolder = "Assets/TrainingData";

    [System.Serializable]
    private class DronePersistenceData
    {
        public int inputCount;
        public int paramCount;
        public float[] weights;
        public float bias;
        public int heightIndex;
        public int distanceIndex;
        public int pitchIndex;
        public int yawIndex;
        public int rollIndex;
    }

    public static bool TryLoad(string droneId, BehaviourNN behaviour, out int heightIndex, out int distanceIndex, out int pitchIndex, out int yawIndex, out int rollIndex)
    {
        heightIndex = 0;
        distanceIndex = 0;
        pitchIndex = 0;
        yawIndex = 0;
        rollIndex = 0;

        if (behaviour == null)
        {
            Debug.LogWarning($"Persistence load skipped: missing BehaviourNN for drone {droneId}.");
            return false;
        }

        string path = GetPath(droneId);
        if (!File.Exists(path))
        {
            Debug.Log($"Persistence load skipped: no data at {path} for drone {droneId}.");
            return false;
        }

        try
        {
            string json = File.ReadAllText(path);
            DronePersistenceData data = JsonUtility.FromJson<DronePersistenceData>(json);
            if (data == null)
            {
                Debug.LogWarning($"Persistence load failed: invalid JSON at {path} for drone {droneId}.");
                return false;
            }
            if (data.inputCount != behaviour.InputCount || data.paramCount != behaviour.ParamCount)
            {
                Debug.LogWarning($"Persistence load discarded: shape mismatch at {path} for drone {droneId} (inputs {data.inputCount} vs {behaviour.InputCount}, params {data.paramCount} vs {behaviour.ParamCount}).");
                return false;
            }
            if (data.weights == null || data.weights.Length != behaviour.InputCount)
            {
                Debug.LogWarning($"Persistence load discarded: weight length mismatch at {path} for drone {droneId}.");
                return false;
            }

            if (!behaviour.TryApplyWeights(data.weights, data.bias))
            {
                Debug.LogWarning($"Persistence load discarded: failed to apply weights at {path} for drone {droneId}.");
                return false;
            }

            heightIndex = data.heightIndex;
            distanceIndex = data.distanceIndex;
            pitchIndex = data.pitchIndex;
            yawIndex = data.yawIndex;
            rollIndex = data.rollIndex;

            Debug.Log($"Persistence load succeeded at {path} for drone {droneId}.");
            return true;
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Persistence load failed at {path} for drone {droneId}: {e.Message}");
            return false;
        }
    }

    public static bool TrySave(string droneId, BehaviourNN behaviour, int heightIndex, int distanceIndex, int pitchIndex, int yawIndex, int rollIndex)
    {
        if (behaviour == null)
        {
            Debug.LogWarning($"Persistence save skipped: missing BehaviourNN for drone {droneId}.");
            return false;
        }

        string path = GetPath(droneId);
        try
        {
            EnsureTrainingFolderExists();
            DronePersistenceData data = new DronePersistenceData
            {
                inputCount = behaviour.InputCount,
                paramCount = behaviour.ParamCount,
                weights = behaviour.GetWeightsCopy(),
                bias = behaviour.Bias,
                heightIndex = heightIndex,
                distanceIndex = distanceIndex,
                pitchIndex = pitchIndex,
                yawIndex = yawIndex,
                rollIndex = rollIndex
            };

            string json = JsonUtility.ToJson(data, true);
            File.WriteAllText(path, json);
            Debug.Log($"Persistence save succeeded at {path} for drone {droneId}.");
            return true;
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Persistence save failed at {path} for drone {droneId}: {e.Message}");
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

    private static string GetPath(string droneId)
    {
        string safeId = string.IsNullOrWhiteSpace(droneId) ? "drone" : SanitizeFileName(droneId);
        return Path.Combine(TrainingDataFolder, $"drone_{safeId}_persistence.json");
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

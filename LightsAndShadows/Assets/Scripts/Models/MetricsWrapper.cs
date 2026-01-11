using System;
using System.Collections.Generic;

/// <summary>
/// Wrapper class used to serialise a list of RunMetric objects. Unity’s
/// JsonUtility cannot serialise top-level lists, so we wrap them.
/// </summary>
[Serializable]
public class MetricsWrapper
{
    public List<RunMetric> runs;
    public List<DroneGoalProfile> goalProfiles;
}

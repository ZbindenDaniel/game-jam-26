/// <summary>
/// Holds the proportional, integral and derivative gains for a PID controller.
/// Multiple instances of this struct are used to define alternative parameter
/// sets for each PID in the PlayerDrone. These sets are selected at
/// runtime by a behaviour-selection neural network.
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

using UnityEngine;

/// <summary>
/// Encapsulates a PID controller and the state required for simple
/// gradient-descent tuning.  Each instance maintains its own gradient
/// accumulators and allows enabling or disabling optimisation per
/// controller.  At the end of an episode, the ApplyTuning method
/// adjusts the controller gains by an amount proportional to the
/// accumulated gradients and a supplied learning rate.  Integral and
/// derivative information is gathered from the error passed into
/// AccumulateGradients() each frame.  ResetTuning() clears the
/// accumulated gradient state and resets the controller's integral and
/// previous error.
/// </summary>
[System.Serializable]
public class AutoTunePID
{
    /// <summary>The underlying PID controller used for control.</summary>
    public PIDController controller = new PIDController();
    /// <summary>If true, the controller gains will be adjusted when
    /// ApplyTuning() is called.  If false, gradients are still
    /// accumulated but no tuning is applied.</summary>
    public bool optimize = true;
    // Internal state for gradient accumulation
    private float prevError;
    private float integralSum;
    private float gradKp;
    private float gradKi;
    private float gradKd;
    /// <summary>
    /// Call once per frame to accumulate gradient information for the
    /// controller.  The supplied error is the control error for the
    /// current frame and dt is the time step.  Gradients are computed
    /// based on correlations of the error with its derivative and integral.
    /// These terms allow gains to increase or decrease depending on whether
    /// the error is growing or shrinking.  Specifically, Kp uses the
    /// correlation between the error and its derivative; Ki uses the
    /// correlation of the error with the accumulated integral of error;
    /// Kd uses the same error–derivative correlation as Kp by default.
    /// </summary>
    public void AccumulateGradients(float error, float dt)
    {
        // Accumulate the integral of the error for the Ki gradient
        integralSum += error * dt;
        // Derivative of the error for Kp and Kd gradients
        float derivative = dt > 0f ? (error - prevError) / dt : 0f;
        // Gradients: accumulate correlation terms.  These can be
        // positive or negative depending on whether error is increasing
        // or decreasing.  This allows gains to adjust in both directions.
        gradKp += error;
        gradKi += error * integralSum;
        gradKd += error * derivative;
        // Store previous error for next derivative calculation
        prevError = error;
    }
    /// <summary>
    /// Applies the accumulated gradients to the controller gains using
    /// the provided learning rate and the duration of the episode.  The
    /// gradient sum is scaled by 1/duration to obtain an average
    /// gradient.  Gains are clamped to zero to prevent negative values.
    /// After applying tuning the internal gradient and error state is
    /// reset.
    /// </summary>
    public void ApplyTuning(float episodeDuration, float learningRate)
    {
        if (!optimize || episodeDuration <= 0f) {
            ResetTuning();
            return;
        }
        float scale = 1f / episodeDuration;
        // Gradient update: adjust gains in the direction indicated by the
        // accumulated gradients.  A positive gradient will increase the
        // corresponding gain and a negative gradient will decrease it.  Gains
        // are clamped to zero to prevent negative values, but can grow
        // arbitrarily large if the optimiser dictates.
        controller.Kp = Mathf.Max(0f, controller.Kp + learningRate * gradKp * scale);
        controller.Ki = Mathf.Max(0f, controller.Ki + learningRate * gradKi * scale);
        controller.Kd = Mathf.Max(0f, controller.Kd + learningRate * gradKd * scale);
        // Reset gradient state after tuning
        ResetTuning();
    }
    /// <summary>
    /// Resets the gradient accumulation and PID internal state.  Call
    /// this at the start of a new episode to prevent carry‑over from
    /// previous runs.  The underlying PID integral and previous error
    /// are also cleared via controller.Reset().
    /// </summary>
    public void ResetTuning()
    {
        prevError = 0f;
        integralSum = 0f;
        gradKp = 0f;
        gradKi = 0f;
        gradKd = 0f;
        if (controller != null)
        {
            controller.Reset();
        }
    }
}

/// <summary>
/// Simple PID controller with configurable proportional, integral and
/// derivative gains and an integral wind‑up limit.  This class does not
/// perform any filtering on the derivative term.  Reset() clears the
/// accumulated integral and previous error.
/// </summary>
[System.Serializable]
public class PIDController
{
    public float Kp = 2f;
    public float Ki = 0f;
    public float Kd = 0.5f;
    public float integralLimit = 5f;
    public bool activate = true;
    private float integral;
    private float prevError;
    private bool hasPrev;
    /// <summary>
    /// Compute the PID output for a given error and time step.  If the
    /// controller is disabled (activate=false) returns zero.  Integral
    /// accumulation is clamped to ±integralLimit to mitigate wind‑up.
    /// </summary>
    public float Update(float error, float dt)
    {
        if (!activate) return 0f;
        float p = Kp * error;
        integral = Mathf.Clamp(integral + error * dt, -integralLimit, integralLimit);
        float i = Ki * integral;
        float d = hasPrev && dt > 0f ? Kd * (error - prevError) / dt : 0f;
        prevError = error;
        hasPrev = true;
        return p + i + d;
    }
    /// <summary>
    /// Reset the internal integral and derivative state of the PID.  Call
    /// this when starting a new episode or when resetting the plant to
    /// avoid carry‑over of past error.
    /// </summary>
    public void Reset()
    {
        integral = 0f;
        prevError = 0f;
        hasPrev = false;
    }
}
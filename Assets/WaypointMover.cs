using UnityEngine;

/// <summary>
/// Moves a waypoint GameObject along a smooth, erratic sinusoidal path in
/// the XZ plane.  The motion is based on a base sine wave combined with
/// Perlin noise for both lateral deviation and speed modulation.  The
/// waypoint’s Y coordinate remains fixed at its initial value to
/// prevent it from following the altitude of any drone.  Attach this
/// script to the waypoint GameObject to allow multiple drones to
/// independently follow their own targets without mixing control logic.
/// </summary>
public class WaypointMover : MonoBehaviour
{
    public float targetAltitude = 1.5f;

    
    [Header("Path Type")]
    /// <summary>
    /// If true the waypoint follows a smooth sine path in XZ (with noise).  If false
    /// and useCircularPath is true, a circular path is used.  At least one
    /// of useSinePath or useCircularPath must be true for movement.
    /// </summary>
    public bool useSinePath = false;
    /// <summary>
    /// If true the waypoint moves along a circular path around its start position.
    /// The radius and angular speed can be configured.  If both
    /// useCircularPath and useSinePath are false, the waypoint remains
    /// stationary.  Defaults to true for bounded motion.
    /// </summary>
    [Header("Circular Path Settings")]
    public bool useCircularPath = true;
    /// <summary>Radius of the circular path in metres.</summary>
    public float circleRadius = 5f;
    /// <summary>Angular speed of the circular path in radians per second.  Positive
    /// values result in counter‑clockwise motion when looking down the Y axis.</summary>
    public float circleAngularSpeed = 1f;
    // --------------------------------------------------------------------
    // Sine path settings
    // --------------------------------------------------------------------
    /// <summary>The nominal speed along the X direction (for sine path).</summary>
    public float pathSpeed = 1f;
    /// <summary>The amplitude of the sine wave in the Z direction.</summary>
    public float pathAmplitudeZ = 5f;
    /// <summary>The frequency of the sine wave in cycles per second.</summary>
    public float pathFrequency = 0.5f;
    /// <summary>The amplitude of the Perlin noise deviations in X and Z.</summary>
    public float deviationAmplitude = 2f;
    /// <summary>The frequency of the Perlin noise used for positional deviation.</summary>
    public float noiseFrequency = 0.5f;
    /// <summary>The amplitude of the speed deviation (noise factor).</summary>
    public float speedDeviation = 1f;
    /// <summary>The frequency of the Perlin noise used for speed modulation.</summary>
    public float speedNoiseFrequency = 0.2f;

    // Internal state for sine path
    private float pathTime;
    private float pathDistanceX;
    // Internal state for circular path
    private float circleAngle;

    // Internal state
    private Vector3 startPos;
    private float noiseOffsetX;
    private float noiseOffsetZ;
    private float speedNoiseOffset;

    void Start()
    {
        // Record the initial position so the waypoint returns here when the path resets
        startPos = transform.position;
        // Randomise noise offsets to avoid synchronous patterns when multiple
        // waypoints use the same parameters.
        noiseOffsetX = Random.Range(0f, 1000f);
        noiseOffsetZ = Random.Range(0f, 1000f);
        speedNoiseOffset = Random.Range(0f, 1000f);
        // Initialise internal state
        pathTime = 0f;
        pathDistanceX = 0f;
        circleAngle = 0f;
    }

    void Update()
    {
        float dt = Time.deltaTime;
        // Circular path mode takes precedence over sine path
        if (useCircularPath)
        {
            // Advance the circular angle
            circleAngle += circleAngularSpeed * dt;
            // Base circular position
            float x = startPos.x + circleRadius * Mathf.Cos(circleAngle);
            float z = startPos.z + circleRadius * Mathf.Sin(circleAngle);
            // Perlin noise for organic motion
            float dxNoise = (Mathf.PerlinNoise(Time.time * noiseFrequency + noiseOffsetX, 0f) - 0.5f) * 2f * deviationAmplitude;
            float dzNoise = (Mathf.PerlinNoise(0f, Time.time * noiseFrequency + noiseOffsetZ) - 0.5f) * 2f * deviationAmplitude;
            Vector3 newPos = new Vector3(x + dxNoise, Terrain.activeTerrain.SampleHeight(transform.position) + targetAltitude, z + dzNoise);
            transform.position = newPos;
            return;
        }
        // Sine path mode
        if (useSinePath)
        {
            // Smooth noise to modulate speed
            float speedNoise = (Mathf.PerlinNoise(Time.time * speedNoiseFrequency + speedNoiseOffset, 0f) - 0.5f) * 2f;
            float currentSpeed = pathSpeed + speedNoise * speedDeviation;
            if (currentSpeed < 0f) currentSpeed = 0f;
            pathDistanceX += currentSpeed * dt;
            pathTime += dt;
            // Base sine wave for Z
            float baseZ = pathAmplitudeZ * Mathf.Sin(2f * Mathf.PI * pathFrequency * pathTime);
            // Perlin noise for deviations
            float dxNoise = (Mathf.PerlinNoise(Time.time * noiseFrequency + noiseOffsetX, 0f) - 0.5f) * 2f * deviationAmplitude;
            float dzNoise = (Mathf.PerlinNoise(0f, Time.time * noiseFrequency + noiseOffsetZ) - 0.5f) * 2f * deviationAmplitude;
            Vector3 newPos = startPos + new Vector3(pathDistanceX + dxNoise, 0f, baseZ + dzNoise);
            transform.position = newPos;
        }
        // If neither path is enabled, do nothing
    }

    /// <summary>
    /// Resets the path generator to its initial state.  Use this to restart
    /// the path from the original position.  Noise offsets are re-
    /// randomised to vary the path between runs.
    /// </summary>
    public void ResetPath()
    {
        // Reset sine path state
        pathTime = 0f;
        pathDistanceX = 0f;
        // Reset circular path state
        circleAngle = 0f;
        // Optionally move back to start position to restart the loop
        transform.position = startPos;
        // Randomise noise offsets to vary the path between runs
        noiseOffsetX = Random.Range(0f, 1000f);
        noiseOffsetZ = Random.Range(0f, 1000f);
        speedNoiseOffset = Random.Range(0f, 1000f);
    }
}
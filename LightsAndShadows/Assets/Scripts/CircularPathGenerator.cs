using UnityEngine;

/// <summary>
/// Generates positions along a circular path around a given centre in the XZ
/// plane.  The angular speed can be set to control how fast the
/// object moves around the circle.  Optional Perlin noise can be used
/// to add smooth deviations to the path for more organic motion.  This
/// class is not a MonoBehaviour; it is intended to be instantiated
/// directly and updated each frame via the Update() method.  The Y
/// coordinate of the generated positions is taken from the supplied
/// centre so that the path remains level.
/// </summary>
public class CircularPathGenerator
{
    public Vector3 centre;
    public float radius = 5f;
    public float angularSpeed = 1f;
    public float deviationAmplitude = 0f;
    public float noiseFrequency = 0.5f;
    private float angle;
    private float noiseOffsetX;
    private float noiseOffsetZ;

    /// <summary>
    /// Construct a circular path generator around a given centre.  The
    /// radius and angularSpeed define the circle; noise parameters
    /// control optional path deviations.
    /// </summary>
    public CircularPathGenerator(Vector3 centre, float radius, float angularSpeed, float deviationAmplitude = 0f, float noiseFrequency = 0.5f)
    {
        this.centre = centre;
        this.radius = radius;
        this.angularSpeed = angularSpeed;
        this.deviationAmplitude = deviationAmplitude;
        this.noiseFrequency = noiseFrequency;
        // Random offsets to decorrelate noise across instances
        noiseOffsetX = Random.Range(0f, 10f);
        noiseOffsetZ = Random.Range(0f, 10f);
        angle = 0f;
    }
    /// <summary>
    /// Update the path position by advancing the internal angle based on
    /// angularSpeed and delta time.  Returns the new position along the
    /// circle with optional noise deviations.  The Y component of the
    /// returned position matches the centre’s Y.
    /// </summary>
    public Vector3 Update(float dt)
    {
        // Advance the angle
        angle += angularSpeed * dt;
        // Compute base circular position
        float x = centre.x + radius * Mathf.Cos(angle);
        float z = centre.z + radius * Mathf.Sin(angle);
        float y = centre.y;
        // Add Perlin noise for organic motion
        float dxNoise = 0f;
        float dzNoise = 0f;
        if (deviationAmplitude > 0f)
        {
            dxNoise = (Mathf.PerlinNoise(Time.time * noiseFrequency + noiseOffsetX, 0f) - 0.5f) * 2f * deviationAmplitude;
            dzNoise = (Mathf.PerlinNoise(0f, Time.time * noiseFrequency + noiseOffsetZ) - 0.5f) * 2f * deviationAmplitude;
        }
        return new Vector3(x + dxNoise, y, z + dzNoise);
    }
    /// <summary>
    /// Resets the path generator angle and randomises noise offsets.
    /// </summary>
    public void Reset()
    {
        angle = 0f;
        noiseOffsetX = Random.Range(0f, 1000f);
        noiseOffsetZ = Random.Range(0f, 1000f);
    }
}
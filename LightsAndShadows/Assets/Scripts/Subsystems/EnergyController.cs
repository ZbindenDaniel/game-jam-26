using System;
using UnityEngine;

public class EnergyController : MonoBehaviour
{
    [Header("Energy Settings")]
    [SerializeField] private float maxEnergy = 100f;
    [SerializeField] private float energy = 100f;
    [SerializeField] private float energyDrainRate = 5f;
    [SerializeField] private float energyRechargeRate = 10f;
    [Range(0f, 1f)]
    [SerializeField] private float minEnergyThrustMultiplier = 0.2f;
    [SerializeField] private Light sunLight;
    [SerializeField] private LayerMask lightOccluderLayers = ~0;
    [SerializeField] private float lightCheckOffset = 0.1f;
    [SerializeField] private bool logEnergyChanges = false;
    [SerializeField] private float energyLogInterval = 1f;

    [Header("Dependencies")]
    [SerializeField] private CanTakeShootingDamage damageReceiver;
    [SerializeField] private Rigidbody rb;

    private float lastEnergyLogTime = float.NegativeInfinity;
    private float energyRatio = 1f;

    public float EnergyRatio => energyRatio;
    public float MinEnergyThrustMultiplier => minEnergyThrustMultiplier;
    public float MaxEnergy => maxEnergy;
    public float Energy => energy;

    private void Awake()
    {
        if (rb == null)
        {
            rb = GetComponent<Rigidbody>();
        }

        if (damageReceiver == null)
        {
            damageReceiver = GetComponent<CanTakeShootingDamage>();
        }

        if (damageReceiver != null)
        {
            maxEnergy = damageReceiver.MaxEnergy;
            energy = damageReceiver.Energy;
        }
    }

    public void Tick(float dt)
    {
        UpdateEnergy(dt);
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

                if ((rb != null && hit.collider.attachedRigidbody == rb) || hit.collider.transform.IsChildOf(transform))
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
}

using UnityEngine;
using UnityEngine.Serialization;

public class CanTakeShootingDamage : MonoBehaviour
{
    public float MaxEnergy = 100f;
    [FormerlySerializedAs("Health")]
    public float Energy = 100f;

    public void TakeDamage(float damageAmount)
    {
        try
        {
            Energy = Mathf.Clamp(Energy - damageAmount, 0f, MaxEnergy);
            Debug.Log($"{gameObject.name} took {damageAmount} damage, remaining energy: {Energy}");
            if (Energy <= 0f)
            {
                // Destroy(gameObject);
                Energy = MaxEnergy; // Reset energy for testing purposes
                Debug.Log($"{gameObject.name} has been destroyed!");
            }
        }
        catch (System.Exception ex)
        {
            Debug.LogWarning($"[CanTakeShootingDamage] Damage handling failed on {name}: {ex.Message}");
        }
    }
    
}

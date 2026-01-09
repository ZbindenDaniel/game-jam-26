using UnityEngine;

public class CanTakeShootingDamage : MonoBehaviour
{
    public float Health = 100f;

    public void TakeDamage(float damageAmount)
    {
        Health -= damageAmount;
        Debug.Log($"{gameObject.name} took {damageAmount} damage, remaining health: {Health}");
        if (Health <= 0f)
        {
            // Destroy(gameObject);
            Health = 100f; // Reset health for testing purposes
            Debug.Log($"{gameObject.name} has been destroyed!");
        }
    }
    
}

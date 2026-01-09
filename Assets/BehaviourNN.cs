using UnityEngine;

/// <summary>
/// A lightweight neural network used to select among multiple PID gain
/// parameter sets based on high‑level state inputs.  Each output
/// corresponds to a PID controller and produces an integer index into the
/// associated parameter array.  The network structure is deliberately
/// simple: a linear layer followed by a sigmoid activation for each
/// output.  The resulting value in [0,1] is scaled to the number of
/// available parameter sets and quantised to an integer.  Weights and
/// biases can be initialised arbitrarily; training infrastructure can
/// adjust them externally if desired.
/// 
/// 
///   weights = new float[5,5]//TODO:[numOutputs, numInputs]
        // {
        //     { 0.17241962f, 0.19063652f, 0.18077548f, -0.8466811f, -1.2664876f },
        //     { 0.16533452f, 0.23476353f, 0.14199287f, -0.8296914f, -1.299879f },
        //     { -0.2033094f, -0.16822706f, -0.13516092f, 0.8427362f, 1.2630439f },
        //     { 0.060904976f, 0.22732234f, 0.25393358f, -0.82979625f, -1.3017689f },
        //     { 0.7366702f, 0.7428996f, 0.77167016f, 0.36660504f, 0.8675859f }
        // };
        // biases = new float[5]//TODO:[numOutputs]
        // {
        //     0.12716077f,
        //     0.15355732f,
        //     -0.16169009f,
        //     0.1548475f,
        //     0.67329186f
        // };
/// </summary>
public class BehaviourNN : MonoBehaviour
{
    private readonly int numInputs;
    private readonly int paramCount;
    private readonly float[] weights; // weights[input]
    private float bias;

    public int InputCount => numInputs;
    public int ParamCount => paramCount;
    public float Bias => bias;

    /// <summary>
    /// Constructs a BehaviourNN with the specified dimensions.  Weights are
    /// initialised to small random values and biases are zero.  The number
    /// of parameter sets determines the range of indices produced by the
    /// network.
    /// </summary>
    /// <param name="inputs">Number of input features.</param>
    /// <param name="outputs">Number of outputs (one per PID).</param>
    /// <param name="paramCount">Number of parameter sets available.</param>
    public BehaviourNN(int inputs, int paramCount)
    {
        this.numInputs = inputs;
        this.paramCount = Mathf.Max(1, paramCount);
        weights = new float[numInputs];
        bias = 0f;
        // Initialise weights with small random values for diversity.  A
        // deterministic pseudo‑random sequence ensures reproducibility.
        System.Random rng = new System.Random(0);
        for (int j = 0; j < numInputs; j++)
        {
            // Small values around zero
            weights[j] = (float)(rng.NextDouble() * 0.2 - 0.1);
        }
    }

    /// <summary>
    /// Applies the network to the provided input vector and returns an
    /// index corresponding to the selected parameter set.
    /// Uses a logistic activation to map the weighted sum of inputs into [0,1],
    /// then multiplies by paramCount and floors the result to an integer in [0, paramCount-1].
    /// </summary>
    /// <param name="inputs">Input vector with length equal to numInputs.</param>
    /// <returns>Selected parameter index in range [0, paramCount-1].</returns>
    public int SelectParamSet(float[] inputs)
    {
        float sum = bias;
        // Safely handle mismatched input lengths by using the minimum
        int len = Mathf.Min(numInputs, inputs != null ? inputs.Length : 0);
        for (int j = 0; j < len; j++)
        {
            sum += weights[j] * inputs[j];
        }
        float output = Sigmoid(sum);
        int index = (int)Mathf.Floor(output * paramCount);
        if (index < 0) index = 0;
        if (index >= paramCount) index = paramCount - 1;
        return index;
    }

    /// <summary>
    /// Trains the neural network on a batch of examples using a simple
    /// gradient descent algorithm.  Each example consists of an input
    /// vector and a target index. The network predicts a continuous value
    /// in [0,1] via a sigmoid; this value is compared to the normalised
    /// target index using mean squared error loss.  The weights and bias
    /// are updated in the negative gradient direction scaled by the learning
    /// rate.  When only a single parameter set exists (paramCount == 1) the
    /// target normalisation collapses to zero and the gradients become zero,
    /// so no updates occur.  Multiple epochs can be run to iterate over the
    /// batch repeatedly.
    /// </summary>
    /// <param name="inputsBatch">Array of input feature vectors.  Each entry must have length numInputs.</param>
    /// <param name="targetsBatch">Array of target indices.  Each entry must be a valid parameter set index.</param>
    /// <param name="learningRate">Step size for gradient descent.  Typical values are small (e.g. 0.01).</param>
    /// <param name="epochs">Number of full passes over the training data to perform.  At least 1.</param>
    public void Train(float[][] inputsBatch, int[] targetsBatch, float learningRate, int epochs)
    {
        if (inputsBatch == null || targetsBatch == null) return;
        int sampleCount = Mathf.Min(inputsBatch.Length, targetsBatch.Length);
        if (sampleCount <= 0) return;
        // Ensure at least one epoch
        int maxEpochs = Mathf.Max(1, epochs);
        for (int epoch = 0; epoch < maxEpochs; epoch++)
        {
            // Loop over each sample in the batch
            for (int s = 0; s < sampleCount; s++)
            {
                float[] x = inputsBatch[s];
                int targetIndex = targetsBatch[s];
                // Forward pass: compute raw sum and activation
                float sum = bias;
                int len = Mathf.Min(numInputs, x != null ? x.Length : 0);
                for (int j = 0; j < len; j++)
                {
                    sum += weights[j] * x[j];
                }
                float activation = Sigmoid(sum);
                // Backward pass: update weights and bias
                // Normalise the target index to [0,1] when multiple parameter sets exist
                float targetNorm = 0f;
                if (paramCount > 1)
                {
                    int clampedTarget = Mathf.Clamp(targetIndex, 0, paramCount - 1);
                    targetNorm = (float)clampedTarget / (float)(paramCount - 1);
                }
                float y = activation;
                // Compute gradient of loss w.r.t activation (sigmoid output)
                float diff = y - targetNorm;
                float gradActivation = diff * y * (1f - y);
                // Update weights
                for (int j = 0; j < len; j++)
                {
                    float gradW = gradActivation * x[j];
                    weights[j] -= learningRate * gradW;
                }
                // Update bias
                bias -= learningRate * gradActivation;
            }
        }
    }

    /// <summary>
    /// Convenience method to train the network on a single example.  This
    /// wraps the batch training method with a batch size of one and a
    /// single epoch.  It can be called repeatedly online as new data
    /// becomes available.
    /// </summary>
    /// <param name="input">Input feature vector.</param>
    /// <param name="targetIndex">Target index for the output.</param>
    /// <param name="learningRate">Learning rate for the update.</param>
    public void TrainSample(float[] input, int targetIndex, float learningRate)
    {
        float[][] inputs = new float[1][] { input };
        int[] targets = new int[1] { targetIndex };
        Train(inputs, targets, learningRate, 1);
    }

    public float[] GetWeightsCopy()
    {
        float[] copy = new float[weights.Length];
        weights.CopyTo(copy, 0);
        return copy;
    }

    public bool TryApplyWeights(float[] newWeights, float newBias)
    {
        if (newWeights == null || newWeights.Length != weights.Length)
        {
            return false;
        }
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = newWeights[i];
        }
        bias = newBias;
        return true;
    }

    private static float Sigmoid(float x)
    {
        return 1f / (1f + Mathf.Exp(-x));
    }
}

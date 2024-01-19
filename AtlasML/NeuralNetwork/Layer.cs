namespace AtlasML.NeuralNetwork;
public class Layer
{
  public Neuron[] Neurons { get; set; }
  private readonly string _activationFunction;

  public Layer(string activationFunction, int neuronCount)
  {
    _activationFunction = activationFunction;
  }
  public double[] CalculateOutputs(double[] inputs)
  {
    double[] outputs = new double[Neurons.Length];
    for (int i = 0; i < Neurons.Length; i++)
      outputs[i] = Neurons[i].CalculateOutput(inputs);

    if (_activationFunction == "Softmax")
      outputs = Softmax(outputs);

    return outputs;
  }

  /// <summary>
  /// Softmax is used for multi class classification. In multi class classification output layer contains multiple neurons, every neuron uses linear
  /// activation function and Layer calculates output values.
  /// </summary>
  /// <param name="z"></param>
  /// <returns></returns>
  private static double[] Softmax(double[] z)
  {
    // Calculate exponential sum
    double sum = z.Sum(x => Math.Exp(x));

    // Calculate and return normalized values
    return z.Select(x => Math.Exp(x) / sum).ToArray();
  }
}

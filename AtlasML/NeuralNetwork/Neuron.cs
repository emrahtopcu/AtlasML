using AtlasML.Extensions;

namespace AtlasML.NeuralNetwork;
public class Neuron
{
  public double[] InputWeights { get; set; }
  public double[] Gradients { get; set; }
  public double Bias { get; set; }
  public double LeakParameter { get; set; }
  public double WeightedSum { get; set; }
  public double Output { get; set; }

  private readonly Func<double, double> _activationFunction;

  public Neuron(string activationFunction)
  {
    _activationFunction = activationFunction switch
    {
      "Linear" => Linear,
      "Sigmoid" => Sigmoid,
      "ReLU" => ReLU,
      "LReLU" => LReLU,
      "Tanh" => Tanh,
      "Softmax" => Linear,
      _ => Sigmoid,
    };
  }
  public double CalculateOutput(double[] imputs)
  {
    var dotProduct = imputs.Dot(InputWeights) + Bias;
    WeightedSum = dotProduct;
    Output = _activationFunction(dotProduct);
    return Output;
  }
  private protected double Linear(double z)
  {
    return z;
  }

  /// <summary>
  /// The sigmoid function is a common choice for activation functions in neural networks, especially in classification tasks.
  /// It maps any real-valued input to a value between 0 and 1, making it useful for representing probabilities or confidence levels.
  /// </summary>
  /// <param name="z"></param>
  /// <returns></returns>
  private protected double Sigmoid(double z)
  {
    return 1.0 / (1.0 + Math.Exp(-z));
  }

  /// <summary>
  /// The ReLU function is a popular choice for activation functions in neural networks, especially in regression tasks.
  /// It has a number of advantages over other activation functions, including:
  /// It is computationally efficient.
  /// It is less likely to cause the vanishing gradient problem.
  /// However, it can lead to the dying ReLU problem, where some neurons become inactive and contribute nothing to the network's output.
  /// </summary>
  /// <param name="z"></param>
  /// <returns></returns>
  private protected double ReLU(double z)
  {
    return Math.Max(.0, z);
  }

  /// <summary>
  /// Leaky ReLU addresses the "dying ReLU" problem where negative neurons never activate.
  /// It maintains computational efficiency compared to sigmoid.
  /// Choosing the optimal alpha value requires experimentation and consideration of your specific task.
  /// Conditional Operator: The function uses a conditional operator based on the sign of z:
  /// For positive inputs (z >= 0), simply return the input value.
  /// For negative inputs, apply the leak by multiplying z by LeakParameter.
  /// </summary>
  /// <param name="z"></param>
  /// <returns></returns>
  private protected double LReLU(double z)
  {
    var alpha = Math.Max(.0, LeakParameter);
    return (z >= .0) ? z : z * alpha;
  }

  /// <summary>
  /// Tanh offers a smooth S-shaped curve, similar to sigmoid, but centered around 0.
  /// It's often preferred in hidden layers for its ability to capture both positive and negative values.
  /// It can help mitigate the vanishing gradient problem compared to sigmoid.
  /// </summary>
  /// <param name="z"></param>
  /// <returns>Returns the computed hyperbolic tangent value, which ranges between -1 and 1</returns>
  private protected double Tanh(double z)
  {
    return Math.Tanh(z);
  }
}

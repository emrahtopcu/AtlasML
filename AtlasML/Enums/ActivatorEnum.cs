namespace AtlasML.Enums;
public enum ActivatorEnum
{
  /// <summary>
  /// No activation.
  /// </summary>
  Linear,

  /// <summary>
  /// The sigmoid function is a common choice for activation functions in neural networks, especially in classification tasks.
  /// It maps any real-valued input to a value between 0 and 1, making it useful for representing probabilities or confidence levels.
  /// </summary>
  Sigmoid,

  /// <summary>
  /// The ReLU function is a popular choice for activation functions in neural networks, especially in regression tasks.
  /// It has a number of advantages over other activation functions, including:
  ///   It is computationally efficient.
  ///   It is less likely to cause the vanishing gradient problem.
  /// However, it can lead to the dying ReLU problem, where some neurons become inactive and contribute nothing to the network's output.
  /// </summary>
  ReLU,

  /// <summary>
  /// Leaky ReLU addresses the "dying ReLU" problem where negative neurons never activate.
  /// It maintains computational efficiency compared to sigmoid.
  /// Choosing the optimal alpha value requires experimentation and consideration of your specific task.
  /// Conditional Operator: The function uses a conditional operator based on the sign of z:
  ///   For positive inputs (z >= 0), simply return the input value.
  ///   For negative inputs, apply the leak by multiplying z by LeakParameter.
  /// </summary>
  LReLU,

  /// <summary>
  /// Tanh offers a smooth S-shaped curve, similar to sigmoid, but centered around 0.
  /// It's often preferred in hidden layers for its ability to capture both positive and negative values.
  /// It can help mitigate the vanishing gradient problem compared to sigmoid.
  /// </summary>
  Tanh,

  /// <summary>
  /// Softmax is used for multi class classification. In multi class classification output layer contains multiple neurons, every neuron uses linear
  /// activation function and Layer calculates output values.
  /// </summary>
  Softmax
}

namespace AtlasML.Enums;
public enum InitializerEnum
{
  /// <summary>
  /// Issues: 
  ///   Symmetry breaking: Identical neurons lead to stagnant learning.
  ///   Vanishing gradients: Activations become identical, hindering gradient flow.
  /// </summary>
  Zero,

  /// <summary>
  /// Improvements:
  ///   Breaks symmetry, allowing neurons to learn different features.
  /// Issues:
  ///   Still susceptible to vanishing/exploding gradients in deep networks.
  /// </summary>
  Random,

  /// <summary>
  /// Scaled random initialization: Uses a scaling factor based on the number of input and output connections for each neuron.
  /// Goal: 
  ///   Keep variance of activations and gradients consistent across layers, reducing vanishing/exploding gradients.
  /// </summary>
  Xavier,

  /// <summary>
  /// Variant of Xavier initialization: Designed for ReLU activation functions, using a scaling factor of 2 instead of 1.
  /// Addresses ReLU's asymmetry: Accounts for ReLU's zero gradient for negative inputs.
  /// </summary>
  He,

  /// <summary>
  /// Use Cases: 
  ///   Orthogonal initialization is often used with linear activation functions and for specific network architectures.
  /// </summary>
  Orthogonal
}

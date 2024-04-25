namespace AtlasML.Enums;
public enum OptimizerEnum
{
  /// <summary>
  /// Batch optimisation performs forward propaganation for entire training data and then calculates loss for all and then back propaganates.
  /// Calculate the loss for each sample within the batch.
  /// Sum the individual losses to obtain the total loss for the entire batch.
  /// Use this total loss for backpropagation.
  /// </summary>
  Batch,

  /// <summary>
  /// Mini batch is similar to Batch optimization instead all the train data mini batch choses a fraction of train data to performe a Batch optimization.
  /// Calculate the loss for each sample within the batch.
  /// Sum the individual losses to obtain the total loss for the entire batch.
  /// Use this total loss for backpropagation.
  /// </summary>
  MiniBatch,

  /// <summary>
  /// Stochastic Gradient Descent: Online learning, or true stochastic gradient descent (SGD), processes one data point at a time, 
  /// making it adaptable to changing data distributions and suitable for real-time updates. 
  /// However, it introduces noise and may require careful tuning of hyperparameters, 
  /// making it a choice dependent on the dynamic nature of the data and specific application requirements.
  /// </summary>
  SGD,

  /// <summary>
  /// Adaptive Moment Estimation: Adam is an adaptive optimization algorithm widely used in training neural networks. 
  /// By dynamically adjusting learning rates based on historical gradients and incorporating momentum and RMSprop concepts, 
  /// Adam offers efficient convergence and robust performance across various machine learning tasks.
  /// </summary>
  Adam,

  /// <summary>
  /// Root Mean Square Propagation: It addresses the challenges of vanishing or exploding gradients by maintaining an exponentially decaying average of squared gradients for each parameter. 
  /// This normalization helps stabilize the learning process, making RMSprop particularly effective in scenarios with sparse or noisy gradients.
  /// </summary>
  RMSprop
}

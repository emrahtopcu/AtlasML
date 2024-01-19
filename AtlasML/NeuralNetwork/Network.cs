using AtlasML.Extensions;

namespace AtlasML.NeuralNetwork;
public class Network
{
  public Layer[] Layers { get; set; }
  public double LearningRate { get; set; }

  private readonly string _activationFunction;
  private readonly int _epocs;

  public Network(string activation, int[] layerLengts, int epocs, string initialization = "Zero")
  {
    _activationFunction = activation;
    _epocs = epocs;
    var neuronActivation = activation == "Softmax" ? "Linear" : activation;

    Layers = new Layer[layerLengts.Length];
    for (int i = 0; i < layerLengts.Length; i++)
    {
      Layers[i] = new(_activationFunction, layerLengts[i]);

      var inputCount = i == 0 ? layerLengts[i] : layerLengts[i - 1];
      for (int j = 0; j < Layers[i].Neurons.Length; j++)
      {
        Layers[i].Neurons[j] = new Neuron(neuronActivation);

        switch (initialization)
        {
          case "Zero":
            ZeroInitialization(Layers[i].Neurons[j], inputCount);
            break;
          case "Random":
            RandomInitialization(Layers[i].Neurons[j], inputCount);
            break;
          case "Xavier":
            var fanIn = i == 0 ? layerLengts[i] : layerLengts[i - 1];
            var fanOut = i == Layers.Length - 1 ? Layers[i].Neurons.Length : Layers[i + 1].Neurons.Length;

            XavierInitialization(Layers[i].Neurons[j], inputCount, fanIn, fanOut);
            break;
          case "He":
            var fanIn_he = i == 0 ? layerLengts[i] : layerLengts[i - 1];

            HeInitialization(Layers[i].Neurons[j], inputCount, fanIn_he);
            break;
          case "Orthogonal":
            var fanIn_ortg = i == 0 ? layerLengts[i] : layerLengts[i - 1];

            OrthogonalInitialization(Layers[i].Neurons[j], inputCount, fanIn_ortg);
            break;
          default:
            throw new Exception("Invalid initialization function.");
        }
      }
    }
  }
  public void Train(double[][] X, double[][] Y)
  {
    for (int i = 0; i < _epocs; i++)
    {
      // Shuffle training data to avoid bias
      X.Shuffle();

      //canculate split index
      var splitIndex = X.Length / 3;

      //split test data
      var x_train = X[..splitIndex];
      var y_train = Y[..splitIndex];

      //split cross validation data
      var x_cv = X[splitIndex..];
      var y_cv = Y[splitIndex..];

      for (int j = 0; j < x_train.Length; j++)
      {
        var x = x_train[j];

        var outputs = ForwardPropagation(x);
        BackwardPropagation(outputs);
      }
    }
  }
  private double[] ForwardPropagation(double[] inputs)
  {
    var outputs = inputs;
    for (int i = 0; i < Layers.Length; i++)
      outputs = Layers[i].CalculateOutputs(outputs);
    return outputs;
  }
  private void BackwardPropagation(double[] inputs)
  {
    var errors = inputs;
    for (int i = Layers.Length - 1; i >= 0; i--)
    {
      errors = CalculateErrors(errors, Layers[i]);
      for (int j = 0; j < Layers[i].Neurons.Length; j++)
        for (int l = 0; l < Layers[i].Neurons[j].InputWeights.Length; l++)
          Layers[i].Neurons[j].Gradients[l] = Layers[i].Neurons[j].InputWeights[l] * errors[j];
      errors = CalculateErrors(errors, Layers[i]);
    }

    for (int i = Layers.Length - 1; i >= 0; i--)
    {
      // Update weights
      for (int j = 0; j < Layers[i].Neurons.Length; j++)
        UpdateWeights(Layers[i]);
    }
  }
  private double[] CalculateErrors(double[] inputs, Layer layer)
  {
    var outputErrors = new double[layer.Neurons.Length];
    if (_activationFunction == "Softmax")
    {
      // Calculate exponential sum
      double sum = layer.Neurons.Select(n => n.WeightedSum).Sum(x => Math.Exp(x));
      var outputProbabilities = layer.Neurons.Select(n => n.WeightedSum).Select(x => Math.Exp(x) / sum).ToArray();

      // Calculate error vector for each class
      double[] errors = new double[outputProbabilities.Length];
      for (int i = 0; i < outputProbabilities.Length; i++)
        errors[i] = (inputs[i] - outputProbabilities[i]) * outputProbabilities[i];

      // Adjust for numerical stability (optional)
      for (int i = 0; i < errors.Length; i++)
        outputErrors[i] = Math.Max(errors[i], 1e-10); // Prevent zero or very small errors
    }
    else
      for (int i = 0; i < layer.Neurons.Length; i++)
      {
        switch (_activationFunction)
        {
          case "Linear":
            outputErrors[i] = layer.Neurons[i].Output - inputs[i];
            break;
          case "Sigmoid":
            double sigmoidDerivative = layer.Neurons[i].Output * (1 - layer.Neurons[i].Output);
            outputErrors[i] = (inputs[i] - layer.Neurons[i].Output) * sigmoidDerivative;
            break;
          case "ReLU":
            outputErrors[i] = (inputs[i] - layer.Neurons[i].Output) * (layer.Neurons[i].Output > .0 ? 1.0 : .0);
            break;
          case "LReLU":
            if (layer.Neurons[i].Output >= .0)
              outputErrors[i] = inputs[i] - layer.Neurons[i].Output;
            else
              outputErrors[i] = (inputs[i] - layer.Neurons[i].Output) * layer.Neurons[i].LeakParameter;
            break;
          case "Tanh":
            double tanhDerivative = 1.0 - (layer.Neurons[i].Output * layer.Neurons[i].Output);
            outputErrors[i] = (inputs[i] - layer.Neurons[i].Output) * tanhDerivative;
            break;
          default:
            break;
        }
      }
    return outputErrors;
  }
  private void UpdateWeights(Layer layer)
  {
    for (int j = 0; j < layer.Neurons.Length; j++)
    {
      for (int l = 0; l < layer.Neurons[j].InputWeights.Length; l++)
      {
        double weightUpdate = -LearningRate * layer.Neurons[j].Gradients[l]; // Negative for gradient descent
        layer.Neurons[j].InputWeights[l] += weightUpdate;
      }

      // Update bias
      layer.Neurons[j].Bias -= LearningRate * layer.Neurons[j].Gradients[^1];
    }
  }

  /// <summary>
  /// Issues: 
  ///   Symmetry breaking: Identical neurons lead to stagnant learning.
  ///   Vanishing gradients: Activations become identical, hindering gradient flow.
  /// </summary>
  /// <param name="neuron"></param>
  /// <param name="parametersCount"></param>
  private void ZeroInitialization(Neuron neuron, int parametersCount)
  {
    neuron.InputWeights = new double[parametersCount];
    neuron.Gradients = new double[parametersCount];
    neuron.LeakParameter = .01;
  }

  /// <summary>
  /// Improvements:
  ///   Breaks symmetry, allowing neurons to learn different features.
  /// Issues:
  ///   Still susceptible to vanishing/exploding gradients in deep networks.
  /// </summary>
  /// <param name="neuron"></param>
  /// <param name="parametersCount"></param>
  private void RandomInitialization(Neuron neuron, int parametersCount)
  {
    ZeroInitialization(neuron, parametersCount);

    //Give a seed to random generator for reproductivity of random results. Or different seeds for every run for non linearity.
    var seed = DateTime.UtcNow.Year + DateTime.UtcNow.Month + DateTime.UtcNow.Day + DateTime.UtcNow.Hour + DateTime.UtcNow.Minute + DateTime.UtcNow.Second;
    var random = new Random(seed);

    //Increase σ for wider weight distribution and faster learning, but potentially less stable convergence.
    //Decrease σ for more focused exploration and potentially more stable convergence, but slower learning.
    double stddev = 0.01; // Standard deviation (σ)

    for (int i = 0; i < neuron.InputWeights.Length; i++)
      neuron.InputWeights[i] = random.NextGaussian() * stddev;
    neuron.Bias = random.NextGaussian() * stddev;
  }

  /// <summary>
  /// Scaled random initialization: Uses a scaling factor based on the number of input and output connections for each neuron.
  /// Goal: 
  ///   Keep variance of activations and gradients consistent across layers, reducing vanishing/exploding gradients.
  /// </summary>
  /// <param name="neuron"></param>
  /// <param name="parametersCount"></param>
  /// <param name="fanIn"></param>
  /// <param name="fanOut"></param>
  private void XavierInitialization(Neuron neuron, int parametersCount, int fanIn, int fanOut)
  {
    RandomInitialization(neuron, parametersCount);

    // Calculate Xavier scaling factor
    double stdDev = Math.Sqrt(2.0 / (fanIn + fanOut));

    // Initialize weights with random values scaled by stdDev
    for (int i = 0; i < neuron.InputWeights.Length; i++)
      neuron.InputWeights[i] = neuron.InputWeights[i] * 2 * stdDev - stdDev;

    // Initialize bias to zero
    neuron.Bias = .0;
  }

  /// <summary>
  /// Variant of Xavier initialization: Designed for ReLU activation functions, using a scaling factor of 2 instead of 1.
  /// Addresses ReLU's asymmetry: Accounts for ReLU's zero gradient for negative inputs.
  /// </summary>
  /// <param name="neuron"></param>
  /// <param name="parametersCount"></param>
  private void HeInitialization(Neuron neuron, int parametersCount, int fanIn)
  {
    RandomInitialization(neuron, parametersCount);

    // Calculate Xavier scaling factor
    double stdDev = Math.Sqrt(2.0 / fanIn);

    // Initialize weights with random values scaled by stdDev
    for (int i = 0; i < neuron.InputWeights.Length; i++)
      neuron.InputWeights[i] = neuron.InputWeights[i] * 2 * stdDev - stdDev;

    // Initialize bias to zero
    neuron.Bias = .0;
  }

  /// <summary>
  /// Use Cases: 
  ///   Orthogonal initialization is often used with linear activation functions and for specific network architectures.
  /// </summary>
  /// <param name="neuron"></param>
  /// <param name="parametersCount"></param>
  /// <param name="fanIn"></param>
  private void OrthogonalInitialization(Neuron neuron, int parametersCount, int fanIn)
  {
    ZeroInitialization(neuron, parametersCount);

    //Give a seed to random generator for reproductivity of random results. Or different seeds for every run for non linearity.
    var seed = DateTime.UtcNow.Year + DateTime.UtcNow.Month + DateTime.UtcNow.Day + DateTime.UtcNow.Hour + DateTime.UtcNow.Minute + DateTime.UtcNow.Second;
    var random = new Random(seed);

    //Create a random matrix.
    double[][] matrix = new double[fanIn][];
    for (int k = 0; k < fanIn; k++)
      matrix[k] = Enumerable.Repeat(0.0, fanIn).Select(_ => random.NextGaussian() - 0.5).ToArray();

    // Apply Gram-Schmidt process to create an orthogonal matrix
    for (int k = 0; k < fanIn; k++)
    {
      for (int l = 0; l < k; l++)
      {
        double projection = matrix[k].Dot(matrix[l]);
        matrix[k] = matrix[k].Minus(matrix[l].Times(projection));
      }

      //Vector Normalization
      var magnitude = Math.Sqrt(matrix[k].Sum(x => x * x));
      if (magnitude > .0)
        matrix[k] = matrix[k].Select(x => x / magnitude).ToArray();
    }

    // Flatten the matrix into a weight vector
    double[] weights = matrix.SelectMany(row => row).ToArray();

    neuron.InputWeights = weights;
    neuron.Bias = 0;
  }
}

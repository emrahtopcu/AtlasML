using AtlasML.Enums;
using AtlasML.Extensions;
using AtlasML.LossComputation;

namespace AtlasML.NeuralNetwork;
public class Network
{
  public Layer[] Layers { get; set; }
  public double LearningRate { get; set; }

  private readonly ActivatorEnum _activationFunction;
  private readonly InitializerEnum _initializer;
  private readonly NeuralNetworkTaskEnum _neuralNetworkTask;
  private readonly int _epocs;
  private readonly OptimizerEnum _optimizer;
  private readonly int[] _dimensions;

  public static Network CreateRegressionNetwork(int[] dimensions, int epocs)
  {
    return new Network(
      activation: ActivatorEnum.Linear,
      dimensions: dimensions,
      epocs: epocs,
      initialization: InitializerEnum.Random,
      optimizer: OptimizerEnum.SGD,
      task: NeuralNetworkTaskEnum.Regression);
  }
  public static Network CreateBinaryClassificationNetwork(int[] dimensions, int epocs)
  {
    return new Network(
      activation: ActivatorEnum.Sigmoid,
      dimensions: dimensions,
      epocs: epocs,
      initialization: InitializerEnum.Random,
      optimizer: OptimizerEnum.SGD,
      task: NeuralNetworkTaskEnum.BinaryClassification);
  }
  public static Network CreateMultiClassClassificationNetwork(int[] dimensions, int epocs)
  {
    return new Network(
      activation: ActivatorEnum.Softmax,
      dimensions: dimensions,
      epocs: epocs,
      initialization: InitializerEnum.Random,
      optimizer: OptimizerEnum.SGD,
      task: NeuralNetworkTaskEnum.MultiClassClassification);
  }

  public Network(ActivatorEnum activation, int[] dimensions, int epocs, InitializerEnum initialization = InitializerEnum.Zero, OptimizerEnum optimizer = OptimizerEnum.SGD, NeuralNetworkTaskEnum task = NeuralNetworkTaskEnum.Regression)
  {
    _activationFunction = activation;
    _initializer = initialization;
    _neuralNetworkTask = task;
    _epocs = epocs;
    _optimizer = optimizer;
    _dimensions = dimensions;
    var neuronActivation = activation == ActivatorEnum.Softmax ? ActivatorEnum.Linear : activation;

    Layers = new Layer[_dimensions.Length];
    for (int i = 0; i < _dimensions.Length; i++)
    {
      Layers[i] = new(_activationFunction, _dimensions[i]);

      var inputCount = i == 0 ? _dimensions[i] : _dimensions[i - 1];
      for (int j = 0; j < Layers[i].Neurons.Length; j++)
      {
        Layers[i].Neurons[j] = new Neuron(neuronActivation);

        switch (_initializer)
        {
          case InitializerEnum.Zero:
            ZeroInitialization(Layers[i].Neurons[j], inputCount);
            break;
          case InitializerEnum.Random:
            RandomInitialization(Layers[i].Neurons[j], inputCount);
            break;
          case InitializerEnum.Xavier:
            var fanIn = i == 0 ? _dimensions[i] : _dimensions[i - 1];
            var fanOut = i == Layers.Length - 1 ? Layers[i].Neurons.Length : Layers[i + 1].Neurons.Length;

            XavierInitialization(Layers[i].Neurons[j], inputCount, fanIn, fanOut);
            break;
          case InitializerEnum.He:
            var fanIn_he = i == 0 ? _dimensions[i] : _dimensions[i - 1];

            HeInitialization(Layers[i].Neurons[j], inputCount, fanIn_he);
            break;
          case InitializerEnum.Orthogonal:
            var fanIn_ortg = i == 0 ? _dimensions[i] : _dimensions[i - 1];

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
    switch (_optimizer)
    {
      case OptimizerEnum.Batch:
        break;
      case OptimizerEnum.MiniBatch:
        break;
      case OptimizerEnum.SGD:
        TrainSGD(X, Y);
        break;
      case OptimizerEnum.Adam:
        break;
      case OptimizerEnum.RMSprop:
        break;
      default:
        break;
    }
  }

  /// <summary>
  /// Stochastic Gradient Descent is a neural network optimization algo which performs forward and backward propaganation for every data point and updates weigts for every data point.
  /// </summary>
  /// <param name="X"></param>
  /// <param name="Y"></param>
  private void TrainSGD(double[][] X, double[][] Y)
  {
    for (int i = 0; i < _epocs; i++)
    {
      // Shuffle training data to avoid bias
      X.Shuffle();

      //canculate split index
      var splitIndex = X.Length - X.Length / 3;

      //split test data
      var x_train = X[..splitIndex];
      var y_train = Y[..splitIndex];

      //split cross validation data
      var x_cv = X[splitIndex..];
      var y_cv = Y[splitIndex..];

      for (int j = 0; j < x_train.Length; j++)
      {
        var x = x_train[j];
        var y = y_train[j];

        ForwardPropagation(x);
        BackwardPropagation(y);

        Console.Write($"Loss: {CalculateLoss(y_train):n4}");
      }
    }
  }

  private double[] ForwardPropagation(double[] inputs)
  {
    var activations = inputs;
    for (int i = 0; i < Layers.Length; i++)
      activations = Layers[i].CalculateOutputs(activations);
    return activations;
  }
  private void BackwardPropagation(double[] inputs)
  {
    CalculateErrors(inputs, Layers[^1]);
    for (int i = Layers.Length - 1; i >= 0; i--)
    {
      for (int j = 0; j < Layers[i].Neurons.Length; j++)
        for (int l = 0; l < Layers[i].Neurons[j].InputWeights.Length; l++)
          Layers[i].Neurons[j].Gradients[l] = Layers[i].Neurons[j].InputWeights[l] * inputs[j];
    }

    //Hidden layers back propagation
    for (int l = Layers.Length - 1; l >= 0; l--)
    {
      for (int i = 0; i < Layers[l].Neurons.Length; i++)
      {
        for (int k = 0; k < Layers[l + 1].Neurons.Length; k++)
        {
          Layers[l].Neurons[i].Error += Layers[l + 1].Neurons[k].Error * Layers[l].Neurons[i].InputWeights[k] * Layers[l].Neurons[i].ActivationFunction(Layers[l].Neurons[i].WeightedSum);
        }
      }
    }

    UpdateWeights();
  }
  private void CalculateErrors(double[] inputs, Layer layer, double lambda = .0)
  {
    if (_activationFunction == ActivatorEnum.Softmax)
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
        layer.Neurons[i].Error = Math.Max(errors[i], 1e-10); // Prevent zero or very small errors
    }
    else
      for (int i = 0; i < layer.Neurons.Length; i++)
      {
        switch (_activationFunction)
        {
          case ActivatorEnum.Linear:
            layer.Neurons[i].Error = layer.Neurons[i].Output - inputs[i];
            break;
          case ActivatorEnum.Sigmoid:
            double sigmoidDerivative = layer.Neurons[i].Output * (1 - layer.Neurons[i].Output);
            layer.Neurons[i].Error = (inputs[i] - layer.Neurons[i].Output) * sigmoidDerivative;
            break;
          case ActivatorEnum.ReLU:
            layer.Neurons[i].Error = (inputs[i] - layer.Neurons[i].Output) * (layer.Neurons[i].Output > .0 ? 1.0 : .0);
            break;
          case ActivatorEnum.LReLU:
            if (layer.Neurons[i].Output >= .0)
              layer.Neurons[i].Error = inputs[i] - layer.Neurons[i].Output;
            else
              layer.Neurons[i].Error = (inputs[i] - layer.Neurons[i].Output) * layer.Neurons[i].LeakParameter;
            break;
          case ActivatorEnum.Tanh:
            double tanhDerivative = 1.0 - (layer.Neurons[i].Output * layer.Neurons[i].Output);
            layer.Neurons[i].Error = (inputs[i] - layer.Neurons[i].Output) * tanhDerivative;
            break;
          default:
            break;
        }

        if (lambda > .0)
        {
          // Add L1 regularization term
          double l1Regularization = lambda * Layers.Sum(l => l.Neurons.Sum(n => n.InputWeights.Sum(x => Math.Abs(x))));

          // Add L2 regularization term
          double l2Regularization = lambda * Layers.Sum(l => l.Neurons.Sum(n => n.InputWeights.Sum(w => w * w)));

          layer.Neurons[i].Error += l2Regularization + l1Regularization;
        }

      }
  }
  private double CalculateLoss(double[][] y)
  {
    double loss = .0;
    switch (_neuralNetworkTask)
    {
      case NeuralNetworkTaskEnum.Regression:
        loss = LossCalculator.RegressionLoss.MeanSquaredErrorLoss(y, GetOutputs());
        break;
      case NeuralNetworkTaskEnum.BinaryClassification:
        loss = LossCalculator.BinaryLoss.BinaryCrossEntropyLoss(y, GetOutputs());
        break;
      case NeuralNetworkTaskEnum.MultiClassClassification:
        loss = LossCalculator.SoftmaxLoss.SoftmaxCrossEntropyLoss(y, GetOutputs());
        break;
      default:
        break;
    }

    return loss;
  }

  private void UpdateWeights()
  {
    switch (_optimizer)
    {
      case OptimizerEnum.SGD:
        for (int i = Layers.Length - 1; i >= 0; i--)
          for (int j = 0; j < Layers[i].Neurons.Length; j++)
            StochasticGradientDescent(Layers[i]);
        break;
      case OptimizerEnum.Adam:
        Adam();
        break;
      case OptimizerEnum.RMSprop:
        RMSPropagation();
        break;
      default:
        throw new Exception($"Unknown optimizer. ({_optimizer})");
    }
  }
  private void StochasticGradientDescent(Layer layer, double lambda = .0)
  {
    for (int j = 0; j < layer.Neurons.Length; j++)
    {
      for (int l = 0; l < layer.Neurons[j].InputWeights.Length; l++)
      {
        var regularization = .0;
        if (lambda > .0)
        {
          // Apply L1 and L2 regularization to gradients
          var l1Regularization = Math.Sign(layer.Neurons[j].InputWeights[l]);
          var l2Regularization = 2 * layer.Neurons[j].InputWeights[l];
          regularization = lambda * (l1Regularization + l2Regularization);
        }
        var weightUpdate = -LearningRate * (layer.Neurons[j].Gradients[l] + regularization);
        layer.Neurons[j].InputWeights[l] += weightUpdate;
      }

      // Update bias
      layer.Neurons[j].Bias -= LearningRate * layer.Neurons[j].Gradients[^1];

      //Reset gradients for next epoch
      layer.Neurons[j].Gradients = new double[layer.Neurons[j].Gradients.Length];
    }
  }
  private void Adam(double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
  {
    var iterationCounter = 0;
    foreach (var layer in Layers)
    {
      var momentum = new Dictionary<int, double[]>();
      var rMSProp = new Dictionary<int, double[]>();

      foreach (var neuron in layer.Neurons)
      {
        int neuronIndex = Array.IndexOf(layer.Neurons, neuron);
        if (!momentum.ContainsKey(neuronIndex))
        {
          momentum[neuronIndex] = new double[neuron.InputWeights.Length];
          rMSProp[neuronIndex] = new double[neuron.InputWeights.Length];
        }

        for (int i = 0; i < neuron.InputWeights.Length; i++)
        {
          momentum[neuronIndex][i] = beta1 * momentum[neuronIndex][i] + (1 - beta1) * neuron.Gradients[i];
          rMSProp[neuronIndex][i] = beta2 * rMSProp[neuronIndex][i] + (1 - beta2) * neuron.Gradients[i] * neuron.Gradients[i];

          double mCorrected = momentum[neuronIndex][i] / (1 - Math.Pow(beta1, iterationCounter + 1));
          double vCorrected = rMSProp[neuronIndex][i] / (1 - Math.Pow(beta2, iterationCounter + 1));

          neuron.InputWeights[i] -= LearningRate * mCorrected / (Math.Sqrt(vCorrected) + epsilon);
        }
      }

      iterationCounter++;
    }
  }
  private void RMSPropagation(double beta = 0.999, double epsilon = 1e-8)
  {
    foreach (var layer in Layers)
    {
      var cache = new Dictionary<int, double[]>();
      foreach (var neuron in layer.Neurons)
      {
        int neuronIndex = Array.IndexOf(layer.Neurons, neuron);
        if (!cache.ContainsKey(neuronIndex))
          cache[neuronIndex] = new double[neuron.InputWeights.Length];

        for (int i = 0; i < neuron.InputWeights.Length; i++)
        {
          cache[neuronIndex][i] = beta * cache[neuronIndex][i] + (1 - beta) * neuron.Gradients[i] * neuron.Gradients[i];
          neuron.InputWeights[i] -= LearningRate * cache[neuronIndex][i] / (Math.Sqrt(cache[neuronIndex][i]) + epsilon);
        }
      }
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

      // Vector Normalization
      var magnitude = Math.Sqrt(matrix[k].Sum(x => x * x));
      if (magnitude > .0)
        matrix[k] = matrix[k].Select(x => x / magnitude).ToArray();
    }

    // Flatten the matrix into a weight vector
    double[] weights = matrix.SelectMany(row => row).ToArray();

    neuron.InputWeights = weights;
    neuron.Bias = 0;
  }

  private double[][] GetOutputs()
  {
    return Layers.Select(l => l.Neurons.Select(n => n.Output).ToArray()).ToArray();
  }
}

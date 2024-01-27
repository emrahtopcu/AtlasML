namespace AtlasML.Regression;
public static class LinearRegression
{
  /// <summary>
  /// 
  /// </summary>
  /// <param name="X">X : (array_like Shape (m,n)    matrix of examples</param>
  /// <param name="y">y : (array_like Shape (m,))    target value of each example</param>
  /// <param name="iterations">num_iters : (int) number of iterations to run gradient descent</param>
  /// <param name="alpha">alpha : (float) Learning rate</param>
  /// <param name="degree">n: scalar polinominal degree</param>
  /// <returns></returns>
  public static (double w, double b, List<double> J_history, List<double> w_history) Fit(double[] x, double[] y, int iterations = 1000, double alpha = 1e-6)
  {
    double w = .0;
    double b = .0;
    return GradientDescent(x, y, w, b, alpha, iterations);
  }

  /// <summary>
  /// Computes the cost function for linear regression.
  /// </summary>
  /// <param name="x">x (ndarray): Shape (m,) Input</param>
  /// <param name="y">y (ndarray): Shape (m,) Label</param>
  /// <param name="w">w, b (scalar): Parameters of the model</param>
  /// <param name="b">w, b (scalar): Parameters of the model</param>
  /// <returns>
  /// total_cost (double): The cost of using w,b as the parameters for linear regression to fit the data points in x and y
  /// </returns>
  /// <exception cref="Exception">Throw exception if length of array parameters are not equal.</exception>
  private static double ComputeCost(double[] x, double[] y, double w, double b)
  {
    if (x.Length != y.Length)
      throw new Exception("The lenght of x and y must be equal.");

    // number of training examples
    var m = x.Length;

    // You need to return this variable correctly
    var total_cost = 0.0f;

    for (var i = 0; i < m; i++)
    {
      var p = w * x[i] + b;
      var v = p - y[i];
      total_cost += Convert.ToSingle(v * v);
    }

    total_cost /= (2 * m);

    return total_cost;
  }

  /// <summary>
  /// Computes the gradient for linear regression.
  /// </summary>
  /// <param name="x">x (ndarray): Shape (m,) Input</param>
  /// <param name="y">y (ndarray): Shape (m,) Label</param>
  /// <param name="w">w, b (scalar): Parameters of the model</param>
  /// <param name="b">w, b (scalar): Parameters of the model</param>
  /// <returns>
  /// dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
  /// dj_db (scalar): The gradient of the cost w.r.t. the parameter b
  /// </returns>
  /// <exception cref="Exception">Throw exception if length of array parameters are not equal.</exception>
  private static (double dj_dw, double dj_db) ComputeGradient(double[] x, double[] y, double w, double b, double lambda = .0)
  {
    if (x.Length != y.Length)
      throw new Exception("The lenght of x and y must be equal.");

    // number of training examples
    var m = x.Length;

    // Return variables
    var dj_dw = 0.0;
    var dj_db = 0.0;

    for (int i = 0; i < m; i++)
    {
      var p = w * x[i] + b;
      dj_db += p - y[i];
      dj_dw += (p - y[i]) * x[i];
    }
    dj_db /= m;
    dj_dw /= m;

    //Regularization
    if (lambda > .0)
      dj_dw += (lambda / m) * w;

    return (dj_dw, dj_db);
  }

  /// <summary>
  /// Performs batch gradient descent to learn theta. Updates theta by taking num_iters gradient steps with learning rate alpha.
  /// </summary>
  /// <param name="x">x (ndarray): Shape (m,)</param>
  /// <param name="y">y (ndarray): Shape (m,)</param>
  /// <param name="w_in">w_in, b_in : (scalar) Initial values of parameters of the model</param>
  /// <param name="b_in">w_in, b_in : (scalar) Initial values of parameters of the model</param>
  /// <param name="alpha">(double) Learning rate</param>
  /// <param name="num_iters">(int) number of iterations to run gradient descent</param>
  /// <returns>
  /// w : (scalar) Updated values of parameters of the model after running gradient descent 
  /// b : (scalar) Updated value of parameter of the model after running gradient descent
  /// </returns>
  /// <exception cref="Exception">Throw exception if length of array parameters are not equal.</exception>
  private static (double w, double b, List<double> J_history, List<double> w_history) GradientDescent(double[] x, double[] y, double w_in, double b_in, double alpha, int num_iters)
  {
    if (x.Length != y.Length)
      throw new Exception("The lenght of x and y must be equal.");

    // number of training examples
    var m = x.Length;

    // An array to store cost J and w's at each iteration — primarily for graphing later
    var J_history = new List<double>();
    var w_history = new List<double>();

    // avoid modifying global w within function
    var w = w_in;
    var b = b_in;

    for (int i = 0; i < num_iters; i++)
    {
      //Calculate the gradient and update the parameters
      var (dj_dw, dj_db) = ComputeGradient(x, y, w, b);

      // Update Parameters using w, b, alpha and gradient
      w -= alpha * dj_dw;
      b -= alpha * dj_db;

      // Save cost J at each iteration
      var cost = ComputeCost(x, y, w, b);
      J_history.Add(cost);
      w_history.Add(w);
    }

    // return w and J,w history for graphing
    return (w, b, J_history, w_history);
  }

  /// <summary>
  /// Train model to find weight and bias for given train set.
  /// </summary>
  /// <param name="x_train">Target values of train set.</param>
  /// <param name="y_train">Label values of train set.</param>
  /// <returns></returns>
  public static (double w, double b, List<double> J_history, List<double> w_history, double[] linear_fit) Train(double[] x_train, double[] y_train)
  {
    // initialize fitting parameters. Recall that the shape of w is (n,)
    var initial_w = 0f;
    var initial_b = 0f;

    // some gradient descent settings
    var iterations = 1500;
    var alpha = 0.1f;

    var (w, b, J_history, w_history) = GradientDescent(x_train, y_train, initial_w, initial_b, alpha, iterations);

    var m = x_train.Length;
    var linear_fit = new double[m];
    for (int i = 0; i < m; i++)
      linear_fit[i] = w * x_train[i] + b;

    return (w, b, J_history, w_history, linear_fit);
  }

  /// <summary>
  /// Calculate regression function for given weight and bias for x.
  /// </summary>
  /// <param name="w"></param>
  /// <param name="b"></param>
  /// <param name="x"></param>
  /// <returns></returns>
  public static (double x, double y_hat) Predict(double w, double b, double x)
  {
    return (x, x * w + b);
  }
}

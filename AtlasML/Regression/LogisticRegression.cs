using AtlasML.Extensions;

namespace AtlasML.Regression;
public static class LogisticRegression
{
  /// <summary>
  /// Single predict using linear regression.
  /// </summary>
  /// <param name="x">ınput feautures.</param>
  /// <param name="w">Model parameters.</param>
  /// <param name="b">Bias</param>
  public static double[] Predict(double[,] X, double[] w, double b)
  {
    return X.Dot(w, b);
  }

  /// <summary>
  /// 
  /// </summary>
  /// <param name="X">X : (array_like Shape (m,n)    matrix of examples</param>
  /// <param name="y">y : (array_like Shape (m,))    target value of each example</param>
  /// <param name="iterations">num_iters : (int) number of iterations to run gradient descent</param>
  /// <param name="alpha">alpha : (float) Learning rate</param>
  /// <param name="degree">n: scalar polinominal degree</param>
  /// <returns></returns>
  public static (double[] w, double b, JHistory[] history) Fit(double[,] X, double[] y, int iterations = 1000, double alpha = 1e-6, double lambda = 1e-6)
  {
    var (m, n) = X.Shape();
    var w = new double[n];
    double b = .0;
    return GradientDescent(X, y, w, b, alpha, iterations, lambda);
  }

  /// <summary>
  /// Performs batch gradient descent to learn theta. Updates theta by taking num_iters gradient steps with learning rate alpha
  /// </summary>
  /// <param name="X">X : (array_like Shape (m,n)    matrix of examples </param>
  /// <param name="y">y : (array_like Shape (m,))    target value of each example</param>
  /// <param name="w_in">w_in : (array_like Shape (n,)) Initial values of parameters of the model</param>
  /// <param name="b_in">b_in : (scalar)                Initial value of parameter of the model</param>
  /// <param name="alpha">alpha : (float) Learning rate</param>
  /// <param name="num_iters">num_iters : (int) number of iterations to run gradient descent</param>
  /// <returns>
  /// w : (array_like Shape (n,)) Updated values of parameters of the model after running gradient descent
  /// b : (scalar)                Updated value of parameter of the model after running gradient descent
  /// </returns>
  public static (double[] w, double b, JHistory[] history) GradientDescent(double[,] X, double[] y, double[] w_in, double b_in, double alpha, int num_iters, double lambda)
  {
    var m = X.Length;
    var history = new List<JHistory>();

    var w = w_in.DeepCopy();
    var b = b_in;

    var consoleDvider = num_iters / 10.0;
    for (int i = 0; i < num_iters; i++)
    {
      //Calculate the gradient and update the parameters
      var (dj_db, dj_dw) = ComputeGradient(X, y, w, b);

      // Update Parameters using w, b, alpha and gradient
      w = w.Minus(dj_dw.Times(alpha));
      b -= alpha * dj_db;

      history.Add(new JHistory(i, ComputeCost(X, y, w, b, lambda), new Parameters(w, b), new Gradiant(dj_dw, dj_db)));

      if (i % Math.Ceiling(num_iters / consoleDvider) == 0)
        Console.WriteLine($"Iteration: {i}, Cost: {history[i].Cost}");
    }

    return (w, b, history.ToArray());
  }

  /// <summary>
  /// Computes the gradient for logistic regression.
  /// </summary>
  /// <param name="X">X (ndarray (m,n): Data, m examples with n features</param>
  /// <param name="y">y (ndarray (m,)): target values</param>
  /// <param name="w">w (ndarray (n,)): model parameters </param>
  /// <param name="b">b (scalar)      : model parameter</param>
  /// <returns>
  /// dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
  /// dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b.
  /// </returns>
  public static (double dj_db, double[] dj_dw) ComputeGradient(double[,] X, double[] y, double[] w, double b, double lambda = .0)
  {
    var (m, n) = X.Shape();
    var dj_dw = new double[n];
    var dj_db = .0;

    for (var i = 0; i < m; i++)
    {
      var f_wb_i = Sigmoid(X.GetRow(i).Dot(w) + b);
      var err_i = f_wb_i - y[i];
      for (int j = 0; j < n; j++)
        dj_dw[j] = dj_dw[j] + err_i * X[i, j];
      dj_db += err_i;
    }
    dj_dw = dj_dw.Over(m);
    dj_db /= m;

    if (lambda > .0)
      for (int i = 0; i < n; i++)
        dj_dw[i] += (lambda / m) * w[i];

    return (dj_db, dj_dw);
  }

  /// <summary>
  /// Computes cost using logistic loss, non-matrix version.
  /// </summary>
  /// <param name="X">X (ndarray): Shape (m,n)  matrix of examples with n features</param>
  /// <param name="y">y (ndarray): Shape (m,)   target values</param>
  /// <param name="w">w (ndarray): Shape (n,)   parameters for prediction</param>
  /// <param name="b">b (scalar):               parameter  for prediction</param>
  /// <param name="lambda">lambda : (scalar, float) Controls amount of regularization, 0 = no regularization</param>
  /// <returns>cost (scalar): cost</returns>
  public static double ComputeCost(double[,] X, double[] y, double[] w, double b, double lambda = .0)
  {
    var (m, n) = X.Shape();
    var cost = .0;
    for (int i = 0; i < m; i++)
    {
      var zi = X.GetRow(i).Dot(w) + b;
      var fwbi = Sigmoid(zi);
      cost += -y[i] * Math.Log(fwbi) - (1 - y[i]) * Math.Log(1 - fwbi);
    }
    cost /= m;

    var regCost = .0;
    if (lambda > .0)
    {
      for (int i = 0; i < n; i++)
        regCost += w[i] * w[i];
      regCost *= lambda / (2 * m);
    }

    return cost + regCost;
  }

  /// <summary>
  /// Compute the sigmoid of z.
  /// </summary>
  /// <param name="z">z : array_like 
  /// A scalar or numpy array of any size.</param>
  /// <returns>
  /// g : array_like sigmoid(z)
  /// </returns>
  public static double[] Sigmoid(double[] z)
  {
    double[] g = new double[z.Length];
    for (int i = 0; i < z.Length; i++)
      g[i] = Sigmoid(z[i]);
    return g;
  }

  /// <summary>
  /// Compute the sigmoid of z.
  /// </summary>
  /// <param name="z">z : scalar</param>
  /// <returns>
  /// g : scalar
  /// </returns>
  public static double Sigmoid(double z)
  {
    return 1.0 / (1.0 + Math.Exp(-z));
  }
}

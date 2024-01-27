using AtlasML.Extensions;

namespace AtlasML.Regression;
public static class PolinominalRegression
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
  /// Single predict using linear regression.
  /// </summary>
  /// <param name="x">ınput feautures.</param>
  /// <param name="w">Model parameters.</param>
  /// <param name="b">Bias</param>
  public static double[] Predict(double[][] X, double[] w, double b)
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
  public static (double[] w, double b, dynamic history) Fit(double[,] X, double[] y, int iterations = 1000, double alpha = 1e-6)
  {
    var (m, n) = X.Shape();
    var w = new double[n];
    double b = .0;
    return GradiantDescent(X, y, w, b, alpha, iterations);
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
  public static (double[] w, double b, dynamic history) Fit(double[][] X, double[] y, int iterations = 1000, double alpha = 1e-6)
  {
    var (m, n) = X.Shape();
    var w = new double[n];
    double b = .0;
    return GradiantDescent(X, y, w, b, alpha, iterations);
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
  public static (double[] w, double b, dynamic history) GradiantDescent(double[,] X, double[] y, double[] w_in, double b_in, double alpha, int num_iters)
  {
    var m = X.Length;
    dynamic history = new
    {
      Cost = new double[num_iters],
      Parameters = new Parameters[num_iters],
      Gradiants = new Gradiant[num_iters],
      Iterations = new double[num_iters]
    };

    var w = w_in.DeepCopy();
    var b = b_in;

    for (int i = 0; i < num_iters; i++)
    {
      //Calculate the gradient and update the parameters
      var (dj_db, dj_dw) = ComputeGradientMatrix(X, y, w, b);

      // Update Parameters using w, b, alpha and gradient
      w = w.Minus(dj_dw.Times(alpha));
      b -= alpha * dj_db;

      history.Cost[i] = ComputeCost(X, y, w, b);
      history.Parameters[i] = new Parameters(w, b);
      history.Gradiants[i] = new Gradiant(dj_dw, dj_db);
      history.Iterations[i] = i;
    }

    return (w, b, history);
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
  public static (double[] w, double b, dynamic history) GradiantDescent(double[][] X, double[] y, double[] w_in, double b_in, double alpha, int num_iters)
  {
    var m = X.Length;
    dynamic history = new
    {
      Cost = new double[num_iters],
      Parameters = new Parameters[num_iters],
      Gradiants = new Gradiant[num_iters],
      Iterations = new double[num_iters]
    };

    var w = w_in.DeepCopy();
    var b = b_in;

    for (int i = 0; i < num_iters; i++)
    {
      //Calculate the gradient and update the parameters
      var (dj_db, dj_dw) = ComputeGradientMatrix(X, y, w, b);

      // Update Parameters using w, b, alpha and gradient
      w = w.Minus(dj_dw.Times(alpha));
      b -= alpha * dj_db;

      history.Cost[i] = ComputeCost(X, y, w, b);
      history.Parameters[i] = new Parameters(w, b);
      history.Gradiants[i] = new Gradiant(dj_dw, dj_db);
      history.Iterations[i] = i;
    }

    return (w, b, history);
  }

  public static (double dj_db, double[] dj_dw) ComputeGradientMatrix(double[,] X, double[] y, double[] w, double b, double lambda = .0)
  {
    var (m, n) = X.Shape();
    var f_wb = X.Dot(w, b);
    var e = f_wb.Minus(y);
    var bias = 1.0 / m;
    var dj_dw = X.TransposeMatrix().MatMul(e).Times(bias);
    var dj_db = bias * e.Sum();

    //Regularization
    if (lambda > .0)
      for (int i = 0; i < n; i++)
        dj_dw[i] += (lambda / m) * w[i];

    return (dj_db, dj_dw);
  }
  public static (double dj_db, double[] dj_dw) ComputeGradientMatrix(double[][] X, double[] y, double[] w, double b, double lambda = .0)
  {
    var (m, n) = X.Shape();
    var f_wb = X.Dot(w, b);
    var e = f_wb.Minus(y);
    var bias = 1.0 / m;
    var dj_dw = X.TransposeMatrix().MatMul(e).Times(bias);
    var dj_db = bias * e.Sum();

    //Regularization
    if (lambda > .0)
      for (int i = 0; i < n; i++)
        dj_dw[i] += (lambda / m) * w[i];

    return (dj_db, dj_dw);
  }

  /// <summary>
  /// Compute cost.
  /// </summary>
  /// <param name="X">X : (ndarray): Shape (m,n) matrix of examples with multiple features</param>
  /// <param name="y">y : (array_like Shape (m,))    target value of each example</param>
  /// <param name="w">w : (ndarray): Shape (n)   parameters for prediction</param>
  /// <param name="b">b : (scalar):              parameter  for prediction</param>
  /// <returns>
  /// cost: (scalar)             cost
  /// </returns>
  public static double ComputeCost(double[,] X, double[] y, double[] w, double b)
  {
    var (m, n) = X.Shape();
    var cost = 0.0;
    for (int i = 0; i < m; i++)
    {
      var f_wb_i = X.GetRow(i).Dot(w) + b;
      var v = f_wb_i - y[i];
      cost += v * v;
    }
    cost /= (2.0 * m);
    return cost;
  }

  /// <summary>
  /// Compute cost.
  /// </summary>
  /// <param name="X">X : (ndarray): Shape (m,n) matrix of examples with multiple features</param>
  /// <param name="y">y : (array_like Shape (m,))    target value of each example</param>
  /// <param name="w">w : (ndarray): Shape (n)   parameters for prediction</param>
  /// <param name="b">b : (scalar):              parameter  for prediction</param>
  /// <returns>
  /// cost: (scalar)             cost
  /// </returns>
  public static double ComputeCost(double[][] X, double[] y, double[] w, double b)
  {
    var (m, n) = X.Shape();
    var cost = 0.0;
    for (int i = 0; i < m; i++)
    {
      var f_wb_i = X[i].Dot(w) + b;
      var v = f_wb_i - y[i];
      cost += v * v;
    }
    cost /= (2.0 * m);
    return cost;
  }
}

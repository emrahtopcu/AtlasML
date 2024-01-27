using AtlasML.Extensions;
using System.Numerics;

namespace AtlasML.Regression;
public static class MultiFeatureLinearRegression
{
  /// <summary>
  /// Single predict using linear regression.
  /// </summary>
  /// <param name="x">ınput feautures.</param>
  /// <param name="w">Model parameters.</param>
  /// <param name="b">Bias</param>
  public static T Predict<T>(T[] x, T[] w, T b) where T : struct, INumber<T>
  {
    return Vector.Dot(new Vector<T>(x), new Vector<T>(w)) + b;
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
    return GradientDescent(X, y, w, b, alpha, iterations);
  }

  /// <summary>
  /// Compute cost.
  /// </summary>
  /// <param name="X">X (ndarray (m,n)): Data, m examples with n features</param>
  /// <param name="y">y (ndarray (m,)) : target values</param>
  /// <param name="w">w (ndarray (n,)) : model parameters </param>
  /// <param name="b">b (scalar)       : model parameter</param>
  /// <returns></returns>
  public static double ComputeCost(double[,] X, double[] y, double[] w, double b, double lambda = 1.0)
  {
    var (m, n) = X.Shape();
    var cost = 0.0;
    for (int i = 0; i < X.Length; i++)
    {
      var f_wb_i = X.GetRow(i).Dot(w) + b;
      var v = f_wb_i - y[i];
      cost += v * v;
    }
    cost /= (2 * m);

    var regCost = .0;
    for (int i = 0; i < n; i++)
      regCost += w[i] * w[i];
    regCost = (lambda / (2.0 * m)) * regCost;

    return cost + regCost;
  }

  /// <summary>
  /// Computes the gradient for linear regression.
  /// </summary>
  /// <param name="X">X (ndarray (m,n)): Data, m examples with n features</param>
  /// <param name="y">y (ndarray (m,)) : target values</param>
  /// <param name="w">w (ndarray (n,)) : model parameters </param>
  /// <param name="b">b (scalar)       : model parameter</param>
  /// <returns>
  /// dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
  /// dj_db (scalar):       The gradient of the cost w.r.t. the parameter b.
  /// </returns>
  public static (double dj_db, double[] dj_dw) ComputeGradient(double[,] X, double[] y, double[] w, double b, double lambda = .0)
  {
    var (m, n) = X.Shape();
    var dj_dw = new double[n];
    var dj_db = .0;

    for (int i = 0; i < m; i++)
    {
      var err = (X.GetRow(i).Dot(w) + b) - y[i];
      for (int j = 0; j < n; j++)
        dj_dw[j] = dj_dw[j] + err * X[i, j];
      dj_db += err;
    }
    dj_dw = dj_dw.Over(m);
    dj_db /= m;

    //Regularization
    if (lambda > .0)
      for (int i = 0; i < n; i++)
        dj_dw[i] += (lambda / m) * w[i];

    return (dj_db, dj_dw);
  }

  /// <summary>
  /// Performs batch gradient descent to learn w and b. Updates w and b by taking num_iters gradient steps with learning rate alpha.
  /// </summary>
  /// <param name="X">X (ndarray (m,n))   : Data, m examples with n features</param>
  /// <param name="y">y (ndarray (m,))    : target values</param>
  /// <param name="w_in">w_in (ndarray (n,)) : initial model parameters</param>
  /// <param name="b_in">b_in (scalar)       : initial model parameter</param>
  /// <param name="alpha">alpha (float)       : Learning rate</param>
  /// <param name="num_iters">num_iters (int)     : number of iterations to run gradient descent</param>
  /// <returns>
  /// w (ndarray (n,))        : Updated values of parameters
  /// b (scalar)              : Updated value of parameter
  /// J_history (num_iters,2) : An array to store cost J and w's at each iteration primarily for graphing later
  /// </returns>
  public static (double[] w, double b, List<JHistory> J_history) GradientDescent(double[,] X, double[] y, double[] w_in, double b_in, double alpha, int num_iters)
  {
    var J_history = new List<JHistory>();
    var w = new double[w_in.Length];
    Array.Copy(w_in, w, w_in.Length);
    var b = b_in;

    for (int i = 0; i < num_iters; i++)
    {
      // Calculate the gradient and update the parameters
      var (dj_db, dj_dw) = ComputeGradient(X, y, w, b);

      //Update Parameters using w, b, alpha and gradient
      for (int j = 0; j < w.Length; j++)
        w[j] -= alpha * dj_dw[j];
      b -= alpha * dj_db;

      //Save cost J at each iteration
      if (i < 100000)
        J_history.Add(new(i, ComputeCost(X, y, w, b), new Parameters(w.DeepCopy(), b), new Gradiant(dj_dw, dj_db)));
    }

    return (w, b, J_history);
  }
}

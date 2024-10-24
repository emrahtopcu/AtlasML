using AtlasML.Common;
using AtlasML.Extensions;

namespace AtlasML.Normalization;
public static class ZScoreNormalization
{
  /// <summary>
  /// computes  X, zcore normalized by column
  /// </summary>
  /// <param name="X">input data, m examples, n features</param>
  /// <returns>input normalized by column</returns>
  public static double[] Normalize(double[] x)
  {
    var normilazed = new double[x.Length];
    var mu = x.Average(x => x);
    var sigma = Convert.ToSingle(Math.Sqrt(x.Average(v => (v - mu) * (v - mu))));
    for (int i = 0; i < x.Length; i++)
      normilazed[i] = (x[i] - mu) / sigma;
    return normilazed;
  }

  /// <summary>
  /// computes  X, zcore normalized by column
  /// </summary>
  /// <param name="x">input data</param>
  /// <returns>
  /// </returns>
  public static (double[], double mu, double sigma) FitNormalize(double[] x)
  {
    var normilazed = new double[x.Length];
    var mu = x.Average(x => x);
    var sigma = Convert.ToSingle(Math.Sqrt(x.Average(v => (v - mu) * (v - mu))));
    for (int i = 0; i < x.Length; i++)
      normilazed[i] = (x[i] - mu) / sigma;
    return (normilazed, mu, sigma);
  }

  public static double[] DeNormalize(double[] x, double mu, double sigma)
  {
    var deNorm = new double[x.Length];
    for (int i = 0; i < x.Length; i++)
      deNorm[i] = x[i] * sigma + mu;
    return deNorm;
  }

  /// <summary>
  /// Computes  X, zcore normalized by column.
  /// </summary>
  /// <param name="X">X (ndarray (m,n))     : input data, m examples, n features</param>
  /// <returns>
  /// X_norm (ndarray (m,n)): input normalized by column
  /// mu (ndarray (n,))     : mean of each feature
  /// sigma (ndarray (n,))  : standard deviation of each feature
  /// </returns>
  public static (double[][] X_norm, double[] mu, double[] sigma) Normalize(double[][] X)
  {
    var colCount = X[0].Length;

    var mu = new double[colCount];
    var sigma = new double[colCount];

    for (int i = 0; i < colCount; i++)
    {
      var vector = X.Col(i);
      (sigma[i], mu[i]) = StandardDeviation.StdDev(vector);
    }

    var X_norm = ArrayExtensions.JaggedArrayInitialize<double[][]>(X.Length, colCount);
    for (int i = 0; i < X.Length; i++)
      for (int j = 0; j < colCount; j++)
      {
        if (sigma[j] != .0)
          X_norm[i][j] = (X[i][j] - mu[j]) / sigma[j];
        else
          X_norm[i][j] = X[i][j];
      }

    return (X_norm, mu, sigma);
  }

  /// <summary>
  /// Computes  X, zcore normalized by column.
  /// </summary>
  /// <param name="X">X (ndarray (m,n))     : input data, m examples, n features</param>
  /// <returns>
  /// X_norm (ndarray (m,n)): input normalized by column
  /// mu (ndarray (n,))     : mean of each feature
  /// sigma (ndarray (n,))  : standard deviation of each feature
  /// </returns>
  public static (double[,] X_norm, double[] mu, double[] sigma) Normalize(double[,] X)
  {
    int numRows = X.GetLength(0);
    int numCols = X.GetLength(1);
    double[] mu = new double[numCols];
    double[] sigma = new double[numCols];

    // Calculate mean (mu) and standard deviation (sigma) by column
    for (int j = 0; j < numCols; j++)
    {
      double[] column = new double[numRows];
      for (int i = 0; i < numRows; i++)
        column[i] = X[i, j];

      (sigma[j], mu[j]) = StandardDeviation.StdDev(column);
    }

    // Initialize the normalized matrix
    double[,] normalizedMatrix = new double[numRows, numCols];
    for (int i = 0; i < numRows; i++)
    {
      for (int j = 0; j < numCols; j++)
      {
        if (sigma[j] != 0)
          normalizedMatrix[i, j] = (X[i, j] - mu[j]) / sigma[j];
        else
          normalizedMatrix[i, j] = X[i, j];
      }
    }

    return (normalizedMatrix, mu, sigma);
  }

  public static double[,] DiNormalize(double[,] X_norm, double[] mu, double[] sigma)
  {
    int numRows = X_norm.GetLength(0);
    int numCols = X_norm.GetLength(1);

    double[,] X = new double[numRows, numCols];
    for (int i = 0; i < numRows; i++)
      for (int j = 0; j < numCols; j++)
        X[i, j] = X_norm[i, j] * sigma[j] + mu[j];

    return X;
  }
}

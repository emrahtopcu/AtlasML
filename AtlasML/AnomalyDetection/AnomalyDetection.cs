using AtlasML.Extensions;

namespace AtlasML.AnomalyDetection;
public static class AnomalyDetection
{
  //Density Estimation
  //Gaussian Estimation
  public static (double[] mean, double[] sigma) Execute(double[][] X)
  {
    var mean = new double[X[0].Length];
    var sigma = new double[X[0].Length];
    for (int i = 0; i < X[0].Length; i++)
    {
      mean[i] = X.Col(i).Average();
      sigma[i] = X.Col(i).Sum(x => Math.Pow(x - mean[i], 2)) / X[i].Length;
    }

    return (mean, sigma);
  }
  public static double CalculateP(double x, double m, double s)
  {
    return 1 / Math.Sqrt(2 * Math.PI * s) * Math.Exp(-(Math.Pow((x - m), 2) / (2 * s)));
  }
}

public class OutlierDetection
{
  public static (double[] mu, double[] var) EstimateGaussian(double[][] X)
  {
    int m = X.Length;
    int n = X[0].Length;

    double[] mu = new double[n];
    double[] var = new double[n];

    for (int j = 0; j < n; j++)
    {
      double sum = X.Sum(row => row[j]);
      mu[j] = sum / m;

      double squaredSum = X.Sum(row => Math.Pow(row[j] - mu[j], 2));
      var[j] = squaredSum / m;
    }

    return (mu, var);
  }
  public static (double epsilon, double F1) SelectThreshold(double[] y_val, double[] p_val)
  {
    double best_epsilon = 0;
    double best_F1 = 0;
    double F1 = 0;

    double step_size = (p_val.Max() - p_val.Min()) / 1000;

    for (double epsilon = p_val.Min(); epsilon < p_val.Max(); epsilon += step_size)
    {
      int tp = p_val.Where((p, idx) => p < epsilon && y_val[idx] == 1).Count();
      int fp = p_val.Where((p, idx) => p < epsilon && y_val[idx] == 0).Count();
      int fn = p_val.Where((p, idx) => p >= epsilon && y_val[idx] == 1).Count();

      double precision = (tp + fp) > 0 ? (double)tp / (tp + fp) : 0;
      double recall = (tp + fn) > 0 ? (double)tp / (tp + fn) : 0;
      F1 = (precision + recall) > 0 ? (2 * precision * recall) / (precision + recall) : 0;

      if (F1 > best_F1)
      {
        best_F1 = F1;
        best_epsilon = epsilon;
      }
    }

    return (best_epsilon, best_F1);
  }
}

using AtlasML.Extensions;

namespace AtlasML.RecommendationSystem;
public static class CollaborativeRecSys
{
  public static void Run()
  {

  }
  public static double CoFiCost(double[][] X, double[][] W, double[] b, double[][] Y, double[][] R, double lambda)
  {
    var J = .0;
    for (int i = 0; i < R.Length; i++)
      for (int j = 0; j < R[i].Length; j++)
        J += R[i][j] * Math.Pow(W[j].Dot(X[i]) + b[j] - Y[i][j], 2);
    J /= 2.0;

    //Regularization
    if (lambda > .0)
    {
      var reg0 = .0;
      for (int j = 0; j < W.Length; j++)
        for (int k = 0; k < W[j].Length; k++)
          reg0 += W[j][k] * W[j][k];
      reg0 *= lambda / 2.0;

      var reg1 = .0;
      for (int i = 0; i < X.Length; i++)
        for (int k = 0; k < X[i].Length; k++)
          reg1 += X[i][k] * X[i][k];
      reg1 *= lambda / 2.0;

      J += reg0 + reg1;
    }

    return J;
  }
}

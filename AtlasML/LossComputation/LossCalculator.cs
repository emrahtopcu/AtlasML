namespace AtlasML.LossComputation;
public static class LossCalculator
{
  public static class RegressionLoss
  {
    //MSE: For Regression
    public static double SquaredErrorLoss(double y, double yHat)
    {
      return (yHat - y) * (yHat - y);
    }
    public static double MeanSquaredErrorLoss(double[] y, double[] yHat)
    {
      if (y.Length != yHat.Length)
        throw new ArgumentException("Input arrays must have the same length.");

      return yHat.Select((x, i) => SquaredErrorLoss(y[i], x)).Sum() / y.Length;
    }
    public static double MeanSquaredErrorLoss(double[][] y, double[][] yHat)
    {
      if (y.Select((x, i) => x.Length != yHat[i].Length).Any())
        throw new ArgumentException("Inner arrays must have the same length.");

      return y.Select((x, i) => MeanSquaredErrorLoss(x, yHat[i])).Sum() / y.Length;
    }
  }
  public static class BinaryLoss
  {
    //Binary Cross-Entropy Loss, Logaritmic Loss, Log Loss : For Binary Classification
    public static double BinaryCrossEntropyLoss(double y, double yHat)
    {
      return -(y * Math.Log(yHat) + (1 - y) * Math.Log(1 - yHat));
    }
    public static double BinaryCrossEntropyLoss(double[] y, double[] yHat)
    {
      if (y.Length != yHat.Length)
        throw new ArgumentException("Input arrays must have the same length.");

      return y.Select((x, i) => BinaryCrossEntropyLoss(y, yHat)).Sum() / y.Length;
    }
    public static double BinaryCrossEntropyLoss(double[][] y, double[][] yHat)
    {
      if (y.Select((x, i) => x.Length != yHat[i].Length).Any())
        throw new ArgumentException("Inner arrays must have the same length.");

      return y.Select((x, i) => BinaryCrossEntropyLoss(x, yHat[i])).Sum() / y.Length;
    }
  }
  public static class SoftmaxLoss
  {
    //Categorical Cross-Entropy Loss, Softmax Cross-Entropy Loss
    public static double SoftmaxCrossEntropyLoss(double[][] y, double[][] yHat)
    {
      if (y.Length != yHat.Length || y.Any(innerArray => innerArray.Length != yHat[0].Length))
        throw new ArgumentException("Input arrays must have the correct dimensions.");

      int n = y.Length;
      int c = yHat[0].Length;

      // Calculate the cross-entropy values
      double[] crossEntropyValues = new double[n];
      for (int i = 0; i < n; i++)
        crossEntropyValues[i] = -Enumerable.Range(0, c).Sum(j => y[i][j] * Math.Log(yHat[i][j]));

      // Calculate the Categorical Cross-Entropy Loss
      double cce = crossEntropyValues.Sum() / n;

      return cce;
    }
  }
}

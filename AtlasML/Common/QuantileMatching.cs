namespace AtlasML.Common;
public static class QuantileMatching
{
  public static List<double> QuantileMatch(List<double> data1, List<double> data2, int numQuantiles = 100)
  {
    double[] quantiles = GenerateQuantiles(numQuantiles); // Function defined below
    double[] quantilesData1 = CalculateQuantilesForData(data1, quantiles);
    double[] quantilesData2 = CalculateQuantilesForData(data2, quantiles);

    List<double> alignedData1 = [];
    foreach (double value in data1)
    {
      // Find the corresponding aligned value using the interpolation function
      double alignedValue = LinearInterpolation.Interpolate(quantilesData1, quantilesData2, value);
      alignedData1.Add(alignedValue);
    }

    return alignedData1;
  }
  private static double[] GenerateQuantiles(int numQuantiles)
  {
    // Generates an array of quantile values between 0 and 1
    double[] quantiles = new double[numQuantiles];
    for (int i = 0; i < numQuantiles; i++)
    {
      quantiles[i] = (double)(i + 1) / (numQuantiles + 1);
    }
    return quantiles;
  }
  private static double[] CalculateQuantilesForData(List<double> data, double[] quantiles)
  {
    // Calculates the data values corresponding to the specified quantiles
    double[] quantileValues = new double[quantiles.Length];
    for (int i = 0; i < quantiles.Length; i++)
    {
      quantileValues[i] = Percentile.Compute(data, quantiles[i] * 100);
    }
    return quantileValues;
  }
}

namespace AtlasML.Common;
public static class StandardDeviation
{
  // Function to calculate the mean of an array
  public static double Mean(double[] array) => array.Average();

  /// <summary>
  /// Welford's standart deviation.
  /// </summary>
  /// <param name="array">Polulation</param>
  /// <param name="isSample">Is array whole or sample population.</param>
  /// <returns></returns>
  public static (double std, double mean) StdDev(double[] array, bool isSample = false)
  {
    double M = 0.0;
    double S = 0.0;
    int k = 1;
    foreach (double value in array)
    {
      double tmpM = M;
      M += (value - tmpM) / k;
      S += (value - tmpM) * (value - M);
      k++;
    }

    if (isSample)
      return (Math.Sqrt(S / (k - 2)), M);
    else
      return (Math.Sqrt(S / (k - 1)), M);
  }
}

namespace AtlasML.Normalization;
public static class MinMaxNormalization
{
  public static (double[] normalized, double min, double max) Normalize(double[] source)
  {
    double minValue = source.Min();
    double maxValue = source.Max();
    double[] normalizedArray = new double[source.Length];

    for (int i = 0; i < source.Length; i++)
    {
      normalizedArray[i] = (source[i] - minValue) / (maxValue - minValue);
    }
    return (normalizedArray, minValue, maxValue);
  }
  public static double[] ReverseNormalization(double[] normalizedArray, double minValue, double maxValue)
  {
    double[] originalArray = new double[normalizedArray.Length];
    for (int i = 0; i < normalizedArray.Length; i++)
      originalArray[i] = normalizedArray[i] * (maxValue - minValue) + minValue;
    return originalArray;
  }
}

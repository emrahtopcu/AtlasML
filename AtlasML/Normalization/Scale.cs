namespace AtlasML.Normalization;
public static class Scale
{
  public static double[] ReScale(double[] firstSeries, double[] secondSeries)
  {
    // Calculate the average of the first series
    double firstSeriesAverage = firstSeries.Average();

    // Calculate the average of the second series
    double secondSeriesAverage = secondSeries.Average();

    // Calculate the scaling factor
    double scalingFactor = firstSeriesAverage / secondSeriesAverage;

    // Scale the second series by the scaling factor
    double[] scaledSecondSeries = secondSeries.Select(x => x * scalingFactor).ToArray();

    return scaledSecondSeries;
  }
}

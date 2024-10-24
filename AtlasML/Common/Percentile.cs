namespace AtlasML.Common;
public static class Percentile
{
  public static double Compute(List<double> sortedData, double percentile)
  {
    // Ensure data is sorted
    sortedData.Sort();

    // Compute the position of the percentile
    double position = (percentile / 100.0) * (sortedData.Count + 1);

    // Find the integer and fractional parts of the position
    int index = (int)position;
    double fraction = position - index;

    if (index <= 0)
    {
      return sortedData.First();
    }
    else if (index >= sortedData.Count)
    {
      return sortedData.Last();
    }
    else
    {
      // Linear interpolation between the two surrounding values
      return sortedData[index - 1] + fraction * (sortedData[index] - sortedData[index - 1]);
    }
  }
}

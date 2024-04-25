namespace AtlasML.Correlation;

/// <summary>
/// Spearman’s, rs , assesses how well an arbitrary monotonic function describes the relationship between two variables, rather than specifically testing for a linear association. A monotonic relationship is one where as one variable increases or decreases, the other variable consistently increases or decreases. This allows Spearman’s correlation to identify nonlinear relationships that may not be evident when using Pearson’s r. It does so by first ranking all data points in each variable from smallest to largest value, then calculating r using these ranks rather than the original measurements. As such, Spearman’s rs is more flexible and can identify more complex monotonic correlations beyond linear trends alone
/// </summary>
public static class SpearmanCorrelation
{
  public static double Measere(double[] x, double[] y)
  {
    if (x.Length != y.Length)
      throw new Exception("Invalid vector sizes.");

    // Create ranked copies of the data
    var rankedX = Rank(x);
    var rankedY = Rank(y);

    // Calculate the difference in ranks
    var d = Enumerable.Zip(rankedX, rankedY, (a, b) => Math.Pow(a - b, 2)).ToArray();

    // Calculate Spearman's rank correlation coefficient
    var n = x.Length;
    var rho = 1 - (6 * d.Sum()) / (n * (n * n - 1));

    return rho;
  }

  // Helper function to rank data with handling of ties
  private static double[] Rank(double[] data)
  {
    var rankings = new Dictionary<double, List<int>>();
    var sortedData = data.OrderBy(d => d).ToArray();

    for (int i = 0; i < sortedData.Length; i++)
    {
      if (!rankings.ContainsKey(sortedData[i]))
        rankings.Add(sortedData[i], new List<int>());

      rankings[sortedData[i]].Add(i + 1);
    }

    var ranked = new double[data.Length];
    int index = 0;
    foreach (var item in rankings)
    {
      var value = item.Key;
      var positions = item.Value;
      double averageRank = positions.Average();
      foreach (var pos in positions)
      {
        ranked[index++] = averageRank;
      }
    }

    return ranked;
  }
}

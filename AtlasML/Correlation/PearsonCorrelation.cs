using AtlasML.Common;

namespace AtlasML.Correlation;

/// <summary>
/// To calculate Pearson’s r, a line of best fit is determined for the two variables using linear regression. This regression line represents the linear relationship that best predicts the values of one variable based on the other. The correlation coefficient is then computed based on how far each data point deviates from this regression line. Data points that lie exactly on the line have a deviation of zero, while points farther away have higher deviations. Pearson’s r factors in both the direction and magnitude of all these deviations to produce a measure between -1 and 1, indicating the overall linear association between the variables. A value closer to the extremes represents less deviation and stronger linear correlation, while a value near zero suggests the data are poorly described by a linear relationship.
/// </summary>
public static class PearsonCorrelation
{

  /// <summary>
  /// Pearson’s r can correlate variables measured on different scales. For example, it could assess the relationship between average temperature (measured in degrees Celsius) and number of ice cream sales (measured as a daily count).
  /// As a dimensionless index, r is unaffected by the original measurement units or scale. Whether temperature was in Fahrenheit or sales in dollars wouldn’t alter the correlation value.
  /// The correlation computed remains identical regardless of how the variables are labeled (e.g. as independent or dependent). Examining if sales drive temperature changes versus temperature influencing sales would generate the same r result, as the coefficient considers only covariation between paired observations.
  /// </summary>
  /// <param name="x"></param>
  /// <param name="y"></param>
  /// <returns>Pearson’s correlation coefficient</returns>
  /// <exception cref="Exception"></exception>
  public static double Measure(double[] x, double[] y)
  {
    if (x.Length != y.Length)
    {
      throw new Exception("Invalid vector sizes.");
    }

    var meanX = x.Average();
    var meanY = y.Average();

    var covariance = x.Select((xVal, i) => (xVal - meanX) * (y[i] - meanY)).Sum();
    var varianceX = StandardDeviation.PopulationVariance(x);
    var varianceY = StandardDeviation.PopulationVariance(y);

    // Handle potential division by zero
    if (varianceX * varianceY == 0)
      return .0;

    var r = covariance / Math.Sqrt(varianceX * varianceY);
    return r;
  }
}

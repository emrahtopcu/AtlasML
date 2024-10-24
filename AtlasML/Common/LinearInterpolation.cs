namespace AtlasML.Common;
public static class LinearInterpolation
{
  public static double Interpolate(double[] x, double[] y, double value)
  {
    if (x.Length != y.Length)
      throw new ArgumentException("The lengths of x and y must be equal.");

    int n = x.Length;

    // Find the interval [x_i, x_(i+1)] that contains value
    for (int i = 0; i < n - 1; i++)
    {
      if (value >= x[i] && value <= x[i + 1])
      {
        // Apply the linear interpolation formula
        return y[i] + (y[i + 1] - y[i]) * (value - x[i]) / (x[i + 1] - x[i]);
      }
    }

    // If value is out of bounds, throw an exception or handle accordingly
    throw new ArgumentOutOfRangeException("value", "The value to interpolate is outside the range of the given data points.");
  }
}

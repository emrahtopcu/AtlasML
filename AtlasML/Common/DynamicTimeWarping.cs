using AtlasML.Normalization;

namespace AtlasML.Common;
public static class DynamicTimeWarping
{
  public static (double distance, List<(int, int)> path) CalculateDTW(double[] series1, double[] series2)
  {
    int n = series1.Length;
    int m = series2.Length;
    double[,] dtw = new double[n + 1, m + 1];

    for (int i = 0; i <= n; i++)
      for (int j = 0; j <= m; j++)
        dtw[i, j] = double.PositiveInfinity;
    dtw[0, 0] = 0;

    for (int i = 1; i <= n; i++)
    {
      for (int j = 1; j <= m; j++)
      {
        double cost = Math.Abs(series1[i - 1] - series2[j - 1]);
        dtw[i, j] = cost + Math.Min(Math.Min(dtw[i - 1, j],    // insertion
                                             dtw[i, j - 1]),   // deletion
                                             dtw[i - 1, j - 1]); // match
      }
    }

    // Traceback from dtw[n, m] to dtw[0, 0] to get the path
    List<(int, int)> path = [];
    int a = n, b = m;
    path.Add((a - 1, b - 1));

    while (a > 1 || b > 1)
    {
      if (a == 1)
        b--;
      else if (b == 1)
        a--;
      else
      {
        double[] options = [dtw[a - 1, b], dtw[a, b - 1], dtw[a - 1, b - 1]];
        double minOption = Math.Min(Math.Min(options[0], options[1]), options[2]);

        if (minOption == options[0])
          a--;
        else if (minOption == options[1])
          b--;
        else
        {
          a--;
          b--;
        }
      }
      path.Add((a - 1, b - 1));
    }

    // Reverse the path to start from (0, 0)
    path.Reverse();

    return (dtw[n, m], path);
  }
  public static double CalculateNormilizedDTW(double[] series1, double[] series2)
  {
    int n = series1.Length;
    int m = series2.Length;
    var costMatrix = new double[n, m];

    // Step 1: Calculate the cost matrix
    for (int i = 0; i < n; i++)
      for (int j = 0; j < m; j++)
        costMatrix[i, j] = Math.Abs(series1[i] - series2[j]);

    // Step 2: Flatten the cost matrix
    double[] flatCostMatrix = FlattenMatrix(costMatrix);

    // Step 3: Apply Min-Max normalization
    var (normalizedFlatCostMatrix, min, max) =  MinMaxNormalization.Normalize(flatCostMatrix);

    // Step 4: Reshape the normalized cost matrix back to its original shape
    double[,] normalizedCostMatrix = ReshapeArray(normalizedFlatCostMatrix, n, m);

    // Step 5: Calculate DTW distance using the normalized cost matrix
    double[,] dtw = new double[n + 1, m + 1];

    for (int i = 0; i <= n; i++)
    {
      for (int j = 0; j <= m; j++)
      {
        dtw[i, j] = double.PositiveInfinity;
      }
    }
    dtw[0, 0] = 0;

    for (int i = 1; i <= n; i++)
    {
      for (int j = 1; j <= m; j++)
      {
        double cost = normalizedCostMatrix[i - 1, j - 1];
        dtw[i, j] = cost + Math.Min(Math.Min(dtw[i - 1, j],    // insertion
                                             dtw[i, j - 1]),   // deletion
                                             dtw[i - 1, j - 1]); // match
      }
    }

    return dtw[n, m];
  }
  private static double[] FlattenMatrix(double[,] matrix)
  {
    int rows = matrix.GetLength(0);
    int cols = matrix.GetLength(1);
    double[] flatArray = new double[rows * cols];
    int index = 0;
    for (int i = 0; i < rows; i++)
    {
      for (int j = 0; j < cols; j++)
      {
        flatArray[index++] = matrix[i, j];
      }
    }
    return flatArray;
  }
  private static double[,] ReshapeArray(double[] flatArray, int rows, int cols)
  {
    double[,] matrix = new double[rows, cols];
    int index = 0;
    for (int i = 0; i < rows; i++)
    {
      for (int j = 0; j < cols; j++)
      {
        matrix[i, j] = flatArray[index++];
      }
    }
    return matrix;
  }
}

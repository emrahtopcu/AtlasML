namespace AtlasML.KMeansClustering;
public static class KMeansCluster
{
  public static (double[][], int[]) Run(double[][] X, double[][] initialCentroids, int maxIters = 10, bool plotProgress = false)
  {
    int m = X.Length;
    int n = X[0].Length;
    int K = initialCentroids.Length;
    double[][] centroids = initialCentroids;
    double[][] previousCentroids = centroids;
    int[] idx = new int[m];

    // Run K-Means
    for (int i = 0; i < maxIters; i++)
    {
      // Output progress
      Console.WriteLine($"K-Means iteration {i}/{maxIters - 1}");

      // For each example in X, assign it to the closest centroid
      idx = FindClosestCentroids(X, centroids);

      // Optionally plot progress
      if (plotProgress)
      {
        // Plotting code here
        // Assuming this functionality is handled elsewhere
        // plotProgress_kMeans(X, centroids, previousCentroids, idx, K, i);
        previousCentroids = centroids;
      }

      // Given the memberships, compute new centroids
      centroids = ComputeCentroids(X, idx, K);
    }

    return (centroids, idx);
  }
  public static double[][] InitCentroids(double[][] X, int K)
  {
    int m = X.Length;
    int n = X[0].Length;

    // Randomly reorder the indices of examples
    Random rnd = new Random();
    int[] randIndices = Enumerable.Range(0, m).OrderBy(x => rnd.Next()).ToArray();

    // Take the first K examples as centroids
    double[][] centroids = new double[K][];
    for (int i = 0; i < K; i++)
    {
      centroids[i] = new double[n];
      Array.Copy(X[randIndices[i]], centroids[i], n);
    }

    return centroids;
  }
  public static int[] FindClosestCentroids(double[][] X, double[][] centroids)
  {
    int K = centroids.Length;
    int m = X.Length;
    int[] idx = new int[m];

    for (int i = 0; i < m; i++)
    {
      double[] distances = new double[K];
      for (int j = 0; j < K; j++)
      {
        double norm_ij = Norm(X[i], centroids[j]);
        distances[j] = norm_ij;
      }
      idx[i] = Array.IndexOf(distances, distances.Min());
    }

    return idx;
  }
  public static double[][] ComputeCentroids(double[][] X, int[] idx, int K)
  {
    int m = X.Length;
    int n = X[0].Length;
    double[][] centroids = new double[K][];

    for (int k = 0; k < K; k++)
    {
      var points = X.Where((_, i) => idx[i] == k).ToArray();
      centroids[k] = ComputeMean(points, n);
    }

    return centroids;
  }
  private static double[] ComputeMean(double[][] points, int n)
  {
    double[] mean = new double[n];
    for (int j = 0; j < n; j++)
    {
      double sum = points.Sum(p => p[j]);
      mean[j] = sum / points.Length;
    }
    return mean;
  }
  public static double Norm(double[] x, NormOrderEnum ord = NormOrderEnum.None)
  {
    return ord switch
    {
      NormOrderEnum.None => Math.Abs(x.Sum()),
      NormOrderEnum.L2 => Math.Sqrt(x.Select(val => val * val).Sum()),
      NormOrderEnum.Inf => x.Select(Math.Abs).Max(),
      NormOrderEnum.MinusInf => x.Select(Math.Abs).Min(),
      _ => throw new ArgumentException("Invalid norm order."),
    };
  }
  public static double Norm(double[,] x, NormOrderEnum ord = NormOrderEnum.None)
  {
    return ord switch
    {
      NormOrderEnum.None or NormOrderEnum.Frobenius => Math.Sqrt(x.Cast<double>().Select(val => val * val).Sum()),
      NormOrderEnum.Nuclear => throw new NotImplementedException("Nuclear norm calculation is not implemented."),
      _ => throw new ArgumentException("Invalid norm order for matrices."),
    };
  }
  public static double Norm(double[] x, int axis, NormOrderEnum ord = NormOrderEnum.None)
  {
    if (axis != 0)
      throw new ArgumentException("Axis must be 0 for 1-D arrays.");

    return Norm(x, ord);
  }
  public static double Norm(double[,] x, int axis, NormOrderEnum ord = NormOrderEnum.None)
  {
    if (axis == 0)
      return Norm(x, ord);
    else if (axis == 1)
      throw new NotImplementedException("Axis = 1 not yet supported for 2-D arrays.");
    else
      throw new ArgumentException("Invalid axis value.");
  }
  public static double Norm(double[] x, double[] y, NormOrderEnum ord = NormOrderEnum.L2)
  {
    if (x.Length != y.Length)
      throw new ArgumentException("Arrays must have the same length.");

    switch (ord)
    {
      case NormOrderEnum.L2:
        double sumOfSquares = 0.0;
        for (int i = 0; i < x.Length; i++)
        {
          double diff = x[i] - y[i];
          sumOfSquares += diff * diff;
        }
        return Math.Sqrt(sumOfSquares);
      default:
        throw new ArgumentException("Invalid or not supported norm order.");
    }
  }
}

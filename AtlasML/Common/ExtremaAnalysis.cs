namespace AtlasML.Common;
public static class ExtremaAnalysis<T> where T : IComparable<T>
{
  public static void FindLocalExtrema(T[] data, out List<T> maxima, out List<T> minima)
  {
    maxima = [];
    minima = [];

    for (int i = 1; i < data.Length - 1; i++)
      if (data[i].CompareTo(data[i - 1]) > 0 && data[i].CompareTo(data[i + 1]) > 0)
        maxima.Add(data[i]);
      else if (data[i].CompareTo(data[i - 1]) < 0 && data[i].CompareTo(data[i + 1]) < 0)
        minima.Add(data[i]);
  }

  /// <summary>
  /// 
  /// </summary>
  /// <param name="data"></param>
  /// <param name="maxima">Array of indexes of local maximums.</param>
  /// <param name="minima">Array of indexes of local minimums.</param>
  public static void FindLocalExtrema(T[] data, out List<int> maxima, out List<int> minima)
  {
    maxima = [];
    minima = [];

    for (int i = 1; i < data.Length - 1; i++)
      if (data[i].CompareTo(data[i - 1]) > 0 && data[i].CompareTo(data[i + 1]) > 0)
        maxima.Add(i);
      else if (data[i].CompareTo(data[i - 1]) < 0 && data[i].CompareTo(data[i + 1]) < 0)
        minima.Add(i);
  }
}

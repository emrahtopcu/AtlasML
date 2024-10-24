using System.Diagnostics;
using System.Runtime.InteropServices;

namespace AtlasML.Extensions;
public static class ArrayExtensions
{
  public static double Dot(this double[] v1, double[] v2)
  {
    if (v1.Length != v2.Length)
      throw new ArgumentException("Invalid lenght.");

    var dp = .0;
    for (int i = 0; i < v1.Length; i++)
      dp += v1[i] * v2[i];

    return dp;
  }
  public static double[] Dot(this double[,] matrix, double[] v2, double bias = .0)
  {
    if (matrix.GetLength(1) != v2.Length)
      throw new ArgumentException("Invalid lenght.");

    var (m, n) = matrix.Shape();
    var dp = new double[m];
    for (int i = 0; i < m; i++)
    {
      for (int j = 0; j < n; j++)
        dp[i] += matrix[i, j] * v2[j];
      dp[i] += bias;
    }

    return dp;
  }
  public static double[] Dot(this double[][] matrix, double[] v2, double bias = .0)
  {
    if (matrix[0].Length != v2.Length)
      throw new ArgumentException("Invalid lenght.");

    var (m, n) = matrix.Shape();
    var dp = new double[m];
    for (int i = 0; i < m; i++)
    {
      for (int j = 0; j < n; j++)
        dp[i] += matrix[i][j] * v2[j];
      dp[i] += bias;
    }

    return dp;
  }
  public static double[] DeepCopy(this double[] arr)
  {
    var newArr = new double[arr.Length];
    for (int i = 0; i < arr.Length; i++)
      newArr[i] = arr[i];
    return newArr;
  }
  public static (int m, int n) Shape(this double[,] arr)
  {
    return (arr.GetLength(0), arr.GetLength(1));
  }
  public static (int m, int n) Shape(this double[][] arr)
  {
    return (arr.Length, arr[0].Length);
  }
  public static double[] Col(this double[][] arr, int colIndex)
  {
    var newArr = new double[arr.Length];
    for (int i = 0; i < arr.Length; i++)
      newArr[i] = arr[i][colIndex];
    return newArr;
  }
  public static T[] GetRow<T>(this T[,] array, int row)
  {
    if (!typeof(T).IsPrimitive)
      throw new InvalidOperationException("Not supported for managed types.");

    if (array == null)
      throw new ArgumentNullException("array");

    int cols = array.GetUpperBound(1) + 1;
    T[] result = new T[cols];

    int size;

    if (typeof(T) == typeof(bool))
      size = 1;
    else if (typeof(T) == typeof(char))
      size = 2;
    else
      size = Marshal.SizeOf<T>();

    Buffer.BlockCopy(array, row * cols * size, result, 0, cols * size);

    return result;
  }
  public static double[] GetColumn(this double[,] matrix, int columnNumber)
  {
    return Enumerable.Range(0, matrix.GetLength(0))
            .Select(x => matrix[x, columnNumber])
            .ToArray();
  }
  public static IEnumerable<double[]> EnumerateRows(this double[,] X)
  {
    for (int i = 0; i < X.GetLength(0); i++)
      yield return X.GetRow(i);
  }
  public static double[] MatMul(this double[,] X, double[] w, double bias = .0)
  {
    int numRows = X.GetLength(0);
    int numCols = X.GetLength(1);
    double[] result = new double[numRows];

    for (int i = 0; i < numRows; i++)
      for (int j = 0; j < numCols; j++)
        result[i] += X[i, j] * w[j] + bias;

    return result;
  }
  public static double[] MatMul(this double[][] X, double[] w, double bias = .0)
  {
    int numRows = X.Length;
    int numCols = X[0].Length;
    double[] result = new double[numRows];

    for (int i = 0; i < numRows; i++)
      for (int j = 0; j < numCols; j++)
        result[i] += X[i][j] * w[j] + bias;

    return result;
  }
  public static double[] Plus(this double[] vector, double bias)
  {
    int length = vector.Length;
    double[] result = new double[length];

    for (int i = 0; i < length; i++)
      result[i] = vector[i] + bias;

    return result;
  }
  public static double[] Times(this double[] vector, double bias)
  {
    int length = vector.Length;
    double[] result = new double[length];

    for (int i = 0; i < length; i++)
      result[i] = vector[i] * bias;

    return result;
  }
  public static double[] Over(this double[] vector, double bias)
  {
    int length = vector.Length;
    double[] result = new double[length];

    for (int i = 0; i < length; i++)
      result[i] = vector[i] / bias;

    return result;
  }
  public static double[] Minus(this double[] vector1, double[] vector2)
  {
    if (vector1.Length != vector2.Length)
      throw new ArgumentException("Vectors must have the same length for subtraction.");

    int length = vector1.Length;
    double[] result = new double[length];

    for (int i = 0; i < length; i++)
      result[i] = vector1[i] - vector2[i];

    return result;
  }
  public static double[,] TransposeMatrix(this double[,] X)
  {
    int numRows = X.GetLength(0);
    int numCols = X.GetLength(1);
    double[,] result = new double[numCols, numRows];

    for (int i = 0; i < numRows; i++)
      for (int j = 0; j < numCols; j++)
        result[j, i] = X[i, j];

    return result;
  }
  public static double[][] TransposeMatrix(this double[][] X)
  {
    int numRows = X.Length;
    int numCols = X[0].Length;
    var result = ArrayExtensions.JaggedArrayInitialize<double[][]>(numCols, numRows);

    for (int i = 0; i < numRows; i++)
      for (int j = 0; j < numCols; j++)
        result[j][i] = X[i][j];

    return result;
  }
  public static double[][] Reshape(this double[][] source, int rows, int columns)
  {
    var newShape = new double[rows][];
    for (int i = 0; i < source.Length; i++)
    {
      newShape[i] = new double[columns];
      for (int j = 0; j < columns; j++)
        newShape[i][j] = source[i][j];
    }
    return newShape;
  }
  public static double[][] Reshape(this double[][] source, int rows, Range columns)
  {
    var newShape = new double[rows][];
    for (int i = 0; i < source.Length; i++)
      newShape[i] = source[i][columns];
    return newShape;
  }
  public static double[,] To2D(this double[] x, int n = 1)
  {
    double[,] X = new double[x.Length, n];
    for (int i = 0; i < x.Length; i++)
      X[i, 0] = x[i];
    return X;
  }
  public static double[][] ToJagged(this double[] x)
  {
    var X = new double[x.Length][];
    for (int i = 0; i < x.Length; i++)
      X[i] = [x[i]];
    return X;
  }
  public static T[,] To2D<T>(this T[][] source)
  {
    try
    {
      int FirstDim = source.Length;
      //throws InvalidOperationException if source is not rectangular
      int SecondDim = source.GroupBy(row => row.Length).Single().Key;

      var result = new T[FirstDim, SecondDim];
      for (int i = 0; i < FirstDim; ++i)
        for (int j = 0; j < SecondDim; ++j)
          result[i, j] = source[i][j];

      return result;
    }
    catch (InvalidOperationException)
    {
      throw new InvalidOperationException("The given jagged array is not rectangular.");
    }
  }
  public static void Print(this double[,] X)
  {
    Console.WriteLine($"[{string.Join(Environment.NewLine, X.EnumerateRows().Select((s, i) => $"[{string.Join(" ", X.GetRow(i).Select(x => x))}]"))}]");
  }
  public static string Print(this double[] x)
  {
    var text = "";
    if (x.Length < 15)
      text = $"[{string.Join(" ", x.Select(v => v.ToString("e")))}]";
    else
      text = $"[{string.Join("\r\n", x.Select(v => v.ToString("e")))}]";
    Console.WriteLine(text);
    return text;
  }
  public static void Print(this double[][] X)
  {
    Console.Write($"[{string.Join(Environment.NewLine, X.Select((s, i) => $"[{string.Join(" ", X[i].Select(x => x))}]"))}]");
    Trace.Write($"[{string.Join(Environment.NewLine, X.Select((s, i) => $"[{string.Join(" ", X[i].Select(x => x))}]"))}]");
  }
  public static void Shuffle<T>(this T[] array)
  {
    var rng = new Random(Environment.TickCount);
    rng.Shuffle(array);
  }
  public static T JaggedArrayInitialize<T>(params int[] lengths)
  {
    return (T)InitializeJaggedArray(typeof(T).GetElementType(), 0, lengths);
  }
  public static object InitializeJaggedArray(Type type, int index, int[] lengths)
  {
    Array array = Array.CreateInstance(type, lengths[index]);
    Type elementType = type.GetElementType();

    if (elementType != null)
      for (int i = 0; i < lengths[index]; i++)
        array.SetValue(InitializeJaggedArray(elementType, index + 1, lengths), i);

    return array;
  }
  public static T[] Initialize<T>(T seed, int length)
  {
    var array = new T[length];
    for (int i = 0; i < length; i++)
      array[i] = seed;
    return array;
  }
  public static void Populate<T>(this T[] arr, T value)
  {
    for (int i = 0; i < arr.Length; i++)
      arr[i] = value;
  }
  public static T[] GetItems<T>(this T[] source, int[] indexes)
  {
    var resultSet = new T[indexes.Length];
    for (int i = 0; i < indexes.Length; i++)
      resultSet[i] = source[indexes[i]];
    return resultSet;
  }
  public static double[] Average(double[][] X)
  {
    var avgArray = new double[X[0].Length];
    for (int i = 0; i < X[0].Length; i++)
      avgArray[i] = X.Col(i).Average();
    return avgArray;
  }
  public static double[] Average(double[,] X)
  {
    var avgArray = new double[X.GetLength(1)];
    for (int i = 0; i < X.Length; i++)
      avgArray[i] = X.GetColumn(i).Average();
    return avgArray;
  }
}

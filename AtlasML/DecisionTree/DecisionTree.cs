using AtlasML.Extensions;

namespace AtlasML.DecisionTree;
public static class DecisionTree
{
  public static double CalcEntropy(double p)
  {
    if (p is .0 or 1.0)
      return .0;
    else
      return -p * Math.Log2(p) - (1.0 - p) * Math.Log2(1.0 - p);
  }

  public static void Ecetute()
  {
    var Xtrain = ArrayExtensions.JaggedArrayInitialize<double[][]>(10, 3);
    Xtrain[0] = [1, 1, 1];
    Xtrain[1] = [0, 0, 1];
    Xtrain[2] = [0, 1, 0];
    Xtrain[3] = [1, 0, 1];
    Xtrain[4] = [1, 1, 1];
    Xtrain[5] = [1, 1, 0];
    Xtrain[6] = [0, 0, 0];
    Xtrain[7] = [1, 1, 0];
    Xtrain[8] = [0, 1, 0];
    Xtrain[9] = [0, 1, 0];
    var yTrain = new double[10] { 1, 1, 0, 0, 1, 1, 0, 1, 0, 0 };

    //([0, 3, 4, 5, 7], [1, 2, 6, 8, 9])
    var (left, right) = Split(Xtrain, 0);

    //0.7219280948873623
    var weightedEntropy = CalcWeightedEntropy(Xtrain, yTrain, left, right);

    //0.2780719051126377
    var informationGain = CalcInformationgain(Xtrain, yTrain, left, right);

    var featureNames = new string[] { "Ear Shape", "Face Shape", "Whiskers" };
    for (int i = 0; i < featureNames.Length; i++)
    {
      var (leftIndices, rightIndices) = Split(Xtrain, i);
      var iGain = CalcInformationgain(Xtrain, yTrain, leftIndices, rightIndices);
      Console.WriteLine($"Feature: {featureNames[i]}, information gain if we split the root node using this feature: {iGain:n2}");
    }
  }
  
  private static (double[] left, double[] right) Split(double[][] X, int indexOfFeatureToSplit)
  {
    var left = new List<double>();
    var right = new List<double>();
    foreach (var node in X)
    {
      if (node[indexOfFeatureToSplit] == 1)
        left.Add(node[indexOfFeatureToSplit]);
      else
        right.Add(node[indexOfFeatureToSplit]);
    }

    return (left.ToArray(), right.ToArray());
  }
  private static double CalcWeightedEntropy(double[][] X, double[] y, double[] left, double[] right)
  {
    var wLeft = left.Length / X.Length;
    var wRight = right.Length / X.Length;
    var pLeft = y.GetItems(left.Select(Convert.ToInt32).ToArray()).Sum() / left.Length;
    var pRight = y.GetItems(right.Select(Convert.ToInt32).ToArray()).Sum() / right.Length;
    var weightedEntropy = wLeft * CalcEntropy(pLeft) + wRight * CalcEntropy(pRight);
    return weightedEntropy;
  }
  private static double CalcInformationgain(double[][] X, double[] y, double[] left, double[] right)
  {
    var pNode = y.Sum() / y.Length;
    var hNode = CalcEntropy(pNode);
    var wEntropy = CalcWeightedEntropy(X, y, left, right);
    return hNode - wEntropy;
  }
}

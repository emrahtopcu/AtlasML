using AtlasML.Extensions;

namespace AtlasMLTester;
public static class AnomalyDetectionExample
{
  //Example of engines. Engines have 2 features: heath,vibration.
  public static void Execute()
  {
    var rnd = new Random();

    //10000 normal engine features
    var X = ArrayExtensions.JaggedArrayInitialize<double[][]>(10000, 2);
    for (int i = 0; i < 10000; i++)
    {
      X[i][0] = rnd.NextGaussian();
      X[i][1] = Math.Sqrt(rnd.NextGaussian() * Math.PI);
    }

    //20 flowed engine features
    var Y_flawedEngines = ArrayExtensions.JaggedArrayInitialize<double[][]>(20, 2);
    for (int i = 0; i < 20; i++)
    {
      Y_flawedEngines[i][0] = rnd.NextGaussian();
      Y_flawedEngines[i][1] = Math.Sqrt(rnd.NextGaussian() * Math.PI);
    };

    var i8 = Convert.ToInt32(X.Length * 0.8);
    var i6 = Convert.ToInt32(X.Length * 0.6);
    var X_train = X[..i6];
    var X_cv = X[i6..i8].Concat(Y_flawedEngines[..10]).ToArray();
    var X_test = X[i8..].Concat(Y_flawedEngines[10..]).ToArray();

    var y_cv = new double[X[i6..i8].Length].Concat(ArrayExtensions.Initialize<double>(1.0, 10)).ToArray();
    var y_test = new double[X[i8..].Length].Concat(ArrayExtensions.Initialize<double>(1.0, 10)).ToArray();

    /*
     * Actual Class		Precision Class
     * 1								1										True Positive
     * 1								0										False Positive
     * 0								1										False Negative
     * 0								0										True Negative
     * 
     * Precission = TruePositive / (TruePositive + FalsePositive) => of all patients where predicted 1 what fraction actually are 1
     * Recall = TruePositive / (TruePositive + FalseNegative) => of all patients are accually 1 what fraction model correctly predicted 1
     * f1Score = 2 * (precision * recall) / (precision + recall);
     */

    var epsilon = 0.02;
    var (mean, sigma) = AtlasML.AnomalyDetection.AnomalyDetection.Execute(X_train);
    var optimizationData = ArrayExtensions.JaggedArrayInitialize<double[][]>(1000, 2);
    var y_hat = new double[X_cv.Length];
    for (int i = 0; i < 1000; i++)
    {
      var tp = .0;
      var fp = .0;
      var fn = .0;
      var tn = .0;

      for (int k = 0; k < X_cv.Length; k++)
      {
        var p_of_x = 1.0;
        for (int j = 0; j < 2; j++)
          p_of_x *= AtlasML.AnomalyDetection.AnomalyDetection.CalculateP(X_cv[k][j], mean[j], sigma[j]);
        
        y_hat[k] = p_of_x < epsilon ? 1.0 : .0;
        if (y_cv[k] == 1.0 && y_hat[k] == 1.0)
          tp++;
        else if (y_cv[k] == 1.0 && y_hat[k] == .0)
          fp++;
        else if (y_cv[k] == .0 && y_hat[k] == 1.0)
          fn++;
        else if (y_cv[k] == .0 && y_hat[k] == .0)
          tn++;
      }

      var p = tp / (tp + fp);
      var r = tp / (tp + fn);
      var f1 = 2 * (p * r) / (p + r);
      optimizationData[i][0] = epsilon;
      optimizationData[i][1] = f1;

      epsilon -= epsilon * 0.01;
    }

    var optimizedEpsilon = optimizationData.MaxBy(x => x[1])[0];
    var truePositive = .0;
    var falsePositive = .0;
    var falseNegative = .0;
    var trueNegative = .0;
    for (int k = 0; k < X_test.Length; k++)
    {
      var p_of_x = 1.0;
      for (int j = 0; j < 2; j++)
        p_of_x *= AtlasML.AnomalyDetection.AnomalyDetection.CalculateP(X_cv[k][j], mean[j], sigma[j]);

      y_hat[k] = p_of_x < optimizedEpsilon ? 1.0 : .0;
      if (y_cv[k] == 1.0 && y_hat[k] == 1.0)
        truePositive++;
      else if (y_cv[k] == 1.0 && y_hat[k] == .0)
        falsePositive++;
      else if (y_cv[k] == .0 && y_hat[k] == 1.0)
        falseNegative++;
      else if (y_cv[k] == .0 && y_hat[k] == .0)
        trueNegative++;
    }

    var precision = truePositive / (truePositive + falsePositive);
    var recall = truePositive / (truePositive + falseNegative);
    var f1Score = 2 * (precision * recall) / (precision + recall);
    Console.WriteLine($"precision: {precision}");
    Console.WriteLine($"recall: {recall}");
    Console.WriteLine($"f1Score: {f1Score}");
  }
}

using AtlasML.Extensions;
using AtlasML.Normalization;
using AtlasML.Regression;
using ScottPlot.WinForms;
using ScottPlot;

namespace AtlasMLTester;
public class PolinomialRegressionExamples
{
  public static void ExecuteComplex()
  {
    double[] x = Enumerable.Range(0, 20).Select(i => (double)i).ToArray();
    double[] y = x.Select(xi => Math.Cos(xi / 2.0)).ToArray();
    var X = CreatePolynomialFeatures(x, 13);
    X.Print();
    var (X_norm, mu, sigma) = ZScoreNormalization.Normalize(X);
    var (model_w, model_b, history) = PolynomialRegression.Fit(X_norm, y, 1000000, 1e-1);
    var predict = PolynomialRegression.Predict(X_norm, model_w, model_b);

    var plt = new Plot();
    var scatterAv = plt.Add.Scatter(xs: x, ys: y);
    scatterAv.Label = "Actual Value";
    var scatterP = plt.Add.Scatter(xs: x, ys: predict);
    scatterP.Label = "Predict";
    plt.ShowLegend();
    FormsPlotViewer.Launch(plt, "Polinominal Regression", 600, 400);
  }
  public static double[][] CreatePolynomialFeatures(double[] x, int degree)
  {
    var result = ArrayExtensions.JaggedArrayInitialize<double[][]>(x.Length, degree + 1);
    for (int i = 0; i < x.Length; i++)
      for (int j = 0; j < degree; j++)
        result[i][j] = j == 0 ? x[i] : Math.Pow(x[i], j + 1);
    return result;
  }
}

namespace AtlasML.Regression;
public class Parameters
{
  public double[] W { get; set; }
  public double B { get; set; }

  public Parameters(double[] w, double b)
  {
    this.W = w;
    this.B = b;
  }
}

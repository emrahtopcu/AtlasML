namespace AtlasML.Regression;
public class Gradiant
{
  public double[] DjDw { get; set; }
  public double DjDb { get; set; }

  public Gradiant(double[] djDw, double djDb)
  {
    DjDw = djDw;
    DjDb = djDb;
  }
}

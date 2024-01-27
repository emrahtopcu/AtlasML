namespace AtlasML.Regression;
public class JHistory
{
  public int I { get; set; }
  public double Cost { get; set; }
  public Parameters Parameters { get; set; }
  public Gradiant Gradiant { get; set; }

  public JHistory(int i, double cost, Parameters p, Gradiant g)
  {
    I = i;
    Cost = cost;
    Parameters = p;
    Gradiant = g;
  }
}

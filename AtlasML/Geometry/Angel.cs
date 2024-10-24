namespace AtlasML.Geometry;
public static class Angel
{
  public static double CalcAngel(double[] line, int lenght)
  {
    var scaledVlues = line.Select(x => x * 1000.0).ToArray();
    var vertex = new Point(1.0, scaledVlues[0]);
    var p1 = new Point(lenght, scaledVlues[^1]);
    return CalcAngel(vertex, p1);
  }
  public static double CalcAngel(Point p1, Point p2)
  {
    var dx = p2.X - p1.X;
    var dy = p2.Y - p1.Y;
    var theta = Math.Atan2(dy, dx);
    var angel = theta * (180 / Math.PI);
    return angel;
  }
}

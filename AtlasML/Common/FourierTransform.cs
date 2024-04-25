using System.Numerics;

namespace AtlasML.Common;
public static class FourierTransform
{
  public static Complex[] CooleyTukeyFFT(Complex[] x)
  {
    int N = x.Length;

    // Base case
    if (N <= 1)
      return x;

    // Split the array into even and odd parts
    var even = new Complex[N / 2];
    var odd = new Complex[N / 2];

    for (int i = 0; i < N / 2; i++)
    {
      even[i] = x[2 * i];
      odd[i] = x[2 * i + 1];
    }

    // Recursively compute the FFT for even and odd parts
    even = CooleyTukeyFFT(even);
    odd = CooleyTukeyFFT(odd);

    var combined = new Complex[N];
    for (int i = 0; i < N / 2; i++)
    {
      double theta = -2 * Math.PI * i / N;
      Complex twiddleFactor = Complex.FromPolarCoordinates(1, theta);

      combined[i] = even[i] + twiddleFactor * odd[i];
      combined[i + N / 2] = even[i] - twiddleFactor * odd[i];
    }

    return combined;
  }
}

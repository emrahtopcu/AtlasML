using AtlasML.Extensions;
using AtlasML.Regression;
using AtlasMLTester.Models;
using System.Text.Json;

namespace AtlasMLTester;
public static class NeuralNetworkBinaryClassificationExample
{
  public static void Execute()
  {
    var raw = File.ReadAllText("Data\\creditcard.json");
    var data = JsonSerializer.Deserialize<CreditCardFraudModel[]>(raw);

    //Balance classes
    var positiveCount = data.Count(x => x.Class == 1);
    var negativeCount = data.Count(x => x.Class == 0);
    if (positiveCount < negativeCount)
      data = data.Where(x => x.Class == 1).Take(positiveCount).Concat(data.Where(x => x.Class == 0).Take(positiveCount)).ToArray();
    else
      data = data.Where(x => x.Class == 1).Take(negativeCount).Concat(data.Where(x => x.Class == 0).Take(negativeCount)).ToArray();

    data.Shuffle();

    var X = data.Select((x, i) => new double[]
    {
      x.V1,
      x.V2,
      x.V3,
      x.V4,
      x.V5,
      x.V6,
      x.V7,
      x.V8,
      x.V9,
      x.V10,
      x.V11,
      x.V12,
      x.V13,
      x.V14,
      x.V15,
      x.V16,
      x.V17,
      x.V18,
      x.V19,
      x.V20,
      x.V21,
      x.V22,
      x.V23,
      x.V24,
      x.V25,
      x.V26,
      x.V27,
      x.V28,
    }).ToArray();
    var y = data.Select(x => Convert.ToDouble(x.Class)).ToArray();
    var halfIndex = X.Length / 2;
    var XTrain = X[0..halfIndex];
    var yTrain = y[0..halfIndex];

    var network = AtlasML.NeuralNetwork.Network.CreateBinaryClassificationNetwork([28, 14, 7, 1], 1000);
    network.Train(XTrain, y.ToJagged());

    
  }
}

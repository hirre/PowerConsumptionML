using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("Training data...");

        Train();

        Console.WriteLine("Predicting data...");

        Predict();

        Console.ReadKey();
    }

    private static void Train()
    {
        var mlContext = new MLContext();

        // Log experiment trials
        mlContext.Log += (_, e) =>
        {
            if (e.Source.Equals("AutoMLExperiment"))
            {
                Console.WriteLine(e.RawMessage);
            }
        };

        // Load data
        var data = mlContext.Data.LoadFromTextFile<PowerConsumptionData>(
            "simulated_electricity_consumption.csv", hasHeader: true, separatorChar: ',');

        Preview(data);

        // Split into Train/Test
        var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

        // Create AutoML Regression Experiment
        var experiment = mlContext.Auto()
            .CreateRegressionExperiment(maxExperimentTimeInSeconds: 60);  // 1 min

        // Run AutoML
        var result = experiment.Execute(split.TrainSet, labelColumnName: nameof(PowerConsumptionData.ConsumptionKwh));

        Console.WriteLine($"Best Model R-squared: {result.BestRun.ValidationMetrics.RSquared:P2}");

        // Save the best model
        mlContext.Model.Save(result.BestRun.Model, split.TrainSet.Schema, "PowerConsumptionModel.zip");
    }

    private static void Predict()
    {
        // Create MLContext
        MLContext mlContext = new();

        // Load and predict
        var loadedModel = mlContext.Model.Load("PowerConsumptionModel.zip", out _);
        var predictionEngine = mlContext.Model.CreatePredictionEngine<PowerConsumptionData, ConsumptionPrediction>(loadedModel);

        var startDate = new DateTime(2024, 12, 21, 15, 0, 0); // Example start date
        var nextDateTime = startDate.AddHours(1);

        // Predict for the next 24 hours
        for (int i = 0; i < 24; i++)
        {
            var newPowerConsumption = new PowerConsumptionData { Month = nextDateTime.Month, Day = nextDateTime.Day, Hour = nextDateTime.Hour };
            var prediction = predictionEngine.Predict(newPowerConsumption);

            Console.WriteLine($"Predicted consumption ({nextDateTime}): {prediction.Consumption:0.##} KWh");

            nextDateTime = nextDateTime.AddHours(1); // Increment the hour
        }
    }

    private static void Preview(IDataView dataView)
    {
        var preview = dataView.Preview();

        foreach (var row in preview.RowView)
        {
            Console.WriteLine(string.Join(", ", row.Values));
        }
    }
}

#region Models

// Define the prediction output class
public class ConsumptionPrediction
{
    [ColumnName("Score")]
    public float Consumption { get; set; }
}

// Define data schema
public class PowerConsumptionData
{
    [LoadColumn(0)]
    public float Month { get; set; }

    [LoadColumn(1)]
    public float Day { get; set; }

    [LoadColumn(2)]
    public float Hour { get; set; }

    [LoadColumn(3)]
    public float ConsumptionKwh { get; set; }
}

#endregion
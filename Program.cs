using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;

class Program
{
    static async Task Main(string[] args)
    {
        Console.WriteLine("Training data...");

        await Train();

        Console.WriteLine("Predicting data...");

        Predict();

        Console.ReadKey();
    }

    private static async Task Train()
    {
        // Define file path
        string dataPath = "simulated_electricity_consumption.csv";

        // Create MLContext
        var mlContext = new MLContext();

        //Preview(dataView);

        ColumnInferenceResults columnInference =
        mlContext.Auto().InferColumns(dataPath, labelColumnName: "ConsumptionKwh", groupColumns: false);

        TextLoader loader = mlContext.Data.CreateTextLoader(columnInference.TextLoaderOptions);
        IDataView data = loader.Load(dataPath);
        TrainTestData trainValidationData = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

        SweepablePipeline pipeline =
        mlContext.Auto().Featurizer(data, columnInformation: columnInference.ColumnInformation)
        .Append(mlContext.Auto().Regression(labelColumnName: columnInference.ColumnInformation.LabelColumnName));

        AutoMLExperiment experiment = mlContext.Auto().CreateExperiment();

        experiment
        .SetPipeline(pipeline)
        .SetRegressionMetric(RegressionMetric.RSquared, labelColumn: columnInference.ColumnInformation.LabelColumnName)
        .SetTrainingTimeInSeconds(60)
        .SetDataset(trainValidationData);

        // Log experiment trials
        mlContext.Log += (_, e) =>
        {
            if (e.Source.Equals("AutoMLExperiment"))
            {
                Console.WriteLine(e.RawMessage);
            }
        };

        TrialResult experimentResults = await experiment.RunAsync();

        // Save the model
        string modelPath = "PowerConsumptionModel.zip";
        mlContext.Model.Save(experimentResults.Model, data.Schema, modelPath);

        Console.WriteLine($"Model trained and saved to {modelPath}");
    }

    private static void Predict()
    {
        // Create MLContext
        MLContext mlContext = new MLContext();

        // Load Trained Model
        DataViewSchema predictionPipelineSchema;
        ITransformer predictionPipeline = mlContext.Model.Load("PowerConsumptionModel.zip", out predictionPipelineSchema);

        // Create a prediction engine
        var predictionEngine = mlContext.Model.CreatePredictionEngine<PowerConsumptionData, ConsumptionPrediction>(predictionPipeline);

        var inputData = new PowerConsumptionData[10];
        var startDate = new DateTime(2024, 12, 21, 15, 0, 0); // Example start date

        for (int i = 0; i < 10; i++)
        {
            // Prepare input data
            var input = new PowerConsumptionData
            {
                Timestamp = startDate.AddHours(i).ToString("o"), // ISO 8601 format
                ConsumptionKwh = 0 // Set a default value if required
            };

            inputData[i] = input;
        }

        // Predict data
        var predictions = inputData.Select(input => predictionEngine.Predict(input));

        foreach (var prediction in predictions)
        {
            Console.WriteLine($"Predicted Consumption: {prediction.Consumption}");
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
    public float Consumption { get; set; }
}

// Define data schema
public class PowerConsumptionData
{
    [LoadColumn(0)]
    public required string Timestamp { get; set; }

    [LoadColumn(1)]
    public float ConsumptionKwh { get; set; }
}

#endregion
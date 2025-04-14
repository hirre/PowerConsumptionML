using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

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
        // Define file path
        string dataPath = "simulated_electricity_consumption.csv";

        // Create MLContext
        var mlContext = new MLContext();

        // Load data
        IDataView dataView = mlContext.Data.LoadFromTextFile<PowerConsumptionData>(
            dataPath,
            hasHeader: true,
            separatorChar: ',');

        // Update the pipeline to use the factory directly instead of the attribute
        var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Consumption", inputColumnName: "ConsumptionKwh")
           .Append(mlContext.Transforms.CustomMapping(
               new ExtractHourAction().GetMapping(), contractName: "ExtractHourMapping"))
           .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "HourEncoded", inputColumnName: "Hour"))
           .Append(mlContext.Transforms.Concatenate("Features", "HourEncoded"))
           .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Consumption", featureColumnName: "Features"));

        // Train the model
        var model = pipeline.Fit(dataView);

        //PrintTransformedData(mlContext, model.Transform(dataView));

        // Save the model
        string modelPath = "PowerConsumptionModel.zip";
        mlContext.Model.Save(model, dataView.Schema, modelPath);

        Console.WriteLine($"Model trained and saved to {modelPath}");
    }

    private static void Predict()
    {
        // Create MLContext
        var mlContext = new MLContext();

        // Load the trained model
        string modelPath = "PowerConsumptionModel.zip";
        mlContext.ComponentCatalog.RegisterAssembly(typeof(ExtractHourAction).Assembly);
        ITransformer trainedModel = mlContext.Model.Load(modelPath, out var modelInputSchema);

        // Create a prediction engine
        var predictionEngine = mlContext.Model.CreatePredictionEngine<PowerConsumptionData, ConsumptionPrediction>(trainedModel);

        // Prepare input data
        var input = new PowerConsumptionData
        {
            Timestamp = new DateTime(2024, 12, 22, 14, 0, 0), // Example timestamp
        };

        // Make a prediction
        var prediction = predictionEngine.Predict(input);

        Console.WriteLine($"Predicted Consumption: {prediction.Consumption}");
    }

    private static void PrintTransformedData(MLContext mlContext, IDataView transformedData)
    {
        var dataEnumerable = mlContext.Data.CreateEnumerable<ExtractedHourData>(
            transformedData, reuseRowObject: true);

        foreach (var item in dataEnumerable)
        {
            Console.WriteLine($"Hour: {item.Hour}");
        }
    }
}

#region Models

// Define the prediction output class
public class ConsumptionPrediction
{
    public float Consumption { get; set; }
}

public class ExtractedHourData
{
    [LoadColumn(0)]
    public float Hour { get; set; }
}

// Define data schema
public class PowerConsumptionData
{
    [LoadColumn(0)]
    public DateTime Timestamp { get; set; }

    [LoadColumn(1)]
    public float ConsumptionKwh { get; set; }
}

[CustomMappingFactoryAttribute("ExtractHourMapping")]
public class ExtractHourAction : CustomMappingFactory<PowerConsumptionData, ExtractedHourData>
{
    public override Action<PowerConsumptionData, ExtractedHourData> GetMapping()
    {
        return (input, output) =>
        {
            output.Hour = input.Timestamp.Hour;
        };
    }
}

#endregion
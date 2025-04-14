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

        Preview(dataView);

        // Update the pipeline to include month and day extraction
        var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Consumption", inputColumnName: "ConsumptionKwh")
           .Append(mlContext.Transforms.CustomMapping(
               new ExtractDatePartsAction().GetMapping(), contractName: "ExtractDatePartsMapping"))
           .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "HourEncoded", inputColumnName: "Hour"))
           .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "DayEncoded", inputColumnName: "Day"))
           .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "MonthEncoded", inputColumnName: "Month"))
           .Append(mlContext.Transforms.Concatenate("Features", "HourEncoded", "DayEncoded", "MonthEncoded"))
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
        mlContext.ComponentCatalog.RegisterAssembly(typeof(ExtractedDatePartsData).Assembly);
        ITransformer trainedModel = mlContext.Model.Load(modelPath, out var modelInputSchema);

        // Create a prediction engine
        var predictionEngine = mlContext.Model.CreatePredictionEngine<PowerConsumptionData, ConsumptionPrediction>(trainedModel);

        var startDate = new DateTime(2024, 12, 21, 15, 0, 0); // Example start date

        for (int i = 0; i < 10; i++)
        {
            // Prepare input data
            var input = new PowerConsumptionData
            {
                Timestamp = startDate.AddHours(i),
            };

            // Make a prediction
            var prediction = predictionEngine.Predict(input);

            Console.WriteLine($"Predicted Consumption ({input.Timestamp}): {prediction.Consumption} KWh");
        }
    }

    //private static void PrintTransformedData(MLContext mlContext, IDataView transformedData)
    //{
    //    var dataEnumerable = mlContext.Data.CreateEnumerable<ExtractedHourData>(
    //        transformedData, reuseRowObject: true);

    //    foreach (var item in dataEnumerable)
    //    {
    //        Console.WriteLine($">>: {item.Hour}");
    //    }
    //}

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

//public class ExtractedHourData
//{
//    [LoadColumn(0)]
//    public float Hour { get; set; }
//}

// Define data schema
public class PowerConsumptionData
{
    [LoadColumn(0)]
    public DateTime Timestamp { get; set; }

    [LoadColumn(1)]
    public float ConsumptionKwh { get; set; }
}

[CustomMappingFactoryAttribute("ExtractDatePartsMapping")]
public class ExtractDatePartsAction : CustomMappingFactory<PowerConsumptionData, ExtractedDatePartsData>
{
    public override Action<PowerConsumptionData, ExtractedDatePartsData> GetMapping()
    {
        return (input, output) =>
        {
            output.Hour = input.Timestamp.Hour;
            output.Day = input.Timestamp.Day;
            output.Month = input.Timestamp.Month;
        };
    }
}

public class ExtractedDatePartsData
{
    public float Hour { get; set; }
    public float Day { get; set; }
    public float Month { get; set; }
}

#endregion
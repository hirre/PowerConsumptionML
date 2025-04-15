using CommandLine;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

class Program
{
    private static Options? _loadedOptions;

    public class Options
    {
        [Option('v', "verbose", Required = false, HelpText = "Set output to verbose messages.")]
        public bool Verbose { get; set; }

        [Option('t', "train", Required = false, HelpText = "Train a model given an input csv file (Timestamp_Month,Timestamp_Day,Timestamp_Hour,Temp_outside,Temp_inside,Consumption_kWh).")]
        public string? Train { get; set; }

        [Option('p', "predict", Required = false, HelpText = "Predict/forecast given a model file that is piped in (file row: \"11,1,3\" means november 1 at hour 3).")]
        public bool Predict { get; set; }

        [Option('o', "output", Required = true, HelpText = "Output model zip file (e.g. \"PowerConsumptionModel.zip\").")]
        public required string OutputModelFile { get; set; }
    }

    static void Main(string[] args)
    {
        _loadedOptions = Parser.Default.ParseArguments<Options>(args).Value;

        if (_loadedOptions == null)
        {
            Console.WriteLine("Invalid options provided.");
            return;
        }

        if (string.IsNullOrEmpty(_loadedOptions.OutputModelFile))
        {
            Console.WriteLine("Output file path is required.");
            return;
        }

        if (!string.IsNullOrEmpty(_loadedOptions.Train))
        {
            Train();
        }

        if (_loadedOptions.Predict)
        {
            if (_loadedOptions.Verbose)
                Console.WriteLine("Predicting data...\n");

            Predict();
        }
    }

    private static void Train()
    {
        if (_loadedOptions == null)
        {
            Console.WriteLine("Invalid options provided.");
            return;
        }

        var mlContext = new MLContext();

        if (_loadedOptions.Verbose)
        {
            // Log experiment trials
            mlContext.Log += (_, e) =>
            {
                if (e.Source.Equals("AutoMLExperiment"))
                {
                    Console.WriteLine(e.RawMessage);
                }
            };
        }

        // Load data
        var data = mlContext.Data.LoadFromTextFile<PowerConsumptionData>(
            _loadedOptions.Train, hasHeader: true, separatorChar: ',');

        if (_loadedOptions.Verbose)
            Preview(data);

        // Split into Train/Test
        var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

        // Create AutoML Regression Experiment
        var experiment = mlContext.Auto()
            .CreateRegressionExperiment(maxExperimentTimeInSeconds: 120);

        // Run AutoML
        var result = experiment.Execute(split.TrainSet, labelColumnName: nameof(PowerConsumptionData.ConsumptionKwh));

        if (_loadedOptions.Verbose)
        {
            Console.WriteLine();
            Console.WriteLine($"Best Model R-squared: {result.BestRun.ValidationMetrics.RSquared}");
            Console.WriteLine($"Absolute loss: {result.BestRun.ValidationMetrics.MeanAbsoluteError}");
            Console.WriteLine($"Squared loss: {result.BestRun.ValidationMetrics.MeanSquaredError}");
            Console.WriteLine($"RMS loss: {result.BestRun.ValidationMetrics.RootMeanSquaredError}");
            Console.WriteLine();
        }

        // Save the best model
        mlContext.Model.Save(result.BestRun.Model, split.TrainSet.Schema, _loadedOptions.OutputModelFile);
    }

    private static void Predict()
    {
        if (_loadedOptions == null)
        {
            Console.WriteLine("Invalid options provided.");
            return;
        }

        // Create MLContext
        MLContext mlContext = new();

        // Load and predict
        var loadedModel = mlContext.Model.Load(_loadedOptions.OutputModelFile, out _);
        var predictionEngine = mlContext.Model.CreatePredictionEngine<PowerConsumptionData, ConsumptionPrediction>(loadedModel);

        string? input;

        while ((input = Console.ReadLine()) != null)
        {
            if (string.IsNullOrEmpty(input))
                break;

            var values = input.Split(',');
            if (values.Length != 3)
            {
                Console.WriteLine($"Invalid input format: {input}");
                continue;
            }
            if (!int.TryParse(values[0], out int month) || !int.TryParse(values[1], out int day) || !int.TryParse(values[2], out int hour))
            {
                Console.WriteLine($"Invalid input values: {input}");
                continue;
            }

            var newPowerConsumption = new PowerConsumptionData { Month = month, Day = day, Hour = hour };
            var prediction = predictionEngine.Predict(newPowerConsumption);

            Console.WriteLine(prediction.Consumption);
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
    public float TempOutside { get; set; }

    [LoadColumn(4)]
    public float TempInside { get; set; }

    [LoadColumn(5)]
    public float ConsumptionKwh { get; set; }
}

#endregion
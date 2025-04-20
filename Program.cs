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

        [Option('t', "train", Required = false, HelpText = "Train a model given an input csv file (Timestamp_Month,Timestamp_Day,Timestamp_Hour,Temp_outside,Temp_inside,Consumption). Usage: -t modeloutputname.zip")]
        public IEnumerable<string>? Train { get; set; }

        [Option('p', "predict", Required = false, HelpText = "Predict/forecast given a model file that is piped in (file row: \"11,1,3\" means november 1 at hour 3).")]
        public bool Predict { get; set; }

        [Option('m', "model", Required = false, HelpText = "Model file (e.g. \"PowerConsumptionModel.zip\").")]
        public string? ModelFile { get; set; }
    }

    static void Main(string[] args)
    {
        _loadedOptions = Parser.Default.ParseArguments<Options>(args).Value;

        if (_loadedOptions == null)
        {
            Console.WriteLine("Invalid options provided.");
            return;
        }

        if (_loadedOptions.Train != null && _loadedOptions.Train.Any() && _loadedOptions.Train.Count() == 2)
        {
            Train();
        }

        if (_loadedOptions.Predict)
        {
            if (string.IsNullOrEmpty(_loadedOptions.ModelFile))
            {
                Console.WriteLine("Output file path is required.");
                return;
            }

            if (_loadedOptions.Verbose)
                Console.WriteLine("Predicting data...\n");

            Predict();
        }

        if (System.Diagnostics.Debugger.IsAttached)
        {
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }
    }

    private static void Train()
    {
        if (_loadedOptions == null || _loadedOptions.Train == null)
        {
            Console.WriteLine("Invalid options provided.");
            return;
        }

        var trainOptions = _loadedOptions.Train.ToList();

        var inputFile = trainOptions[0];
        var outputFile = trainOptions[1];

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
            inputFile, hasHeader: true, separatorChar: ',');

        if (_loadedOptions.Verbose)
            Preview(data);

        try
        {
            var whiteList = mlContext.Data
            .CreateEnumerable<ObjectClass>(data, reuseRowObject: false)
            .Select(x => x.ObjectId)
            .Distinct()
            .Cast<string>();

            var whiteListIdFilePath = outputFile.Replace(".zip", "") + ".dat";

            if (whiteList != null)
            {
                File.WriteAllLines(whiteListIdFilePath, whiteList);

                if (_loadedOptions.Verbose)
                {
                    Console.WriteLine($"ObjectId list saved to {whiteListIdFilePath}");
                }
            }
        }
        catch (Exception e)
        {
            Console.Error.WriteLine(e);
        }

        // Split into Train/Test
        var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

        // Create AutoML Regression Experiment
        var experiment = mlContext.Auto()
            .CreateRegressionExperiment(maxExperimentTimeInSeconds: 120);

        var columnInfo = new ColumnInformation()
        {
            LabelColumnName = nameof(PowerConsumptionData.Consumption)
        };

        // Run AutoML
        var result = experiment.Execute(split.TrainSet, columnInfo);

        if (_loadedOptions.Verbose)
        {
            Console.WriteLine();
            Console.WriteLine($"Best Model: {result.BestRun.TrainerName}");
            Console.WriteLine($"Best Model R-squared: {result.BestRun.ValidationMetrics.RSquared}");
            Console.WriteLine($"Absolute loss: {result.BestRun.ValidationMetrics.MeanAbsoluteError}");
            Console.WriteLine($"Squared loss: {result.BestRun.ValidationMetrics.MeanSquaredError}");
            Console.WriteLine($"RMS loss: {result.BestRun.ValidationMetrics.RootMeanSquaredError}");

            // Inspect the output schema of the best model
            var outputSchema = result.BestRun.Model.GetOutputSchema(split.TrainSet.Schema);

            Console.WriteLine("Model Output Schema:");
            foreach (var column in outputSchema)
            {
                Console.WriteLine($"Column Name: {column.Name}, Column Type: {column.Type}");
            }

            Console.WriteLine();
        }

        // Save the best model
        mlContext.Model.Save(result.BestRun.Model, split.TrainSet.Schema, outputFile);
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
        var loadedModel = mlContext.Model.Load(_loadedOptions.ModelFile, out _);
        var predictionEngine = mlContext.Model.CreatePredictionEngine<PowerConsumptionData, ConsumptionPrediction>(loadedModel);

        var whiteListMap = new HashSet<string>();

        if (!string.IsNullOrEmpty(_loadedOptions.ModelFile))
        {
            var whiteListIdFilePath = _loadedOptions.ModelFile.Replace(".zip", "") + ".dat";

            whiteListMap = [.. File.ReadAllLines(whiteListIdFilePath)];
        }

        string? input;

        while ((input = Console.ReadLine()) != null)
        {
            if (string.IsNullOrEmpty(input))
                break;

            var values = input.Split(',');

            if (values.Length != 4)
            {
                Console.WriteLine($"Invalid input format: {input}");
                continue;
            }

            if (!int.TryParse(values[1], out int month) || !int.TryParse(values[2], out int day) || !int.TryParse(values[3], out int hour))
            {
                Console.WriteLine($"Invalid input values: {input}");
                continue;
            }

            if (whiteListMap.Contains(values[0]))
            {
                var newPowerConsumption = new PowerConsumptionData { ObjectId = values[0], Month = month, Day = day, Hour = hour };
                var prediction = predictionEngine.Predict(newPowerConsumption);

                Console.WriteLine($"{values[0]},{prediction.Consumption}");
            }
        }
    }

    private static void Preview(IDataView dataView)
    {
        var preview = dataView.Preview();

        foreach (var row in preview.RowView)
        {
            Console.WriteLine(string.Join(", ", row.Values));
        }

        foreach (var col in preview.Schema)
        {
            Console.WriteLine($"{col.Name} - {col.Type}");
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

public class ObjectClass
{
    public string? ObjectId { get; set; }
}

// Define data schema
public class PowerConsumptionData
{
    [LoadColumn(0)]
    public required string ObjectId { get; set; }

    [LoadColumn(1)]
    public float Month { get; set; }

    [LoadColumn(2)]
    public float Day { get; set; }

    [LoadColumn(3)]
    public float Hour { get; set; }

    [LoadColumn(4)]
    public float TempOutside { get; set; }

    [LoadColumn(5)]
    public float TempInside { get; set; }

    [LoadColumn(6)]
    public float Consumption { get; set; }
}

#endregion
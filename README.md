# Power Consumption ML

Training data CSV file format:
```
Timestamp_Month,Timestamp_Day,Timestamp_Hour,Temp_outside,Temp_inside,Consumption_kWh
```

Training:
```
PowerConsumption.exe -t simulated_electricity_consumption.csv PowerConsumptionModel.zip -v
```

Forecast:
```
cat input.txt | PowerConsumption.exe -m PowerConsumptionModel.zip -p
```
Where input.txt contains wanted power consumption forecasts (one wanted forecast per input row). For example the row:

4,5,16 

indicates wanted power consumption forecast for april 5th at time 16 (24 hour clock). This will give you a forecast value (kWh) in the corresponding output row.

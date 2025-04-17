# Power Consumption ML

Training data CSV file format (header):
```
Timestamp_Month, Timestamp_Day, Timestamp_Hour, Temp_outside, Temp_inside, Consumption_kWh
```

Example:

```
1,1,0,-8.6,19.7,1.83
1,1,1,-7.6,21.2,1.88
1,1,2,-6.1,20.5,1.86
1,1,3,-5.9,20.4,1.8
1,1,4,-4.6,21.3,1.95
```

Training:
```
PowerConsumption.exe -t simulated_electricity_consumption.csv PowerConsumptionModel.zip -v
```

Forecast:
```
cat input.txt | PowerConsumption.exe -m PowerConsumptionModel.zip -p
```
Where input.txt contains wanted power consumption forecasts (one wanted forecast per input row). For example the rows:

```
4,5,16 
4,5,17
```


indicate wanted power consumption forecast for april 5th 16 o'clock (24 hour clock) and for 17 o'clock. This will, in this example, give you two forecast values (kWh), one in each corresponding output row, for example:
```
1.3232
1.4411
```

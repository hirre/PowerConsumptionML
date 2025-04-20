# Power Consumption ML

How it works:

![image](https://github.com/user-attachments/assets/af0c74c9-d518-470e-83e6-e05c4ffe3681)



Training data CSV file format (header):
```
ObjectId,Timestamp_Month, Timestamp_Day, Timestamp_Hour, Temp_outside, Temp_inside, Consumption_kWh
```

Example:

```
321432345,1,1,0,-8.6,19.7,1.83
321432345,1,1,1,-7.6,21.2,1.88
321432345,1,1,2,-6.1,20.5,1.86
321432345,1,1,3,-5.9,20.4,1.8
321432345,1,1,4,-4.6,21.3,1.95
```

Training:
```
PowerConsumptionML.exe -t simulated_electricity_consumption.csv PowerConsumptionModel.zip -v
```

Forecast:
```
cat input.txt | PowerConsumptionML.exe -m PowerConsumptionModel.zip -p
```
Where input.txt contains wanted power consumption forecasts (one wanted forecast per input row). For example the rows:

```
321432345,4,5,16 
321432345,4,5,17
```


indicate wanted power consumption forecast for april 5th 16 o'clock (24 hour clock) and for 17 o'clock. This will, in this example, give you two forecast values (kWh), one in each corresponding output row, for example:
```
321432345,1.3232
321432345,1.4411
```

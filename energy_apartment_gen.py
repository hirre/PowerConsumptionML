import pandas as pd
import numpy as np
import datetime

# --- Parameters ---
num_points = 10000
start_date = datetime.datetime(2023, 11, 1) # Startdatum (kan vara godtyckligt)
output_filename = 'simulated_electricity_consumption.csv'

# --- Simulation Logic ---

# 1. Create Timestamp Index
timestamps = pd.date_range(start=start_date, periods=num_points, freq='h')

# 2. Baseline Consumption (fridge, standby, etc.)
base_consumption = np.random.uniform(0.05, 0.15, num_points) # kWh per hour

# 3. Diurnal Pattern (Daily cycle)
# Higher consumption morning (6-9) and evening (17-22)
hours = timestamps.hour
diurnal_pattern = np.zeros(num_points)
# Morning peak
diurnal_pattern += np.sin(np.pi * (hours - 6) / 3) * np.random.uniform(0.1, 0.3, num_points) * ((hours >= 6) & (hours < 9))
# Evening peak (stronger)
diurnal_pattern += np.sin(np.pi * (hours - 17) / 5) * np.random.uniform(0.3, 0.8, num_points) * ((hours >= 17) & (hours < 22))
# Lower daytime activity (slight increase)
diurnal_pattern += np.random.uniform(0.05, 0.15, num_points) * ((hours >= 9) & (hours < 17))

# Ensure non-negative contribution from sine waves
diurnal_pattern = np.maximum(0, diurnal_pattern)

# 4. Weekly Pattern
# Slightly higher/different pattern on weekends (Sat=5, Sun=6)
dayofweek = timestamps.dayofweek
weekend_factor = np.where((dayofweek >= 5), np.random.uniform(1.1, 1.4, num_points), 1.0)
# Make daytime higher on weekends, shift evening peak slightly
weekend_day_boost = np.random.uniform(0.1, 0.25, num_points) * ((hours >= 9) & (hours < 17)) * (dayofweek >= 5)
diurnal_pattern = (diurnal_pattern + weekend_day_boost) * weekend_factor


# 5. Winter Effect / Heating Component (Stochastic)
# Assume occasional use of electric heater or similar during colder hours (more likely morning/evening)
heating_probability = 0.15 + 0.1 * np.sin(np.pi * (hours - 16) / 12) # Higher prob in evening/night
heating_boost = np.random.uniform(0.5, 2.0, num_points) # kWh boost when heating is on
heating_component = (np.random.rand(num_points) < heating_probability) * heating_boost
heating_component = np.maximum(0, heating_component) # Ensure non-negative

# 6. Random Noise
noise = np.random.normal(0, 0.05, num_points) # Gaussian noise

# 7. Combine Components
total_consumption = base_consumption + diurnal_pattern + heating_component + noise
total_consumption = np.maximum(0.02, total_consumption) # Ensure a minimum plausible consumption

# 8. Create DataFrame
df = pd.DataFrame({
    'Timestamp_Month': timestamps.month,
    'Timestamp_Day': timestamps.day,
    'Timestamp_Hour': timestamps.hour,
    'Consumption_kWh': total_consumption
})

# Round consumption to reasonable precision
df['Consumption_kWh'] = df['Consumption_kWh'].round(3)

# 9. Save to CSV
df.to_csv(output_filename, index=False, date_format='%Y-%m-%d %H:%M:%S')

print(f"Successfully generated '{output_filename}' with {len(df)} data points.")
print("\nFirst 5 rows:")
print(df.head().to_string())
print("\nLast 5 rows:")
print(df.tail().to_string())
print("\nBasic statistics:")
print(df['Consumption_kWh'].describe())
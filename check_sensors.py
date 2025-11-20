import pandas as pd
df = pd.read_csv('data/R1_vir_telemetry_data.csv', usecols=['telemetry_name'])
print("SENSORS:", df['telemetry_name'].unique())

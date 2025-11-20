import pandas as pd
try:
    df = pd.read_csv('data/R1_vir_telemetry_data.csv', nrows=5)
    print("COLUMNS:", df.columns.tolist())
    print("SAMPLE ROW:\n", df.iloc[0])
except Exception as e:
    print(e)

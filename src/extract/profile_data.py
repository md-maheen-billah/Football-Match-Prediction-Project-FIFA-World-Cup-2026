import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")

for file in RAW_DIR.glob("*.csv"):

    print("\n" + "="*60)
    print(file.name)
    print("="*60)

    df = pd.read_csv(file)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nDtypes:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())
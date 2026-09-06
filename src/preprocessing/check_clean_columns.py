from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/interim/CIC-IDS2018")

files = sorted(DATA_DIR.glob("*.csv"))

if not files:
    print("No cleaned CSV files found.")
    exit()

file_path = files[0]

print(f"Checking: {file_path.name}\n")

df = pd.read_csv(
    file_path,
    nrows=5
)

print("Number of columns:", len(df.columns))
print("\nColumns:\n")

for i, column in enumerate(df.columns, start=1):
    print(f"{i:02d}. {column}")
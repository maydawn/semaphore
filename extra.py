import pandas as pd

# Load the CSV
df = pd.read_csv("annotations_gs54.csv", header = None)

# Drop a column (e.g., 'personID')
df = df.drop(columns=[1])

# Save the updated CSV
df.to_csv("annotations_gs54_cleaned.csv", header = False, index=False)
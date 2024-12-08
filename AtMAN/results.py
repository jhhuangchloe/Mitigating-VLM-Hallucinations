import json
import numpy as np


def interquartile_mean(data):
    # Convert to numpy array (in case it's a list or other sequence)
    data = np.array(data)
    
    # Calculate Q1 and Q3
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    
    # Filter data within the interquartile range
    in_iqr = data[(data >= Q1) & (data <= Q3)]
    
    # Calculate and return the mean of the filtered data
    return np.mean(in_iqr)


with open("results/atman.json", "r") as f:
    results = json.load(f)

precisions = []
recalls = []

for r in results:
    precisions.append(r["precision"])
    recalls.append(r["recall"])

print(f"Average precision: {np.mean(precisions)*100:.2f}")
print(f"Average Recall: {np.mean(recalls)*100:.2f}")
print(f"Average IQ precision: {interquartile_mean(precisions)*100:.2f}")
print(f"Average IQ Recall: {interquartile_mean(recalls)*100:.2f}")


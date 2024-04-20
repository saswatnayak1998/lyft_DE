import pandas as pd
import numpy as np

# Parameters for the hypergeometric distribution
# M = 500 (total number of objects)
# n = 100 (total number of type I objects)
# N = 50 (number of draws)
M = 500
n = 100
N = 50

# Generating random data using hypergeometric distribution
# Generating 1000 samples
hypergeom_samples = np.random.hypergeometric(n, M-n, N, 1000)

# Creating a DataFrame
df_hypergeom = pd.DataFrame(hypergeom_samples, columns=['Hypergeometric'])

# Save the DataFrame to a CSV file
csv_path = 'Hypergeometric_Distribution_Samples.dat'
df_hypergeom.to_csv(csv_path, index=False)


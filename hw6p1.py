
import numpy as np
import pandas as pd

# Load the dataset using a raw string for the path to avoid escape character issues
data = pd.read_csv(r"C:\Users\david\OneDrive\Desktop\Misc\School\Code\School\CSC395\hw6\crime_data.csv")

# Calculate the crime rates per capita
crime_columns = ['MURDER', 'ROBBERY', 'AGASSLT', 'BURGLRY', 'LARCENY', 'MVTHEFT', 'ARSON']
data.replace(0, np.nan, inplace=True)  # Replace 0 with NaN to avoid issues with log(0)
data.dropna(subset=crime_columns + ['population'], inplace=True)  # Remove any rows with NaN values in crime columns or population

for column in crime_columns:
    data[f'{column}_rate'] = (data[column] / data['population']) * 100000

# Logarithmic transformation, adding 1 to avoid log(0)
data['log_population'] = np.log(data['population'] + 1)
for column in crime_columns:
    data[f'log_{column}_rate'] = np.log(data[f'{column}_rate'] + 1)

# Prepare results dictionary
results = {}

# Perform linear least squares for each crime type
for column in crime_columns:
    # Constructing matrix J and vector y
    J = np.vstack([data['log_population'], np.ones(len(data))]).T
    y = data[f'log_{column}_rate'].values
    
    # Solving the normal equation J^T J ξ = J^T y
    JTJ = np.dot(J.T, J)
    JTy = np.dot(J.T, y)
    xi = np.linalg.solve(JTJ, JTy)
    
    # Store results (gamma, c)
    results[column] = {'gamma': xi[0], 'c': xi[1]}

# Printing results using standard characters
for crime, params in results.items():
    print(f"{crime}: gamma = {params['gamma']:.3f}, c = {params['c']:.3f}")

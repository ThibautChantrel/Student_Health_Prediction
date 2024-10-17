# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


fileName = "Modified_sleep_health_and_lifestyle_dataset"+".csv"
FullPath = "./Modified_csv/" + fileName
data = pd.read_csv(FullPath, sep=';')  # Ajuster le séparateur si nécessaire (tabulation '\t' ou virgule ',')

# Check the first few rows of the dataset to understand its structure
print(data.head())

# Compute the correlation matrix
correlation_matrix = data.corr()

# Set up the matplotlib figure for the heatmap
plt.figure(figsize=(12, 8))

# Draw the heatmap using seaborn
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

# Show the heatmap
plt.title('Correlation Heatmap')
plt.show()

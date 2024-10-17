import pandas as pd
import numpy as np

# Load the dataset (replace 'your_dataset.csv' with the actual dataset file path)
df = pd.read_csv('./Modified_csv/Sleep_health_and_lifestyle_dataset_V2.csv', sep=';')  

print("Display the dimension of the dataset :")
print(df.shape)

print("\nDisplay the columns of the dataset :")
print('Colonne : '+ df.columns);

missing_values = df.isnull().sum()
print("\nMissing values for each column :")
print(missing_values)


# Function to detect outliers using IQR (Interquartile Range) method
def detect_outliers(df):
    outliers = {}
    
    # Numerical columns for outlier detection
    numerical_columns = [
        'Age', 'Sleep Duration', 'Quality of Sleep', 
        'Physical Activity Level', 'Stress Level', 
        'Heart Rate', 'Daily Steps'
    ]
    
    for col in numerical_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    return outliers

# Function to calculate summary statistics
def summary_stats(df):
    summary = {}
    
    summary['Average Age'] = df['Age'].mean()
    summary['Average Sleep Duration'] = df['Sleep Duration'].mean()
    summary['Average Quality of Sleep'] = df['Quality of Sleep'].mean()
    summary['Average Physical Activity'] = df['Physical Activity Level'].mean()
    summary['Average Stress Level'] = df['Stress Level'].mean()
    summary['Average Heart Rate'] = df['Heart Rate'].mean()
    summary['Average Daily Steps'] = df['Daily Steps'].mean()

    summary['Gender Distribution'] = df['Gender'].value_counts()
    summary['BMI Category Distribution'] = df['BMI Category'].value_counts()
    summary['Sleep Disorder Distribution'] = df['Sleep Disorder'].value_counts()

    return summary

# Detect outliers
outliers = detect_outliers(df)

# Generate summary
summary = summary_stats(df)


print("Outliers Detected:")
for col, outliers_df in outliers.items():
    print(f"{col}:\n{outliers_df}\n")

print("\nSummary Statistics:")
for key, value in summary.items():
    print(f"{key}: {value}\n")

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
df = pd.read_csv('./Modified_csv/Sleep_health_and_lifestyle_dataset_V2.csv', sep=';')  

# Gender: Male -> 0, Female -> 1
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

df['BMI Category'] = df['BMI Category'].map({'underweight': 0, 'Normal': 1, 'Normal Weight' : 1, 'Overweight': 2, 'Obese': 3})

df['Sleep Disorder'] = df['Sleep Disorder'].map({'None': 0, 'Sleep Apnea': 1, 'Insomnia': 2})
df['Sleep Disorder'] = df['Sleep Disorder'].fillna(0)  # Assuming 'None' as default for missing or unrecognized values


# Split Blood Pressure into Systolic and Diastolic
df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True)
df['Systolic'] = pd.to_numeric(df['Systolic'])
df['Diastolic'] = pd.to_numeric(df['Diastolic'])

# Drop the original 'Blood pressure' column as it's no longer needed
df = df.drop(columns=['Blood Pressure'])

# One-Hot Encoding for Occupation
#onehot_encoder = OneHotEncoder(sparse=False, drop='first')  # drop='first' to avoid multicollinearity
onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
occupation_encoded = onehot_encoder.fit_transform(df[['Occupation']])

# Get the names of the new columns from the encoder
occupation_columns = onehot_encoder.get_feature_names_out(['Occupation'])

# Convert the encoded columns into a DataFrame and merge it back with the original DataFrame
occupation_df = pd.DataFrame(occupation_encoded, columns=occupation_columns)
df = pd.concat([df, occupation_df], axis=1)

# Drop the original 'Occupation' column as it's no longer needed
df = df.drop(columns=['Occupation'])

# Save the modified DataFrame to a new CSV file
df.to_csv("./Modified_csv/Modified_sleep_health_and_lifestyle_dataset.csv", index=False)

print("Data transformations completed and saved to Modified_sleep_health_and_lifestyle_dataset.csv")

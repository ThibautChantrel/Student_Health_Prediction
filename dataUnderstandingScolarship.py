import pandas as pd

df = pd.read_csv('./Modified_csv/Student_performance_data_V2.csv', sep=';')  # Ajuster le séparateur si nécessaire (tabulation '\t' ou virgule ',')

print("Display the dimension of the dataset :")
print(df.shape)

print("\nDisplay the columns of the dataset :")
print('Colonne : '+ df.columns);

missing_values = df.isnull().sum()



def detect_outliers(df):
    outliers = {}
    # Age expected between 15 and 18 yo
    outliers['Age'] = df[(df['Age'] < 15) | (df['Age'] > 18)]
    # GPA expected between 2.0 and 4.0
    outliers['GPA'] = df[(df['GPA'] < 2.0) | (df['GPA'] > 4.0)]
    # ethnicity : valeus expected between 0 and 3
    outliers['Ethnicity'] = df[(df['Ethnicity'] < 0) | (df['Ethnicity'] > 3)]
    # ParentalEducation : valeus expected between 0 and 4
    outliers['ParentalEducation'] = df[(df['ParentalEducation'] < 0) | (df['ParentalEducation'] > 4)]
    return outliers

outliers = detect_outliers(df)

# Info summary
summary = {}

summary['Average Age'] = df['Age'].mean()

summary['Average GPA'] = df['GPA'].mean()

summary['Ethnicity Distribution'] = df['Ethnicity'].value_counts()

summary['Gender Distribution'] = df['Gender'].value_counts()

summary['Average Study Time Weekly'] = df['StudyTimeWeekly'].mean()

summary['Average Absences'] = df['Absences'].mean()

summary['Extracurricular Participation'] = df['Extracurricular'].value_counts()

summary['Sports Participation'] = df['Sports'].value_counts()

summary['Music Participation'] = df['Music'].value_counts()

summary['Volunteering Participation'] = df['Volunteering'].value_counts()

print("Missing values for each column :")
print(missing_values)

print("\n Outliers detected :")
for key, value in outliers.items():
    if not value.empty:
        print(f"\n{key} :")
        print(value)

print("\n Summary :")
for key, value in summary.items():
    print(f"{key} : {value}")
    
    
    
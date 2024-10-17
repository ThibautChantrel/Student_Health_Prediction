import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split

fileName = "Student_performance_data_V2"+".csv"
FullPath = "./Modified_csv/" + fileName
df = pd.read_csv(FullPath, sep=';')  # Ajuster le séparateur si nécessaire (tabulation '\t' ou virgule ',')

# Define the feature columns and target variable
X = df.drop(columns=['GPA', 'GradeClass', 'StudentID']) 
y = df['GPA']  # The target column

numberOfcolumn = len(X.columns)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)

bestR2 = 0
bestFeature = 0

for i in range(numberOfcolumn):
    print("Number of Features Selected:", i+1)
    selector = RFE(model, n_features_to_select=i+1, step=1)  # Select the top n features
    selector = selector.fit(X_train, y_train)
    # Display the selected features
    selected_features = X_train.columns[selector.support_]
    print("Selected Features:", selected_features)

    # Fit the model using the selected features
    model.fit(X_train[selected_features], y_train)

    # Evaluate the model (for example, using R^2 score)
    score = model.score(X_test[selected_features], y_test)
    print(f"R^2 Score of the model: {score}")
    if score > bestR2:
        bestR2 = score
        bestFeature = i+1
    print("---------------------------------------------------")
print(f"Best R^2 Score: {bestR2} with {bestFeature} features")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the dataset
data = pd.read_csv('data/processed_data.csv')

# Setting up the visualization style
sns.set(style="whitegrid")

# 1. Distribution of the Target Variable
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Target', palette='viridis')
plt.title('Distribution of the Target Variable')
plt.xlabel('Target')
plt.ylabel('Count')
plt.savefig('images/distribution_target.png')
plt.show()

# 2. Correlation Heatmap of the Features
plt.figure(figsize=(14, 12))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of the Features')
plt.savefig('images/correlation_heatmap.png')
plt.show()

# 3. Distribution of Grades by Marital Status
plt.figure(figsize=(12, 6))
sns.boxplot(data=data, x='Marital status', y='Previous qualification (grade)', palette='muted')
plt.title('Distribution of Grades by Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Previous Qualification Grade')
plt.savefig('images/grades_marital_status.png')
plt.show()

# 4. Unemployment Rate vs. Target Outcome
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Target', y='Unemployment rate', palette='coolwarm')
plt.title('Unemployment Rate vs. Target Outcome')
plt.xlabel('Target')
plt.ylabel('Unemployment Rate')
plt.savefig('images/unemployment_target.png')
plt.show()

# Selecting features and target variable
features = data.drop(columns=['Target'])
target = data['Target']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Training a RandomForest classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Generating the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(10, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix of the Classification Model')
plt.savefig('images/confusion_matrix.png')
plt.show()

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.feature_selection import SelectFromModel

df = pd.read_csv("mushrooms.csv")

print(df.head())

print(f"Dataset shape: {df.shape}")

print(df.info())

print(df['class'].value_counts())

print(df.isnull().sum())

for col in df.columns:
    print(f"{col}: {df[col].unique()}")

print(df['stalk-root'].value_counts())
print(df['cap-shape'].value_counts())
print(df['cap-surface'].value_counts())
print(df['cap-color'].value_counts())
print(df['bruises'].value_counts())
print(df['odor'].value_counts())
print(df['gill-attachment'].value_counts())
print(df['gill-spacing'].value_counts())
print(df['gill-size'].value_counts())
print(df['gill-color'].value_counts())
print(df['stalk-shape'].value_counts())
print(df['habitat'].value_counts())
print(df['population'].value_counts())
print(df['spore-print-color'].value_counts())
print(df['ring-type'].value_counts())
print(df['ring-number'].value_counts())
print(df['veil-color'].value_counts())
print(df['veil-type'].value_counts())
print(df['stalk-color-below-ring'].value_counts())
print(df['stalk-color-above-ring'].value_counts())
print(df['stalk-surface-below-ring'].value_counts())
print(df['stalk-surface-above-ring'].value_counts())

df['stalk-root'] = df['stalk-root'].replace('?', 'b')

print(df['stalk-root'].value_counts())

le = LabelEncoder()
df_encoded = df.copy()
for col in df.columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])

print(df.head())

X = df_encoded.drop('class', axis=1)
y = df_encoded['class']

# Feature selection using DecisionTreeClassifier
feature_selector = DecisionTreeClassifier(random_state=42)
feature_selector.fit(X, y)
selector = SelectFromModel(feature_selector, threshold="mean", prefit=True)
X = selector.transform(X)
selected_features = X.shape[1]
print(f"Number of selected features: {selected_features}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report: ", classification_report(y_test, y_pred))


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# 1. Leitura dos dados
df = pd.read_csv("data/diamonds.csv")

# Drop the unnamed index column (first column)
if df.columns[0] == 'Unnamed: 0' or df.columns[0] == '':
    df = df.drop(df.columns[0], axis=1)

# Separate features and target
X = df.drop("cut", axis=1)
y = df["cut"]

# Encode categorical features (color and clarity)
label_encoders = {}
for column in ['color', 'clarity']:
    if column in X.columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le

# 2. Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Treinamento do modelo (multi-class classification)
model = LogisticRegression(
    max_iter=1000,
    multi_class='multinomial',
    solver='lbfgs',
    random_state=42
)
model.fit(X_train, y_train)

# 4. Avaliação
y_pred = model.predict(X_test)

# Get classification report
report = classification_report(y_test, y_pred)

print("Classification Report:")
print(report)

# 5. Salvamento do relatório
with open("report.txt", "w") as f:
    f.write("Diamond Cut Classification Report\n")
    f.write("="*50 + "\n\n")
    f.write(f"Model: Logistic Regression (Multinomial)\n")
    f.write(f"Dataset: {len(df)} samples\n")
    f.write(f"Features: {', '.join(X.columns)}\n")
    f.write(f"Target: cut (5 classes)\n\n")
    f.write("Classification Report:\n")
    f.write("-"*50 + "\n")
    f.write(report)
    
print("\nReport saved to report.txt")
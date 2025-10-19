import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Leitura dos dados
df = pd.read_csv("data/diamonds.csv")

X = df.drop("target", axis=1)
y = df["target"]

# 2. Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Treinamento do modelo
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Avaliação
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)

# 5. Salvamento do relatório
with open("report.txt", "w") as f:
    f.write(report)
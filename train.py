import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import os

# 1. İşlenmiş verileri yükle
X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

print("Model eğitiliyor...")

# 2. XGBoost Modelini Tanımla
# scale_pos_weight: Dengesiz veriyle başa çıkmak için (Normal / Fraud oranı)
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=500, # Fraud sınıfına daha fazla ağırlık ver
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# 3. Eğitimi Başlat
model.fit(X_train, y_train)

# 4. Tahmin ve Değerlendirme
y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred)

print("\n--- Model Performansı ---")
print(f"F1 Score: {f1:.4f}")
print("\nKarmaşıklık Matrisi (Confusion Matrix):")
print(confusion_matrix(y_test, y_pred))
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

# 5. Modeli Kaydet
os.makedirs('models', exist_ok=True)
import joblib
joblib.dump(model, 'models/fraud_model.pkl')
print("\nModel 'models/fraud_model.pkl' olarak kaydedildi.")

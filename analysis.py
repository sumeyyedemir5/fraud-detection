import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Veriyi oku
df = pd.read_csv('data/raw/creditcard.csv')

# 1. Sınıf Dağılımına Bakalım
print("Sınıf Dağılımı:")
print(df['Class'].value_counts(normalize=True) * 100)

# 2. Görselleştirme
plt.figure(figsize=(8, 5))
sns.countplot(x='Class', data=df)
plt.title('Sınıf Dağılımı (0: Normal, 1: Fraud)')
plt.yscale('log') # Veri farkı çok olduğu için logaritmik ölçek daha iyi gösterir
plt.show()

# 3. İşlem Tutarlarını İnceleyelim
print("\nDolandırıcılık İşlemleri Tutar Özeti:")
print(df[df['Class'] == 1]['Amount'].describe())

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Wczytaj zbiór penguins.csv
df = pd.read_csv('penguins.csv', sep=';', decimal=',', encoding='utf-8')

print("Kształt zbioru:", df.shape)
print("\nLiczba brakujących wartości:\n", df.isnull().sum())

# --- Identyfikacja i korekta wartości odstających ---
numeric_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)][col]
    if not outliers.empty:
        print(f"\nWartości odstające w kolumnie '{col}':")
        print(outliers)
    df[col] = df[col].clip(lower=lower, upper=upper)

# --- Imputacja wartości brakujących ---
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Kolumna 'sex' – imputacja modą
df['sex'] = df['sex'].fillna(df['sex'].mode()[0])

print("\nLiczba brakujących wartości po imputacji:\n", df.isnull().sum())

# --- One-hot encoding atrybutów nominalnych (sex) ---
# Kolumna 'class' jest atrybutem decyzyjnym – zachowujemy ją jako zmienną kategoryczną
df = pd.get_dummies(df, columns=['sex'], prefix='sex')

# Zmień kolumny boolowskie (True/False) na int (0/1)
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

print("\nKolumny po one-hot encoding:", df.columns.tolist())

# --- Skalowanie wartości za pomocą StandardScaler ---
# Skalujemy wszystkie kolumny numeryczne oprócz atrybutu decyzyjnego 'class'
cols_to_scale = [c for c in df.columns if c != 'class']

scaler = StandardScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

print("\nPierwsze wiersze po skalowaniu:")
print(df.head())

# --- Zapisz przygotowany zbiór do pliku my-penguins.csv ---
df.to_csv('my-penguins.csv', sep=';', decimal=',', encoding='utf-8', index=False)
print("\nZapisano plik my-penguins.csv")
print("Czy udało się przygotować zbiór? TAK")

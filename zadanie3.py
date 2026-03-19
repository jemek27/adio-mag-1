import pandas as pd
from sklearn.metrics import adjusted_rand_score

# Wczytaj dane z grupowaniem
df = pd.read_csv('my-penguins-with-groups.csv', sep=';', decimal=',', encoding='utf-8')

# --- Macierz zgodności: wiersze = class, kolumny = group ---
confusion = pd.crosstab(df['class'], df['group'],
                        rownames=['class'], colnames=['group'])

print("Macierz zgodności (class vs group):")
print(confusion)

# --- Ocena jakości grupowania ---
ari = adjusted_rand_score(df['class'], df['group'])
print(f"\nadjusted_rand_score: {ari:.4f}")

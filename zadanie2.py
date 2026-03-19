import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Wczytaj przygotowane dane
df = pd.read_csv('my-penguins.csv', sep=';', decimal=',', encoding='utf-8')

# Oddziel atrybut decyzyjny 'class'
X = df.drop(columns=['class'])

# --- Metoda łokciowa (Elbow Method) ---
inertias = []
k_range = range(1, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(k_range), inertias, marker='o')
plt.xlabel('n_clusters')
plt.ylabel('inertia_')
plt.title('Metoda łokciowa (Elbow Method)')
plt.xticks(list(k_range))
plt.tight_layout()
plt.savefig('elbow.png', dpi=150)
plt.show()
print("Wykres zapisany jako elbow.png")

# --- Wybór optymalnej liczby grup ---
# Na podstawie wykresu łokciowego wybieramy k=3 (trzy gatunki pingwinów)
best_k = 3
print(f"\nWybrana liczba grup: {best_k}")

km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['group'] = km_best.fit_predict(X)

# --- Wizualizacja wyników za pomocą PCA (2D) ---
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['group'], cmap='tab10', alpha=0.7)
plt.colorbar(scatter, label='Grupa (KMeans)')
plt.xlabel('Pierwsza główna składowa (PC1)')
plt.ylabel('Druga główna składowa (PC2)')
plt.title(f'Grupowanie KMeans (k={best_k}) – wizualizacja PCA')
plt.tight_layout()
plt.savefig('kmeans_pca.png', dpi=150)
plt.show()
print("Wykres zapisany jako kmeans_pca.png")

# --- Zapisz dane z kolumną 'group' ---
df.to_csv('my-penguins-with-groups.csv', sep=';', decimal=',', encoding='utf-8', index=False)
print("\nZapisano plik my-penguins-with-groups.csv")
print("Czy udało się przeprowadzić grupowanie i zwizualizować wyniki? TAK")

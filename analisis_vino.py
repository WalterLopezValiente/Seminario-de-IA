
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Usamos de preferencia el estilo de seaborn para las gráficas
plt.style.use('seaborn')

# Primero cargamos los datos
print("Cargando el dataset de vinos...")
df = pd.read_csv('datos_vino/winequality-red.csv', delimiter=';')

# Miramos rápido a los datos
print("\nAsí se ven los primeros registros:")
print(df.head())

# Separamos features y target (la calidad es lo que quiero predecir)
X = df.drop('quality', axis=1)
y = df['quality']

# Dividimos en train y test (80-20 me parece una buena proporción)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # uso 42 porque es mi número de la suerte
)

# Normalizamos los datos porque están en escalas muy diferentes
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Primero probamos con un árbol de decisión simple
print("\nEntrenando el árbol de decisión...")
dt_classifier = DecisionTreeClassifier(
    max_depth=5,  # después de probar varios valores, 5 me dio buenos resultados
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

dt_classifier.fit(X_train_scaled, y_train)
dt_predictions = dt_classifier.predict(X_test_scaled)

print("\nResultados del árbol:")
print(f"Precisión: {accuracy_score(y_test, dt_predictions):.4f}")
print("\nReporte detallado:")
print(classification_report(y_test, dt_predictions))

# Ahora vamos con el random forest que debería ser mejor
print("\nProbando con Random Forest...")
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_classifier.fit(X_train_scaled, y_train)
rf_predictions = rf_classifier.predict(X_test_scaled)

print("\nResultados del Random Forest básico:")
print(f"Precisión: {accuracy_score(y_test, rf_predictions):.4f}")

# Vamos a buscar los mejores parámetros
print("\nBuscando los mejores parámetros (esto puede tardar un rato)...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    n_jobs=-1,  # uso todos los cores disponibles
    scoring='accuracy'
)

grid_search.fit(X_train_scaled, y_train)

print("\n¡Encontré los mejores parámetros!")
print(grid_search.best_params_)

# Pruebo el modelo optimizado
best_rf = grid_search.best_estimator_
best_predictions = best_rf.predict(X_test_scaled)

print("\nResultados del Random Forest mejorado:")
print(f"Precisión: {accuracy_score(y_test, best_predictions):.4f}")
print("\nReporte completo:")
print(classification_report(y_test, best_predictions))

# Guardamos algunas gráficas para el análisis
print("\nGenerando visualizaciones...")

# Importancia de las características
feature_importance = pd.DataFrame({
    'caracteristica': X.columns,
    'importancia': best_rf.feature_importances_
}).sort_values('importancia', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importancia', y='caracteristica')
plt.title('¿Qué características son más importantes?')
plt.tight_layout()
plt.savefig('resultados/importancia_features.png')
plt.close()

# Distribución de la calidad
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='quality', bins=len(df['quality'].unique()))
plt.title('Distribución de la calidad del vino')
plt.savefig('resultados/distribucion_calidad.png')
plt.close()

# Matriz de correlación (me ayuda a ver relaciones entre variables)
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de correlación')
plt.tight_layout()
plt.savefig('resultados/correlaciones.png')
plt.close()

# Guardamos una comparación de los modelos
results_df = pd.DataFrame({
    'Real': y_test,
    'Árbol': dt_predictions,
    'Random Forest': best_predictions
})

plt.figure(figsize=(10, 6))
sns.boxplot(data=results_df)
plt.title('Comparación de modelos')
plt.savefig('resultados/comparacion_modelos.png')
plt.close()

print("\n¡Listo! He guardado todas las gráficas en la carpeta 'resultados'")
print("Nota personal: el Random Forest funcionó mejor, como esperaba :)")

# Algunas observaciones que encontré
print("\nObservaciones importantes:")
print("1. El contenido de alcohol parece ser el factor más importante")
print("2. La mayoría de los vinos tienen una calidad media (5-6)")
print("3. El Random Forest mejoró bastante después de optimizar los parámetros")
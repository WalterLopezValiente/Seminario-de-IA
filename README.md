# Predicción de Calidad del Vino - Seminario Práctico
**Autores:** WalterLopezValiente, Anais Martinez Morales

## Sobre este proyecto
Este es nuestro trabajo para el seminario práctico sobre predicción de calidad del vino. Nos centramos en usar árboles de decisión y random forest para predecir la calidad del vino basándome en sus características químicas.

## ¿Qué vamos a hacer?
Trabajaremos con un dataset de vinos tintos, analizando sus propiedades químicas para predecir su calidad. Es interesante ver cómo diferentes características pueden influir en la calidad final del vino.

## Nuestra implementación

### Procesamiento inicial de datos
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargamos el dataset
df = pd.read_csv('data/winequality-red.csv', delimiter=';')

# Separamos features y target
X = df.drop('quality', axis=1)
y = df['quality']

# Dividimos los datos (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Estandarizamos las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Nuestro árbol de decisión
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Configuramos estos parámetros después de algunas pruebas
dt_classifier = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
dt_classifier.fit(X_train_scaled, y_train)

dt_predictions = dt_classifier.predict(X_test_scaled)

# Ver qué tal lo hicimos
print("Resultados del árbol:")
print(f"Precisión: {accuracy_score(y_test, dt_predictions):.4f}")
print("\nReporte detallado:")
print(classification_report(y_test, dt_predictions))
```

### Random Forest
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Primero probamos un modelo básico
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_classifier.fit(X_train_scaled, y_train)

# Después buscamos los mejores parámetros
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
    n_jobs=-1,
    scoring='accuracy'
)

grid_search.fit(X_train_scaled, y_train)
```

## Nuestros resultados y observaciones
1. El random forest funcionó mejor que el árbol de decisión simple
2. Las características más importantes fueron:
   - alcohol
   - volatile acidity
   - sulphates
3. La distribución de calidad está desbalanceada (más vinos de calidad media)

## Archivos del proyecto
```
├── datos_vino/
│   └── winequality-red.csv
├── codigo/
│   └── analisis_vino.py
└── resultados/
    ├── arbol_decision.png
    ├── importancia_features.png
    └── comparacion_modelos.png
```

## Cómo ejecutar nuestro código
1. Clona el repo
2. Instala las dependencias:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```
3. Ejecuta el script principal:
```bash
python codigo/analisis_vino.py
```

## Notas personales
- Nos tomó tiempo ajustar los hiperparámetros del random forest
- Tuvimis que normalizar los datos porque había valores muy dispersos
- La visualización del árbol ayuda mucho a entender el modelo

## Próximas mejoras
- [ ] Probar con PCA para reducir dimensionalidad
- [ ] Implementar validación cruzada
- [ ] Añadir más visualizaciones

---
*Entrega para el curso de Machine Learning - Lunes 21 de abril de 2025, 12:00 M*

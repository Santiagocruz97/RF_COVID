import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Para machine learning más adelante:
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree
from sklearn.metrics import roc_curve, auc


# Ruta del archivo CSV
ruta = 'C:/Git_SCA/RF_COVID/datos_covid.csv'  # <-- Cambia a tu ruta real

# Leer el CSV
df = pd.read_csv(ruta)

# Verificar los datos
print(df.head())
print(df.shape)
print(df.columns)

# Variables de entrada
X = df[['Edad', 'Sexo', 'Ciudad', 'Municipio']]

# Variable objetivo
y = df['Estado']

print(X.head())
print(y.head())

# Crear objeto codificador
le = LabelEncoder()

X.loc[:, 'Sexo'] = le.fit_transform(X['Sexo'])
# Si también estás codificando 'Ciudad' o 'Municipio':
X.loc[:, 'Ciudad'] = le.fit_transform(X['Ciudad'])
X.loc[:, 'Municipio'] = le.fit_transform(X['Municipio'])

# Separar datos: 70% entrenamiento, 30% prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Crear el modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')


# Entrenar el modelo
modelo.fit(X_train, y_train)

# Hacer predicciones sobre el conjunto de prueba
y_pred = modelo.predict(X_test)

# Calcular la precisión
print("Precisión del modelo:", accuracy_score(y_test, y_pred))

# Matriz de confusión
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

# Reporte de clasificación completo
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))


# Obtener las importancias
importances = modelo.feature_importances_

# Nombres de las variables
features = X.columns

# Ordenar importancias
indices = np.argsort(importances)[::-1]

# Imprimir
print("Importancia de cada variable:")
for i in range(len(features)):
    print(f"{features[indices[i]]}: {importances[indices[i]]:.4f}")

# Gráfico
plt.figure(figsize=(8,6))
plt.title('Importancia de las Variables')
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=45)
plt.ylabel('Importancia')
plt.xlabel('Variables')
plt.tight_layout()
plt.show()


# Generar matriz de confusión
cm = confusion_matrix(y_test, y_pred)

 #Gráfico tipo heatmap
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=modelo.classes_, yticklabels=modelo.classes_)
plt.title('Matriz de Confusión')
plt.ylabel('Etiqueta Real')
plt.xlabel('Etiqueta Predicha')
plt.tight_layout()
plt.show()

errores = []
arboles = list(range(1, 101))  # de 1 a 100 árboles

for n in arboles:
    modelo_temp = RandomForestClassifier(n_estimators=n, random_state=42, class_weight='balanced')
    modelo_temp.fit(X_train, y_train)
    y_temp = modelo_temp.predict(X_test)
    error = 1 - accuracy_score(y_test, y_temp)
    errores.append(error)
# Graficar error vs número de árboles
plt.figure(figsize=(10,6))
plt.plot(arboles, errores, marker='o')
plt.title('Error vs Número de Árboles')
plt.xlabel('Número de Árboles')
plt.ylabel('Error de Predicción')
plt.grid(True)
plt.tight_layout()
plt.show()
    
# Obtener las probabilidades de la clase positiva
# Vamos a usar predict_proba para obtener probabilidades en vez de etiquetas duras
y_prob = modelo.predict_proba(X_test)[:,1]  # Probabilidad de ser 'Fallecido'

# Necesitamos que y_test esté en formato binario (Fallecido = 1, Leve = 0)
# Asumimos que en tus datos 'Fallecido' es una categoría en y_test
# Vamos a convertir y_test en valores binarios
y_test_bin = (y_test == 'Fallecido').astype(int)

# Calcular fpr (False Positive Rate), tpr (True Positive Rate) y thresholds
fpr, tpr, thresholds = roc_curve(y_test_bin, y_prob)

# Calcular área bajo la curva (AUC)
roc_auc = auc(fpr, tpr)

# Graficar
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0,1], [0,1], 'k--', label='Modelo Aleatorio')
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC - Random Forest COVID')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()    


#Dibujar el primer árbol del Random Forest
plt.figure(figsize=(40,20))
plot_tree(modelo.estimators_[0], 
          feature_names=X.columns, 
          class_names=modelo.classes_, 
          filled=True, 
          rounded=True, 
          fontsize=5)
plt.title('Primer Árbol del Bosque')
plt.show()

# Suponiendo que ya tienes entrenado tu modelo y preparado el LabelEncoder le

# Pedir datos al usuario
edad_input = int(input("Ingrese la edad de la persona: "))
sexo_input = input("Ingrese el sexo (M/F): ")
Ciudad_input = input("Ingrese la Ciudad: ")
municipio_input = input("Ingrese la Ciudad: ")

# Codificar sexo y ubicación usando el mismo codificador que usaste antes
sexo_codificado = le.fit(df['Sexo']).transform([sexo_input])[0]
Ciudad_codificada = le.fit(df['Ciudad']).transform([Ciudad_input])[0]
municipio_codificada = le.fit(df['Municipio']).transform([municipio_input])[0]

# Crear el array de entrada
X_nuevo = pd.DataFrame({
    'Edad': [edad_input],
    'Sexo': [sexo_codificado],
    'Ciudad': [Ciudad_codificada],
    'Municipio':[municipio_codificada]

})

# Predecir probabilidades
probas = modelo.predict_proba(X_nuevo)

# Mostrar resultados
print(f"\nProbabilidad de Fallecer: {probas[0][list(modelo.classes_).index('Fallecido')]*100:.2f}%")
print(f"Probabilidad de Sobrevivir (Leve): {probas[0][list(modelo.classes_).index('Leve')]*100:.2f}%")

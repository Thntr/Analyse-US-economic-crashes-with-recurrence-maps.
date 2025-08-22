import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# Cargar los datos usando pandas
yield_data_df = pd.read_csv("yield_data_limited.csv")
unemployment_data_df = pd.read_csv("unemployment_data_limited.csv")
house_sales_data_df = pd.read_csv("house_sales_data_limited.csv")

# Extraer las fechas (se supone que hay una columna llamada 'Date' en cada archivo)
dates = yield_data_df['DATE'].values  # Suponiendo que las fechas son las mismas en todos los datasets

# Asegurarnos de usar solo las columnas numéricas
yield_data = yield_data_df['T10Y2Y'].values
unemployment_data = unemployment_data_df['UNRATE'].values
house_sales_data = house_sales_data_df['HSN1F'].values

# Construir la matriz de datos U
U = np.vstack((yield_data, unemployment_data, house_sales_data)).T

# Calcular la matriz de distancias
distances = squareform(pdist(U, metric="euclidean"))

#Histograma de distribucion de distancias euclidianas
plt.hist(np.ravel(distances))

# Establecer epsilon con base en estadísticas de las distancias
epsilon_mean = np.mean(distances)
epsilon_std = np.std(distances)
epsilon = epsilon_mean - 0.5 * epsilon_std  # Ajusta el coeficiente según lo deseado

print(f"Epsilon calculado: {epsilon:.4f}")

# Crear la matriz de recurrencia
recurrence_matrix = (distances <= epsilon).astype(int)

# Crear el mapa de recurrencia
plt.figure(figsize=(12, 10))
plt.imshow(recurrence_matrix, cmap="binary", origin="upper")

# Configurar los ejes para que muestren las fechas
num_points = recurrence_matrix.shape[0]
plt.xticks(ticks=np.arange(num_points), labels=dates, fontsize=6, rotation=90)  # Fechas en eje X
plt.yticks(ticks=np.arange(num_points), labels=dates, fontsize=6)  # Fechas en eje Y

# Agregar una malla para facilitar la identificación
plt.grid(visible=True, which='both', color='gray', linestyle='--', linewidth=0.5)

# Dibujar diagonales paralelas a la principal con mayor separación
spacing = num_points // 8  # Define el espacio entre diagonales
for d in range(spacing, num_points, spacing):  # Diagonales espaciadas por `spacing`
    # Por encima de la diagonal principal
    if d < num_points:
        plt.plot(np.arange(num_points - d), np.arange(d, num_points), color='red', linestyle='--', linewidth=1)
    # Por debajo de la diagonal principal
    plt.plot(np.arange(d, num_points), np.arange(num_points - d), color='red', linestyle='--', linewidth=1)

# Agregar etiquetas y título
plt.xlabel("Fechas (Eje Horizontal)", fontsize=12)
plt.ylabel("Fechas (Eje Vertical)", fontsize=12)
plt.title("Mapa de Recurrencia con Guías Diagonales Espaciadas", fontsize=14)

# Ajustar la escala del color
cbar = plt.colorbar()
cbar.set_label("Recurrencia (1: Blanco, 0: Negro)", fontsize=12)

# Mostrar el gráfico
plt.tight_layout()
plt.show()

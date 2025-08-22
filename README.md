# Generación de Mapas de Recurrencia con Datos Económicos de EE.UU.

Este repositorio contiene un script en Python que genera mapas de recurrencia, una herramienta utilizada en teoría del caos y análisis estadístico.

El script utiliza tres series de datos económicos de Estados Unidos:

Diferencia de los yields de los bonos del Tesoro a 2 y 10 años.

Tasa de desempleo.

Ventas de casas.<br><br>

📂 Archivos del repositorio

my_own_rmap.py → Script principal en Python que genera el mapa de recurrencia.

yield_data_limited.csv → Datos de la diferencia entre el bono a 2 y 10 años.

unemployment_data_limited.csv → Datos de desempleo en EE.UU.

home_sales_data_limited.csv → Datos de ventas de casas en EE.UU. <br><br>

⚙️ Requisitos

Antes de ejecutar el script, asegúrate de tener instalado:

Python 3.8+

Las siguientes librerías de Python:

<pre> pip install numpy matplotlib pandas pdist squareform </pre> <br><br>

🚀 Cómo usarlo

Descarga los 4 archivos (script.py + los tres .csv).

Colócalos en la misma carpeta.

Abre una terminal en esa carpeta y ejecuta:

<pre> python my_own_rmap.py </pre>

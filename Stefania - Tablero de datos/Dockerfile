# Imagen base
FROM python:3.11-slim

# Crear carpeta de trabajo
WORKDIR /app

# Copiar archivos
COPY . /app

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto del Dash
EXPOSE 8050

# Ejecutar la app
CMD ["python", "Dash_Completo.py"]

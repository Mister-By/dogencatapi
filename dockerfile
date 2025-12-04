#FROM python:3.8-slim

# # Dépendances système nécessaires
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     python3-dev \
#     libatlas-base-dev \
#     liblapack-dev \
#     libjpeg-dev \
#     libpng-dev \
#     ffmpeg \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# # Création du dossier app
# WORKDIR /app

# # Installer pip + dépendances python
# COPY requirements.txt .

# RUN pip install --upgrade pip
# RUN pip install --no-cache-dir -r requirements.txt

# # Copier le projet
# COPY . .

# # Exposer l’API
# EXPOSE 8000

# # Lancer l’API FastAPI
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


FROM python:3.10-slim

# Pour TensorFlow 2.10: besoin de ces libs
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installer les deps en premier pour utiliser le cache Docker
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copier le code et les modèles
COPY . .

# Le port exposé
EXPOSE 8001

# Commande de lancement
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]

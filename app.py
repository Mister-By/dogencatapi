from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from io import BytesIO
import base64

# ============================================
# Charger les modèles
# ============================================
model1 = load_model("best_model.h5")
model2 = load_model("best_model.h5")  # ou ton modèle pré-entraîné

res = ["Chat", "Chien"]

# ============================================
# Initialisation FastAPI et CORS
# ============================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou ["http://localhost:3000"] pour limiter
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Modèles de réponse
# ============================================
class RepError(BaseModel):
    error: bool
    message: str
    
class RepSuccess(BaseModel):
    error: bool
    prediction: str
    
class Data(BaseModel):
    image: str  # image en base64
    model: str  # 0 ou 1

# ============================================
# Endpoint de prédiction
# ============================================
@app.post("/predict", response_model=RepSuccess | RepError)
async def doPred(data: Data):
    # Vérification des données
    if not data.model or not data.image:
        return RepError(error=True, message="Une ou plusieurs données manquantes")

    try:
        choix = int(data.model)
    except ValueError:
        return RepError(error=True, message="Le modèle choisi doit être 0 ou 1")

    if choix != 0 and choix != 1:
        return RepError(error=True, message="Modèle choisi incorrect")
    
    # Conversion base64 -> image
    try:
        img_bytes = base64.b64decode(data.image.split(",")[-1]) 
        img = BytesIO(img_bytes)
    except Exception:
        return RepError(error=True, message="Impossible de lire l'image")

    # Sélection du modèle et taille
    if choix == 0:
        taille = 128
        mod = model1
    else:
        taille = 224
        mod = model2

    # Prédiction
    pred = make_pred(mod, img, taille)
    if pred == "X":
        return RepError(error=True, message="Nous n'arrivons pas à prédire cette photo")
    else:
        return RepSuccess(error=False, prediction=f'{pred}')

# ============================================
# Fonction de prédiction
# ============================================
def make_pred(mod, img, taille):
    try:
        img = image.load_img(img, target_size=(taille, taille))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        p = mod.predict(img_array)
        idmax = np.argmax(p[0])
        valmax = p[0][idmax]
        print(valmax)
        
        if valmax > 0.70:
            return f"{res[idmax]} à {valmax * 100} %"
        else:
            return "X"
    except Exception:
        return "X"

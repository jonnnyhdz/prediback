from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Union
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi.responses import FileResponse


# === CONFIGURACIÓN ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
excel_path = os.path.join(BASE_DIR, "StudentsSocial.xlsx")
principales = [
    "Mental_Health_Score",
    "Addicted_Score",
    "Affects_Academic_Performance",
    "Avg_Daily_Usage_Hours",
    "Sleep_Hours_Per_Night",
    "Conflicts_Over_Social_Media",
    "Relationship_Status"
]
clasificacion_discreta = ["Mental_Health_Score", "Addicted_Score", "Affects_Academic_Performance"]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class StudentInput(BaseModel):
    Student_ID: int
    Age: int
    Gender: Union[str, int]
    Academic_Level: Union[str, int]
    Country: str
    Avg_Daily_Usage_Hours: float
    Most_Used_Platform: str
    Conflicts_Over_Social_Media: int
    Sleep_Hours_Per_Night: float
    Relationship_Status: Union[str, int]


class PredictionRequest(StudentInput):
    pass

class AskRequest(BaseModel):
    student_id: int
    question: str

def reentrenar_modelos():
    df = pd.read_excel(excel_path)

    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

    df.dropna(subset=principales, inplace=True)

    label_encoders = {}
    scalers = {}

    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    for col in df.select_dtypes(include='number').columns:
        if col not in ['Student_ID'] + principales:
            scaler = StandardScaler()
            df[col] = scaler.fit_transform(df[[col]])
            scalers[col] = scaler

    models = {}
    model_columns = {}

    for target in principales:
        X = df.drop(columns=[target])
        y = df[target]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        models[target] = model
        model_columns[target] = X.columns.tolist()

    joblib.dump(models, os.path.join(BASE_DIR, "modelos_entrenados.pkl"))
    joblib.dump(label_encoders, os.path.join(BASE_DIR, "encoders.pkl"))
    joblib.dump(model_columns, os.path.join(BASE_DIR, "columnas_modelos.pkl"))
    joblib.dump(scalers, os.path.join(BASE_DIR, "scalers.pkl"))

@app.post("/predict")
def predict(data: PredictionRequest):
    input_data = data.model_dump()

    df_original = pd.read_excel(excel_path)
    columnas_validas = df_original.columns

    encoders = joblib.load(os.path.join(BASE_DIR, "encoders.pkl"))
    models = joblib.load(os.path.join(BASE_DIR, "modelos_entrenados.pkl"))
    model_columns = joblib.load(os.path.join(BASE_DIR, "columnas_modelos.pkl"))
    scalers = joblib.load(os.path.join(BASE_DIR, "scalers.pkl"))

    df = pd.DataFrame([input_data])
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.lower()
        if col in encoders:
            try:
                df[col] = encoders[col].transform(df[col])
            except:
                df[col] = 0
        if col in scalers:
            df[col] = scalers[col].transform(df[[col]])

    pred_principales = {}
    for target in models:
        cols = model_columns[target]
        for col in cols:
            if col not in df.columns:
                df[col] = 0
        pred = models[target].predict(df[cols])[0]
        pred_principales[target] = int(round(pred)) if target in clasificacion_discreta else float(round(pred, 2))

    # Guardar solo ciertas predicciones
    nueva_fila = input_data.copy()
    for col in ["Mental_Health_Score", "Addicted_Score", "Affects_Academic_Performance"]:
        nueva_fila[col] = pred_principales.get(col)

    for col in columnas_validas:
        if col not in nueva_fila:
            nueva_fila[col] = None

    df_nuevo = pd.concat([df_original, pd.DataFrame([nueva_fila])], ignore_index=True)
    df_nuevo.to_excel(excel_path, index=False)

    reentrenar_modelos()

    return {
        "predictions": pred_principales,
        "message": "✅ Predicción completada con resultados principales precargados."
    }



@app.post("/ask")
def ask_question_post(data: AskRequest):
    student_id = data.student_id
    pregunta = data.question.strip().lower()

    df = pd.read_excel(excel_path)
    fila = df[df["Student_ID"] == student_id].copy()

    if fila.empty:
        return {"error": f"❌ No se encontró ningún registro con Student_ID={student_id}"}

    encoders = joblib.load(os.path.join(BASE_DIR, "encoders.pkl"))
    models = joblib.load(os.path.join(BASE_DIR, "modelos_entrenados.pkl"))
    model_columns = joblib.load(os.path.join(BASE_DIR, "columnas_modelos.pkl"))
    scalers = joblib.load(os.path.join(BASE_DIR, "scalers.pkl"))

    for col in fila.columns:
        if fila[col].dtype == object:
            fila[col] = fila[col].astype(str).str.lower()
        if col in encoders:
            try:
                fila[col] = encoders[col].transform(fila[col])
            except:
                fila[col] = 0
        if col in scalers:
            fila[col] = scalers[col].transform(fila[[col]])

    etiquetas_humanas = {
        "Mental_Health_Score": "salud mental",
        "Addicted_Score": "adicción a redes",
        "Conflicts_Over_Social_Media": "conflictos sociales",
        "Avg_Daily_Usage_Hours": "uso de redes sociales",
        "Relationship_Status": "estado de relación",
        "Sleep_Hours_Per_Night": "horas de sueño",
        "Affects_Academic_Performance": "rendimiento escolar"
    }

    promedios_ideales = {
        "Sleep_Hours_Per_Night": 8.0,
        "Avg_Daily_Usage_Hours": 3.0
    }

    preguntas_entrenadas = {
        "¿cómo está mi salud mental?": ["Mental_Health_Score"],
        "¿tengo conflictos por redes sociales?": ["Conflicts_Over_Social_Media"],
        "¿soy adicto a las redes sociales?": ["Addicted_Score"],
        "¿cuánto uso las redes sociales?": ["Avg_Daily_Usage_Hours"],
        "¿duermo lo suficiente?": ["Sleep_Hours_Per_Night"],
        "¿cómo afecta mi rendimiento académico?": ["Affects_Academic_Performance"],
        "¿cómo será mi próxima relación?": ["Relationship_Status", "Mental_Health_Score", "Addicted_Score"],
        "¿mi salud emocional está bien?": ["Mental_Health_Score", "Conflicts_Over_Social_Media", "Sleep_Hours_Per_Night"],
        "¿estoy equilibrado emocionalmente?": ["Mental_Health_Score", "Addicted_Score", "Sleep_Hours_Per_Night"],
        "¿mi relación me afecta?": ["Relationship_Status", "Affects_Academic_Performance", "Mental_Health_Score"]
    }

    vectorizer = TfidfVectorizer().fit(preguntas_entrenadas.keys())
    vectores = vectorizer.transform(list(preguntas_entrenadas.keys()) + [pregunta])
    similitudes = cosine_similarity(vectores[-1], vectores[:-1])[0]
    mejor_index = int(np.argmax(similitudes))
    pregunta_base = list(preguntas_entrenadas.keys())[mejor_index]
    targets = preguntas_entrenadas[pregunta_base]
    target = targets[0]

    predicciones = []
    for t in targets:
        model = models[t]
        cols = model_columns[t]
        for col in cols:
            if col not in fila.columns:
                fila[col] = 0
        pred = model.predict(fila[cols])[0]
        pred_format = int(round(pred)) if t in clasificacion_discreta else float(round(pred, 2))
        predicciones.append((t, pred_format))

    valor_usuario = predicciones[0][1]

    if target in df.columns:
        try:
            promedio_general = pd.to_numeric(df[target], errors='coerce').dropna().mean()
            promedio_general = round(promedio_general, 2)
        except:
            promedio_general = None
    else:
        promedio_general = None

    ideal = promedios_ideales.get(target, promedio_general)

    comparacion = None
    if isinstance(valor_usuario, (int, float)) and isinstance(ideal, (int, float)):
        if valor_usuario > ideal + 1:
            comparacion = "por encima de lo recomendado"
        elif valor_usuario < ideal - 1:
            comparacion = "por debajo de lo recomendado"
        else:
            comparacion = "dentro del rango saludable"

    analisis = f"📊 Según tus respuestas, {etiquetas_humanas.get(target, target)} es de {valor_usuario}."
    if promedio_general is not None and comparacion:
        analisis += f" El promedio general es {promedio_general}. Estás {comparacion}."

    return {
        "student_id": student_id,
        "question": data.question,
        "target": target,
        "valor_usuario": valor_usuario,
        "promedio_general": promedio_general,
        "comparacion": comparacion,
        "analisis": analisis
    }


@app.get("/descargar-excel")
def descargar_excel():
    ruta = excel_path  # usa el mismo path que ya usas para guardar
    if not os.path.exists(ruta):
        return {"error": "❌ No hay respuestas registradas aún"}
    
    return FileResponse(
        ruta,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="respuestas_encuesta.xlsx"
    )

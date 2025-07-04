from fastapi import FastAPI
from pydantic import BaseModel, Field
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

@app.get("/")
def root():
    return {"status": "🟢 Backend activo. Ve a /docs para ver la documentación."}


class StudentInput(BaseModel):
    Student_ID: int
    Age: int
    Gender: Union[str, int]
    Academic_Level: Union[str, int]
    Country: str
    Avg_Daily_Usage_Hours: float
    Most_Used_Platform: str
    Conflicts_Over_Social_Media: int = Field(..., ge=0, le=10)
    Sleep_Hours_Per_Night: float
    Relationship_Status: Union[str, int]


class PredictionRequest(StudentInput):
    pass

class AskRequest(BaseModel):
    student_id: int
    question: str


@app.post("/predict")
def predict(data: PredictionRequest):
    input_data = data.model_dump()

    # === Cargar artefactos y dataset original ===
    df_original = pd.read_excel(excel_path)
    columnas_validas = df_original.columns

    encoders = joblib.load(os.path.join(BASE_DIR, "encoders.pkl"))
    models = joblib.load(os.path.join(BASE_DIR, "modelos_entrenados.pkl"))
    model_columns = joblib.load(os.path.join(BASE_DIR, "columnas_modelos.pkl"))
    scalers = joblib.load(os.path.join(BASE_DIR, "scalers.pkl"))

    df = pd.DataFrame([input_data])

    # === Preprocesamiento: limpieza, codificación y escalado ===
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.lower().str.strip()

        if col in encoders:
            le = encoders[col]
            val = df[col].iloc[0]
            if val in le.classes_:
                df[col] = le.transform([val])
            else:
                print(f"⚠️ Valor desconocido para '{col}': {val}. Se asigna 0.")
                df[col] = [0]

        if col in scalers:
            df[col] = scalers[col].transform(df[[col]])

    # === Agregar variables derivadas ===
    df["Is_Overuser"] = (df.get("Avg_Daily_Usage_Hours", 0) > 4).astype(int)
    df["Sleeps_Enough"] = (df.get("Sleep_Hours_Per_Night", 0) >= 7).astype(int)
    df["Usage_Conflict_Ratio"] = df.get("Conflicts_Over_Social_Media", 0) / (df.get("Avg_Daily_Usage_Hours", 0) + 1)

    pred_principales = {}

    # === 1. Predecir Mental_Health_Score y Addicted_Score ===
    for target in ["Mental_Health_Score", "Addicted_Score"]:
        cols = model_columns[target]
        df_input = df[cols].copy() if all(col in df.columns for col in cols) else df.copy()
        for col in cols:
            if col not in df_input.columns:
                df_input[col] = 0
        pred = models[target].predict(df_input[cols])[0]
        if target in scalers:
            try:
                pred = scalers[target].inverse_transform([[pred]])[0][0]
            except:
                pass
        pred_principales[target] = float(round(pred, 2))
        df[target] = pred_principales[target]  # Agregar al dataframe para usarlo después

    # === 2. Predecir Affects_Academic_Performance usando los anteriores ===
    target = "Affects_Academic_Performance"
    cols = model_columns[target]
    df_input = df[cols].copy() if all(col in df.columns for col in cols) else df.copy()
    for col in cols:
        if col not in df_input.columns:
            df_input[col] = 0
    pred = models[target].predict(df_input[cols])[0]
    if target in scalers:
        try:
            pred = scalers[target].inverse_transform([[pred]])[0][0]
        except:
            pass
    pred_principales[target] = float(round(pred, 2))

    # === 3. Predicciones adicionales (si hay más modelos)
    for target in models:
        if target in pred_principales:
            continue  # Ya fue predicho
        cols = model_columns[target]
        df_input = df.copy()
        for col in cols:
            if col not in df_input.columns:
                df_input[col] = 0
        pred = models[target].predict(df_input[cols])[0]
        if target in scalers:
            try:
                pred = scalers[target].inverse_transform([[pred]])[0][0]
            except:
                pass
        pred_principales[target] = float(round(pred, 2))

    # === Guardar nueva fila ===
    nueva_fila = input_data.copy()
    for col in pred_principales:
        nueva_fila[col] = pred_principales[col]
    for col in columnas_validas:
        if col not in nueva_fila:
            nueva_fila[col] = None

    df_nuevo = pd.concat([df_original, pd.DataFrame([nueva_fila])], ignore_index=True)
    df_nuevo.to_excel(excel_path, index=False)

    return {
        "predictions": pred_principales,
        "variables_por_modelo": model_columns,
        "message": "✅ Predicción completada correctamente sin inferencia cruzada."
    }

@app.post("/ask")
def ask_question_post(data: AskRequest):
    import warnings
    warnings.filterwarnings("ignore")

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
        "¿cómo está mi salud mental?": ["Mental_Health_Score", "Conflicts_Over_Social_Media", "Addicted_Score"],
        "¿tengo conflictos por redes sociales?": ["Conflicts_Over_Social_Media", "Addicted_Score", "Avg_Daily_Usage_Hours"],
        "¿soy adicto a las redes sociales?": ["Addicted_Score", "Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night"],
        "¿cuánto uso las redes sociales?": ["Avg_Daily_Usage_Hours"],
        "¿es demasiado mi tiempo en redes?": ["Avg_Daily_Usage_Hours"],
        "¿duermo lo suficiente?": ["Sleep_Hours_Per_Night", "Mental_Health_Score"],
        "¿cómo afecta mi rendimiento académico?": ["Affects_Academic_Performance", "Addicted_Score"],
        "¿cómo será mi próxima relación?": ["Relationship_Status", "Mental_Health_Score", "Addicted_Score"],
        "¿mi salud emocional está bien?": ["Mental_Health_Score", "Conflicts_Over_Social_Media", "Sleep_Hours_Per_Night"],
        "¿estoy equilibrado emocionalmente?": ["Mental_Health_Score", "Addicted_Score", "Sleep_Hours_Per_Night"],
        "¿mi relación me afecta?": ["Relationship_Status", "Affects_Academic_Performance", "Mental_Health_Score"]
    }

    # Similaridad
    vectorizer = TfidfVectorizer().fit(preguntas_entrenadas.keys())
    vectores = vectorizer.transform(list(preguntas_entrenadas.keys()) + [pregunta])
    similitudes = cosine_similarity(vectores[-1], vectores[:-1])[0]
    mejor_index = int(np.argmax(similitudes))
    pregunta_base = list(preguntas_entrenadas.keys())[mejor_index]
    targets = preguntas_entrenadas[pregunta_base]

    resultados = []

    for target in targets:
        if target not in models:
            continue

        model = models[target]
        cols = model_columns[target]

        for col in cols:
            if col not in fila.columns:
                fila[col] = 0

        try:
            pred = model.predict(fila[cols])[0]
            pred = int(round(pred)) if target in ["Conflicts_Over_Social_Media", "Addicted_Score", "Mental_Health_Score"] else float(round(pred, 2))
        except:
            continue

        promedio = None
        if target in df.columns:
            try:
                promedio = pd.to_numeric(df[target], errors='coerce').dropna().mean()
                promedio = round(promedio, 2)
            except:
                promedio = None

        ideal = promedios_ideales.get(target, promedio)

        comparacion = None
        if isinstance(pred, (int, float)) and isinstance(ideal, (int, float)):
            if pred > ideal + 1:
                comparacion = "por encima de lo esperado"
            elif pred < ideal - 1:
                comparacion = "por debajo de lo esperado"
            else:
                comparacion = "en el rango saludable"

        texto = f"🔍 Tu nivel de {etiquetas_humanas.get(target, target)} es **{pred}**."
        if promedio is not None:
            texto += f" El promedio general es {promedio}, estás {comparacion}."

        resultados.append({
            "target": target,
            "etiqueta": etiquetas_humanas.get(target, target),
            "valor_usuario": pred,
            "promedio_general": promedio,
            "comparacion": comparacion,
            "analisis": texto
        })

    # Generador de conclusión solo basado en los targets relevantes
    def construir_respuesta_final(pregunta, resultados, targets_relacionados):
        conclusiones = []

        for res in resultados:
            if res["target"] not in targets_relacionados:
                continue

            etiqueta = res["etiqueta"].lower()
            comparacion = res["comparacion"]

            if "salud mental" in etiqueta:
                if "por debajo" in comparacion:
                    conclusiones.append("tu salud mental podría estar deteriorada")
                else:
                    conclusiones.append("tu salud mental es estable")
            elif "adicción" in etiqueta:
                if "por encima" in comparacion:
                    conclusiones.append("muestras señales de adicción")
                else:
                    conclusiones.append("no presentas señales de adicción")
            elif "conflicto" in etiqueta:
                if "por encima" in comparacion:
                    conclusiones.append("hay conflictos sociales presentes")
                else:
                    conclusiones.append("no se observan conflictos sociales")
            elif "rendimiento" in etiqueta:
                if "por debajo" in comparacion:
                    conclusiones.append("tu rendimiento académico podría estar afectado")
                else:
                    conclusiones.append("tu rendimiento académico es adecuado")
            elif "sueño" in etiqueta:
                if "por debajo" in comparacion:
                    conclusiones.append("tienes falta de sueño")
                else:
                    conclusiones.append("tus horas de sueño son saludables")
            elif "uso" in etiqueta:
                if "por encima" in comparacion:
                    conclusiones.append("usas redes sociales en exceso")
                elif "por debajo" in comparacion:
                    conclusiones.append("usas redes sociales menos que el promedio")
                else:
                    conclusiones.append("tu uso de redes es moderado")

        if not conclusiones:
            return "No se pudo generar una conclusión clara."

        return "En resumen, " + ", ".join(conclusiones) + "."

    respuesta_final = construir_respuesta_final(pregunta, resultados, targets)

    return {
        "student_id": student_id,
        "pregunta_original": data.question,
        "pregunta_interpretada": pregunta_base,
        "resultados": resultados,
        "respuesta_final": respuesta_final
    }



@app.get("/descargar-excel")
def descargar_excel():
    if not os.path.exists(excel_path):
        return {"error": "❌ No hay respuestas registradas aún"}
    
    return FileResponse(
        excel_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="respuestas_encuesta.xlsx"
    )

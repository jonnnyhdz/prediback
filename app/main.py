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

    fila = df[df["Student_ID"] == student_id].copy().iloc[[-1]]

    if fila.empty:
        return {"error": f"No se encontró ningún registro con Student_ID={student_id}"}

    valores_predichos = {
        "Mental_Health_Score": float(fila["Mental_Health_Score"].values[0]),
        "Addicted_Score": float(fila["Addicted_Score"].values[0]),
        "Conflicts_Over_Social_Media": float(fila["Conflicts_Over_Social_Media"].values[0]),
        "Sleep_Hours_Per_Night": float(fila["Sleep_Hours_Per_Night"].values[0]),
        "Avg_Daily_Usage_Hours": float(fila["Avg_Daily_Usage_Hours"].values[0]),
        "Affects_Academic_Performance": float(fila["Affects_Academic_Performance"].values[0]),
        "Relationship_Status": float(fila["Relationship_Status"].values[0])
    }

    etiquetas_humanas = {
        "Mental_Health_Score": "salud mental",
        "Addicted_Score": "uso de redes sociales",
        "Conflicts_Over_Social_Media": "conflictos por redes",
        "Avg_Daily_Usage_Hours": "tiempo que pasas en redes",
        "Sleep_Hours_Per_Night": "horas de sueño",
        "Affects_Academic_Performance": "afectación al rendimiento escolar",
        "Relationship_Status": "estado emocional por tu relación"
    }

    ideales = {
        "Sleep_Hours_Per_Night": 8.0,
        "Avg_Daily_Usage_Hours": 3.0
    }

    promedios = {
        k: round(pd.to_numeric(df[k], errors='coerce').dropna().mean(), 2)
        for k in valores_predichos
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

    # Interpretar pregunta
    vectorizer = TfidfVectorizer().fit(preguntas_entrenadas.keys())
    vectores = vectorizer.transform(list(preguntas_entrenadas.keys()) + [pregunta])
    similitudes = cosine_similarity(vectores[-1], vectores[:-1])[0]
    mejor_index = int(np.argmax(similitudes))
    pregunta_base = list(preguntas_entrenadas.keys())[mejor_index]
    targets = preguntas_entrenadas[pregunta_base]

    frases = []
    recomendaciones = []

    target_principal = targets[0]
    valor_principal = valores_predichos[target_principal]
    promedio_principal = promedios[target_principal]
    etiqueta_principal = etiquetas_humanas[target_principal]

    # Porcentaje estimado solo si es aplicable
    porcentaje_estimado = None
    if target_principal in ["Mental_Health_Score", "Addicted_Score"]:
        porcentaje_estimado = int((valor_principal / 10) * 100)
    elif target_principal == "Affects_Academic_Performance":
        porcentaje_estimado = int((valor_principal / 10) * 100)  # Pero se interpreta como % de afectación

    # Frases y recomendaciones
    for target in targets:
        valor = valores_predichos[target]
        etiqueta = etiquetas_humanas[target]

        if target == "Mental_Health_Score":
            if valor >= 7:
                frases.append("Tu salud mental se ve bastante estable. Eso es algo que vale la pena cuidar.")
            elif valor >= 4:
                frases.append("Últimamente podrías estar cargando más de lo que parece. Está bien tomarse un respiro.")
                recomendaciones.append("Habla con alguien de confianza o dedica tiempo a actividades que disfrutes.")
            else:
                frases.append("Tu salud mental podría necesitar más atención. No estás solo en esto.")
                recomendaciones.append("Busca apoyo emocional, incluso algo tan simple como una charla ayuda mucho.")

        elif target == "Addicted_Score":
            if valor >= 7:
                frases.append("Pareces estar muy pegado a las redes. Eso podría quitarte tiempo valioso.")
                recomendaciones.append("Fija momentos sin pantalla durante tu día. Verás cómo mejora tu enfoque.")
            elif valor >= 4:
                frases.append("Tu relación con las redes es intermedia. Observa si a veces te desconectan más de lo que te conectan.")
            else:
                frases.append("Tienes un uso bastante equilibrado de redes sociales. ¡Eso es genial!")

        elif target == "Conflicts_Over_Social_Media":
            if valor >= 6:
                frases.append("Has vivido varios roces en redes. Tal vez sería bueno evitar discusiones digitales.")
                recomendaciones.append("Piensa dos veces antes de responder en caliente. Tu paz vale más.")
            elif valor >= 3:
                frases.append("Has tenido algunos conflictos, pero nada grave. Aún así, cuida tu energía.")
            else:
                frases.append("Casi no tienes problemas por redes. ¡Eso habla bien de cómo las manejas!")

        elif target == "Sleep_Hours_Per_Night":
            if valor >= 7.5:
                frases.append("Duermes lo suficiente, y eso es clave para sentirte bien.")
            elif valor >= 6:
                frases.append("Tus horas de sueño están algo justas. Un poco más de descanso podría ayudarte.")
                recomendaciones.append("Evita usar el celular justo antes de dormir.")
            else:
                frases.append("Duermes muy poco, y eso puede pasarte factura en el día.")
                recomendaciones.append("Intenta acostarte más temprano, incluso solo 30 minutos antes puede marcar la diferencia.")

        elif target == "Avg_Daily_Usage_Hours":
            if valor > 5:
                frases.append("Pasas bastante tiempo en redes. Quizás podrías darte más espacios offline.")
                recomendaciones.append("Reserva una hora al día para desconectarte totalmente.")
            elif valor > 3:
                frases.append("Tu uso de redes es algo elevado. Vale la pena monitorearlo.")
            else:
                frases.append("Tienes un uso de redes muy controlado. ¡Sigue así!")

        elif target == "Affects_Academic_Performance":
            if valor >= 7:
                frases.append("Parece que el uso de redes sociales está afectando bastante tus estudios.")
                recomendaciones.append("Intenta desconectarte al momento de estudiar y crear rutinas sin pantallas.")
            elif valor >= 4:
                frases.append("Podría estar costándote concentrarte. Quizás valga la pena ajustar horarios.")
                recomendaciones.append("Evita tener el celular cerca cuando estudies.")
            else:
                frases.append("Las redes no parecen interferir mucho en tu rendimiento. ¡Bien hecho!")

        elif target == "Relationship_Status":
            if valor >= 7:
                frases.append("Tu relación emocional parece ser un apoyo positivo.")
            elif valor >= 4:
                frases.append("No parece afectarte mucho tu relación. Pero si algo te inquieta, siempre es bueno hablar.")
            else:
                frases.append("Tal vez tu situación sentimental esté influyendo en cómo te sientes día a día.")
                recomendaciones.append("No olvides cuidarte emocionalmente, sobre todo si estás pasando por algo difícil.")

    # Redactar resumen
    resumen = f"{frases[0]}"
    if porcentaje_estimado is not None:
        resumen += f" Según lo que vimos, tu nivel de {etiqueta_principal} está alrededor del {porcentaje_estimado}%."

    if len(frases) > 1:
        resumen += " " + " ".join(frases[1:])

    if recomendaciones:
        resumen += "\n\n📌 Consejos útiles que podrían ayudarte:\n- " + "\n- ".join(recomendaciones)

    resumen += "\n\nGracias por confiar en este espacio. Las respuestas fueron analizadas por modelos de IA que, como una brújula, te orientan considerando todo tu contexto para darte una respuesta útil y cercana."

    return {
        "student_id": student_id,
        "pregunta_original": data.question,
        "pregunta_interpretada": pregunta_base,
        "resultados": [{
            "target": target_principal,
            "etiqueta": etiqueta_principal,
            "valor_usuario": valor_principal,
            "promedio_general": promedio_principal,
            "comparacion": "",
            "analisis": ""
        }],
        "respuesta_final": resumen
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

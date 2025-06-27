dimport os
import pandas as pd

# Ruta al archivo Excel en el servidor
EXCEL_PATH = "data/encuestas.xlsx"

# Encabezados esperados en la encuesta
HEADERS = [
    "edad", "genero", "nivel_estudios", "horas_redes_sociales",
    "red_social_favorita", "afectacion_desempeno", "horas_sueno",
    "estado_emocional", "relacion_actual", "conflictos_redes", "uso_redes"
]

def guardar_respuesta(respuesta: dict):
    # Crear carpeta si no existe
    os.makedirs(os.path.dirname(EXCEL_PATH), exist_ok=True)

    # Si no existe, lo creamos con encabezados
    if not os.path.isfile(EXCEL_PATH):
        df = pd.DataFrame(columns=HEADERS)
        df.to_excel(EXCEL_PATH, index=False)

    # Leemos archivo existente
    df = pd.read_excel(EXCEL_PATH)

    # Agregamos nueva fila
    nueva_fila = pd.DataFrame([respuesta])
    df = pd.concat([df, nueva_fila], ignore_index=True)

    # Guardamos
    df.to_excel(EXCEL_PATH, index=False)

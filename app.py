# ============================================================
# CABECERA
# ============================================================
# Alumno: Javier Cabrera Roca
# URL Streamlit Cloud: https://javiercabrera-bc5-mda13.streamlit.app
# URL GitHub: https://github.com/jacaroca/JavierCabrera_BC5

# ============================================================
# IMPORTS
# ============================================================
# Streamlit: framework para crear la interfaz web
# pandas: manipulación de datos tabulares
# plotly: generación de gráficos interactivos
# openai: cliente para comunicarse con la API de OpenAI
# json: para parsear la respuesta del LLM (que llega como texto JSON)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
# Modelo de OpenAI. No lo cambies.
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# >>> SYSTEM PROMPT — TU TRABAJO PRINCIPAL ESTÁ AQUÍ <<<
# -------------------------------------------------------
# El system prompt es el conjunto de instrucciones que recibe el LLM
# ANTES de la pregunta del usuario. Define cómo se comporta el modelo:
# qué sabe, qué formato debe usar, y qué hacer con preguntas inesperadas.
#
# Puedes usar estos placeholders entre llaves — se rellenan automáticamente
# con información real del dataset cuando la app arranca:
#   {fecha_min}             → primera fecha del dataset
#   {fecha_max}             → última fecha del dataset
#   {plataformas}           → lista de plataformas (Android, iOS, etc.)
#   {reason_start_values}   → valores posibles de reason_start
#   {reason_end_values}     → valores posibles de reason_end
#
# IMPORTANTE: como el prompt usa llaves para los placeholders,
# si necesitas escribir llaves literales en el texto (por ejemplo para
# mostrar un JSON de ejemplo), usa doble llave: {{ y }}
#
SYSTEM_PROMPT = """

    ## ROL:
    Actúa como un Data Analyst, experto en descriptive analytics, dentro del sector musical, 
    con experiencia en análisis de usuarios, y que por esa razón conoce el funcionamiento y las 
    métricas de Spotify. Tu relación con el usuario es la de un amigo al que le dan curiosidad 
    sus datos de reproducción de Spotify. Por esta razón, acaba de acudir a ti, para ver si puedes 
    analizar el dataset y responder a las preguntas que se le ocurran sobre este. Como buen analista,
    te limitas a dar la información que te pide tu amigo, no más, de manera que si tu amigo te pidiera
    saber un único elemento, le darías solo este; si te pidiera 10, le darías exactamente 10.

    Como analista con experiencia en el sector, controlas a la perfección 5 tipos de métricas:
    A. Rankings y favoritos del usuario
    B. Evolución temporal en los usos del usuario
    C. Patrones de uso del usuario
    D. Comportamiento de escucha del usuario
    E. Comparación entre períodos temporales
                
    ## OBJETIVO:
    Con el DataFrame y tras la consulta del usuario, generar un código JSON en el que se
    de una interpretación descriptiva al usuario de lo que pide.
                    
    ## CONTEXTO ANALÍTICO: 
    Columnas y tipos de datos: {schema}
    Primera fecha: {fecha_min}
    Última fecha: {fecha_max}
    Meses: {month_values}
    Días de la semana: {day_of_week_values}
    Épocas del año: {season_values}
    ts: Timestamp de fin de reproducción (ISO 8601, UTC)
    minutes_played: Minutos de reproducción efectiva de la canción. Se considerará escucha cuando la canción se haya reproducido entera (reason_end= trackdone o endplay)
    reason_start: motivo de inicio de la canción
    reason_end: motivo de fin de la canción
    shuffle: si el modo aleatorio estaba activado
    skipped: si la canción se saltó. Es de tipo booleano (True = saltada, False = no saltada)
    Razones de comienzo de canción: {reason_start_values}
    Razones de fin de canción: {reason_end_values}

    ## INSTRUCCIONES OBLIGATORIAS:
    1. El DataFrame se llama exactamente \'df\' y ya está cargado en memoria.
    2. Basa tu análisis exclusivamente en los datos proporcionados, el glosario y la pregunta recibida, sin inventar 
    tendencias ni comparaciones. Cualquier pregunta cuya respuesta requiera de algún elemento que no 
    tienes, considérala "tipo":"fuera_de_alcance".
    3. Limita tu análisis a los 5 tipos de métricas que conoces. Si la pregunta se aparta de estas, 
    considéralo como "tipo":"fuera_de_alcance".
    4. En "interpretacion", incluye una interpretación general, remitiendo al usuario al gráfico para el resultado concreto, 
    pero no menciones ni describas elementos técnicos (código, DataFrame tabla, columnas o metodología).
    5. El tono en "interpretacion" debe ser informal y cercano.
    6. Si los datos son insuficientes para una conclusión sólida, indícalo.
    7. No ofrezcas más de lo contenido en la pregunta y no realices inferencias inventadas.
                                        
    ## FORMATO DE RESPUESTA:
    - Devolverás un string en formato JSON con esta forma:
        {{"tipo": "grafico","codigo": "...", "interpretacion": "..."}}
        {{"tipo": "fuera_de_alcance", "codigo": "",    "interpretacion": "..."}}
    - El "tipo":"grafico" se utilizará cuando la respuesta se pueda dar con los datos proporcionados y se mantenga dentro 
    de los tipos de métricas que conoces. "interpretacion" consiste en una representación gráfica, acompañada de una breve 
    interpretación de lo analizado.
    - El "tipo":"fuera_de_alcance" se utilizará en los momentos indicados dentro de las INSTRUCCIONES OBLIGATORIAS.
    "interpretacion" consiste en una explicación de que no puedes dar una respuesta a la consulta porque no tienes
    información suficiente para ello.
    - Empieza directamente con el contenido, sin introducciones.
    - En el campo "interpretacion", usa exclusivamente Markdown enriquecido (negritas, listas).
    - Dentro del campo `"codigo"` escapa las comillas dobles usando `\"`.
    - Cuando la respuesta sea un dato simple, represéntala en un gráfico de barras
    - En caso de ser "tipo":"grafico", el resultado final debe almacenarse en la variable \'fig\', 
    que debe ser una figura de Plotly (px o go) para poder mostrarlo o graficarlo.
"""


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
# Esta función se ejecuta UNA SOLA VEZ gracias a @st.cache_data.
# Lee el fichero JSON y prepara el DataFrame para que el código
# que genere el LLM sea lo más simple posible.
#
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")

    df = df.rename(columns={'ms_played':'minutes_played','master_metadata_track_name':'track_name','master_metadata_album_artist_name':'main_artist', 'master_metadata_album_album_name':'album_name', 'spotify_track_uri':'track_code'})
    
    df['ts'] = pd.to_datetime(df['ts'])
    df['skipped'] = df['skipped'].fillna(0).astype(bool)
    
    df['minutes_played'] = (df['minutes_played']/60000).round(2)

    cond = df['track_code'].str.contains('track')
    df = df[cond]

    df['hour'] = df['ts'].dt.hour
    df['day_of_month'] = df['ts'].dt.day
    df['day_of_week'] = df['ts'].dt.day_name()
    df['month'] = df['ts'].dt.month_name()
    
    season_map = {
        'December': 'Winter','January': 'Winter','February':'Winter',
        'March': 'Spring','April': 'Spring', 'May':'Spring',
        'June': 'Summer', 'July': 'Summer', 'August': 'Summer',
        'September': 'Autumn', 'October': 'Autumn', 'November': 'Autumn'
    }
    df['season'] = df['month'].map(season_map)

    return df

def build_prompt(df):
    """
    Inyecta información dinámica del dataset en el system prompt.
    Los valores que calcules aquí reemplazan a los placeholders
    {fecha_min}, {fecha_max}, etc. dentro de SYSTEM_PROMPT.

    Si añades columnas nuevas en load_data() y quieres que el LLM
    conozca sus valores posibles, añade aquí el cálculo y un nuevo
    placeholder en SYSTEM_PROMPT.
    """
    schema = df.dtypes.to_string()
    fecha_min = df["ts"].min()
    fecha_max = df["ts"].max()
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()
    month_values = df["month"].unique().tolist()
    day_of_week_values = df["day_of_week"].unique().tolist()
    season_values = df["season"].unique().tolist()

    return SYSTEM_PROMPT.format(
        schema=schema,
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
        month_values=month_values,
        day_of_week_values=day_of_week_values,
        season_values=season_values
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
# Esta función envía DOS mensajes a la API de OpenAI:
# 1. El system prompt (instrucciones generales para el LLM)
# 2. La pregunta del usuario
#
# El LLM devuelve texto (que debería ser un JSON válido).
# temperature=0.2 hace que las respuestas sean más predecibles.
#
# No modifiques esta función.
#
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
# El LLM devuelve un string que debería ser un JSON con esta forma:
#
#   {"tipo": "grafico",          "codigo": "...", "interpretacion": "..."}
#   {"tipo": "fuera_de_alcance", "codigo": "",    "interpretacion": "..."}
#
# Esta función convierte ese string en un diccionario de Python.
# Si el LLM envuelve el JSON en backticks de markdown (```json...```),
# los limpia antes de parsear.
#
# No modifiques esta función.
#
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
# El LLM genera código Python como texto. Esta función lo ejecuta
# usando exec() y busca la variable `fig` que el código debe crear.
# `fig` debe ser una figura de Plotly (px o go).
#
# El código generado tiene acceso a: df, pd, px, go.
#
# No modifiques esta función.
#
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
# Toda la interfaz de usuario. No modifiques esta sección.
#

# Configuración de la página
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
# Lee la contraseña de secrets.toml. Si no coincide, no muestra la app.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

# Cargar datos y construir el prompt con información del dataset
df = load_data()
system_prompt = build_prompt(df)

# Caja de texto para la pregunta del usuario
if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    # Mostrar la pregunta en la interfaz
    with st.chat_message("user"):
        st.write(prompt)

    # Generar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                # 1. Enviar pregunta al LLM
                raw = get_response(prompt, system_prompt)

                # 2. Parsear la respuesta JSON
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    # Pregunta fuera de alcance: mostrar solo texto
                    st.write(parsed["interpretacion"])
                else:
                    # Pregunta válida: ejecutar código y mostrar gráfico
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")


# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# Responde a estas tres preguntas con tus palabras. Sé concreto
# y haz referencia a tu solución, no a generalidades.
# No superes las 30 líneas en total entre las tres respuestas.
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    ¿Cómo funciona la arquitectura de tu aplicación? ¿Qué recibe
#    el LLM? ¿Qué devuelve? ¿Dónde se ejecuta el código generado?
#    ¿Por qué el LLM no recibe los datos directamente?
#
#   El LLM recibe dos cosas: el system prompt que le da instrucciones
#   sobre cómo actuar y la solicitud del usuario. En base a estas, ge-
#   nera un string en formato JSON con el código Python para la visua-
#   lización y la interpretación que hace de los datos. El código se
#   ejecuta en local y el LLM no recibe los datos directamente, por e-
#   ficiencia y privacidad.
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    ¿Qué información le das al LLM y por qué? Pon un ejemplo
#    concreto de una pregunta que funciona gracias a algo específico
#    de tu prompt, y otro de una que falla o fallaría si quitases
#    una instrucción.
#
#   Al LLM le he dado información sobre su rol, para marcar sus conocimien-
#   tos y capacidades, objetivos para que sepa qué se busca en su interac-
#   ción, datos descriptivos del dataset, para que pueda generar un código 
#   funcional sin cargar el dataset directamente, instrucciones obligatorias 
#   sobre cómo debe actuar tanto en la generación de contenido como a nivel 
#   de control de calidad, y una guía sobre el formato que debe tomar su res-
#   puesta. 
#   
#   Las preguntas relacionadas con épocas del año funcionan gracias a la in-
#   troducción de las Seasons en el contexto analítico. Esta permite al modelo
#   filtrar por estación y mes correspondiente a cada una en su código.
#   
#   La pregunta de artista favorito no funciona sin la última línea de ROL.
#   Siempre me daba top 10, porque entendía que parte de su personalidad 
#   incluía el ofrecer mayor contexto a lo que le piden, incluso aunque esto
#   entrara en conflicto con las instrucciones obligatorias.
#
# 3. EL FLUJO COMPLETO
#    Describe paso a paso qué ocurre desde que el usuario escribe
#    una pregunta hasta que ve el gráfico en pantalla.
#
#   El flujo comienza con nuestro código enviando el System Prompt y la
#   pregunta del usuario a la API del LLM. Al recibir esto,el modelo pro-
#   cesa la pregunta y la clasifica como "grafico" o "fuera_de_alcance", 
#   devolviendo un string en formato JSON que contiene un código Python y 
#   una interpretación. Nuestro código coge entonces lo que el LLM ha de-
#   vuelto y lo ejecuta. Si ve que "tipo" está marcado como fuera de alcan-
#   ce, solo devuelve la "interpretacion"; si está marcado como "grafico" 
#   devuelve "codigo" e "interpretacion", con algunos guardarrailes en caso 
#   de que el codigo no produjera ningún gráfico.

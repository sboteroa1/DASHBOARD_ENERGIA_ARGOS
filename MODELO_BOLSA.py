import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Análisis Precio Bolsa", layout="wide", page_icon="⚡")

# Título
st.title("Análisis Precio Bolsa Nacional")
st.markdown("---")

# Función para cargar datos de precios
@st.cache_data
def load_data():
    df = pd.read_csv('PRECIO_BOLSA.csv', sep=';', encoding='utf-8-sig')
    
    # Limpiar nombres de columnas
    df.columns = df.columns.str.strip()
    
    # Convertir precio (primero quitar puntos de miles, luego coma a punto decimal)
    df['Precio Bolsa Nacional'] = (df['Precio Bolsa Nacional']
                                    .str.strip()
                                    .str.replace('.', '', regex=False)
                                    .str.replace(',', '.')
                                    .astype(float))
    
    # Convertir fecha
    df['Fecha'] = pd.to_datetime(df['Fecha'].str.strip())
    
    # Extraer hora numérica de P1-P24
    df['Hora_num'] = df['Hora'].str.extract('(\d+)').astype(int)
    
    # Agregar columnas de tiempo
    df['Año'] = df['Fecha'].dt.year
    df['Mes'] = df['Fecha'].dt.month
    df['Día'] = df['Fecha'].dt.day
    df['Mes_nombre'] = df['Fecha'].dt.strftime('%B')
    df['Día_año'] = df['Fecha'].dt.dayofyear
    
    return df

# Función para cargar datos de generación
@st.cache_data
def load_generacion():
    df_gen = pd.read_csv('GENERACION.csv', sep=';', encoding='utf-8-sig')
    
    # Limpiar nombres de columnas
    df_gen.columns = df_gen.columns.str.strip()
    
    # Convertir generación (coma a punto)
    df_gen['Suma de Generación'] = (df_gen['Suma de Generación']
                                     .str.strip()
                                     .str.replace(',', '.')
                                     .astype(float))
    
    # Convertir fecha
    df_gen['Fecha'] = pd.to_datetime(df_gen['Fecha'].str.strip())
    
    # Agregar columnas de tiempo
    df_gen['Año'] = df_gen['Fecha'].dt.year
    df_gen['Mes'] = df_gen['Fecha'].dt.month
    
    return df_gen

# Función para cargar datos de generación convencional
@st.cache_data
def load_generacion():
    df_gen = pd.read_csv('GENERACION.csv', sep=';', encoding='utf-8-sig')
    
    # Limpiar nombres de columnas
    df_gen.columns = df_gen.columns.str.strip()
    
    # Convertir generación (coma a punto)
    df_gen['Suma de Generación'] = (df_gen['Suma de Generación']
                                     .str.strip()
                                     .str.replace(',', '.')
                                     .astype(float))
    
    # Convertir fecha
    df_gen['Fecha'] = pd.to_datetime(df_gen['Fecha'].str.strip())
    
    # Agregar columnas de tiempo
    df_gen['Año'] = df_gen['Fecha'].dt.year
    df_gen['Mes'] = df_gen['Fecha'].dt.month
    
    return df_gen

# Función para cargar datos de generación alternativos
@st.cache_data
def load_generacion_alternativos():
    df_alt = pd.read_csv('GENERACION_ALTERNATIVOS.csv', sep=';', encoding='utf-8-sig')
    
    # Limpiar nombres de columnas
    df_alt.columns = df_alt.columns.str.strip()
    
    # Convertir generación (coma a punto)
    df_alt['Suma de Generación'] = (df_alt['Suma de Generación']
                                     .str.strip()
                                     .str.replace(',', '.')
                                     .astype(float))
    
    # Convertir fecha
    df_alt['Fecha'] = pd.to_datetime(df_alt['Fecha'].str.strip())
    
    # Agregar columnas de tiempo
    df_alt['Año'] = df_alt['Fecha'].dt.year
    df_alt['Mes'] = df_alt['Fecha'].dt.month
    
    # Sumar todas las fuentes alternativas por fecha
    df_alt_sum = df_alt.groupby('Fecha').agg({
        'Suma de Generación': 'sum',
        'Año': 'first',
        'Mes': 'first'
    }).reset_index()
    
    df_alt_sum['Tipo fuente'] = 'Alternativos'
    
    return df_alt_sum

# Cargar datos
df = load_data()
df_generacion = load_generacion()
df_alternativos = load_generacion_alternativos()

# ========== PANEL DE CONTROL ==========
st.sidebar.header("Panel de Control")

# 1. Selector de fechas
fecha_min = df['Fecha'].min().date()
fecha_max = df['Fecha'].max().date()

col1, col2 = st.sidebar.columns(2)
with col1:
    fecha_inicio = st.date_input("Fecha Inicio", fecha_min, min_value=fecha_min, max_value=fecha_max)
with col2:
    fecha_fin = st.date_input("Fecha Fin", fecha_max, min_value=fecha_min, max_value=fecha_max)

st.sidebar.markdown("### Parámetros Económicos")

# 2. Input de costo variable
costo_variable = st.sidebar.number_input("Costo Variable (COP/kWh)", min_value=0.0, value=150.0, step=10.0)

# Precio contrato
precio_contrato = st.sidebar.number_input("Precio Contrato (COP/kWh)", min_value=0.0, value=200.0, step=10.0)

# Capacidades
capacidad_contrato = st.sidebar.number_input("Capacidad Contrato (MW)", min_value=0.0, value=10.0, step=1.0)
capacidad_excedentes = st.sidebar.number_input("Capacidad Excedentes (MW)", min_value=0.0, value=5.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.markdown("### Opciones de Visualización")

# Checkbox para mostrar área de contribución
mostrar_area = st.sidebar.checkbox("Mostrar área contribución contrato", value=True)

# Slider de transparencia (solo si está activado)
if mostrar_area:
    transparencia = st.sidebar.slider("Transparencia del área (%)", min_value=0, max_value=100, value=20, step=5)
else:
    transparencia = 20  # valor por defecto

st.sidebar.markdown("---")

# 3. Nivel de detalle
granularidad = st.sidebar.radio("Nivel de Detalle", ["Horario", "Diario", "Mensual"])

# ========== FILTRAR DATOS ==========
df_filtrado = df[(df['Fecha'].dt.date >= fecha_inicio) & (df['Fecha'].dt.date <= fecha_fin)].copy()

# ========== CÁLCULOS DE CONTRIBUCIÓN (A NIVEL HORARIO) ==========
# Convertir MW a kW (multiplicar por 1000)
cap_contrato_kw = capacidad_contrato * 1000
cap_excedentes_kw = capacidad_excedentes * 1000

# Contribución del Contrato
df_filtrado['Cont_Contrato'] = df_filtrado.apply(
    lambda row: (precio_contrato - costo_variable) * cap_contrato_kw 
                if row['Precio Bolsa Nacional'] > costo_variable 
                else (precio_contrato - row['Precio Bolsa Nacional']) * cap_contrato_kw,
    axis=1
)

# Contribución de Excedentes
df_filtrado['Cont_Excedentes'] = df_filtrado.apply(
    lambda row: (row['Precio Bolsa Nacional'] - costo_variable) * cap_excedentes_kw 
                if row['Precio Bolsa Nacional'] > costo_variable 
                else 0,
    axis=1
)

# Marcar horas premium
df_filtrado['Es_Premium'] = df_filtrado['Precio Bolsa Nacional'] > costo_variable

# ========== AGREGACIÓN SEGÚN GRANULARIDAD ==========
if granularidad == "Diario":
    df_agg = df_filtrado.groupby(['Fecha', 'Año', 'Mes', 'Día_año']).agg({
        'Precio Bolsa Nacional': ['mean', 'min', 'max'],
        'Cont_Contrato': 'sum',
        'Cont_Excedentes': 'sum',
        'Es_Premium': 'sum'  # Cuenta horas premium por día
    }).reset_index()
    df_agg.columns = ['Fecha', 'Año', 'Mes', 'Día_año', 'Precio', 'Min', 'Max', 'Cont_Contrato', 'Cont_Excedentes', 'Horas_Premium']
    
elif granularidad == "Mensual":
    df_agg = df_filtrado.groupby(['Año', 'Mes']).agg({
        'Precio Bolsa Nacional': ['mean', 'min', 'max'],
        'Cont_Contrato': 'sum',
        'Cont_Excedentes': 'sum',
        'Es_Premium': 'sum'
    }).reset_index()
    df_agg.columns = ['Año', 'Mes', 'Precio', 'Min', 'Max', 'Cont_Contrato', 'Cont_Excedentes', 'Horas_Premium']
    df_agg['Fecha'] = pd.to_datetime(df_agg['Año'].astype(str) + '-' + df_agg['Mes'].astype(str) + '-01')
    
else:  # Horario
    df_agg = df_filtrado.copy()
    df_agg['Precio'] = df_agg['Precio Bolsa Nacional']
    df_agg['Horas_Premium'] = df_agg['Es_Premium'].astype(int)

# ========== CALCULAR MÉTRICAS GLOBALES ==========
total_cont_contrato = df_filtrado['Cont_Contrato'].sum()
total_cont_excedentes = df_filtrado['Cont_Excedentes'].sum()

# Promedio de horas premium
horas_premium_df = df_filtrado[df_filtrado['Es_Premium'] == True]
promedio_horas_premium = horas_premium_df['Precio Bolsa Nacional'].mean() if len(horas_premium_df) > 0 else 0

# Porcentaje ponderado de horas premium por día
if granularidad == "Diario":
    df_agg['Pct_Premium_Dia'] = (df_agg['Horas_Premium'] / 24) * 100
    pct_horas_premium_ponderado = df_agg['Pct_Premium_Dia'].mean()
else:
    # Calcular a nivel diario para obtener el ponderado
    df_diario_temp = df_filtrado.groupby('Fecha').agg({'Es_Premium': 'sum'}).reset_index()
    df_diario_temp['Pct_Premium_Dia'] = (df_diario_temp['Es_Premium'] / 24) * 100
    pct_horas_premium_ponderado = df_diario_temp['Pct_Premium_Dia'].mean()

# ========== KPIs PRINCIPALES ==========
st.markdown("## Indicadores Principales")

# Crear 3 columnas para los grupos
col_grupo1, col_grupo2, col_grupo3 = st.columns(3)

# GRUPO 1: Horas Premium
with col_grupo1:
    st.markdown("### Análisis Horas Premium")
    subcol1, subcol2, subcol3 = st.columns(3)
    
    # Contar horas por encima del costo variable
    if granularidad == "Horario":
        horas_encima = (df_filtrado['Precio Bolsa Nacional'] > costo_variable).sum()
        total_horas = len(df_filtrado)
        pct_encima = (horas_encima / total_horas * 100) if total_horas > 0 else 0
        
        with subcol1:
            st.metric("Horas > Costo Var.", f"{horas_encima:,}")
            st.caption(f"{pct_encima:.1f}% del total")
    else:
        registros_encima = (df_agg['Precio'] > costo_variable).sum()
        total_registros = len(df_agg)
        pct_encima = (registros_encima / total_registros * 100) if total_registros > 0 else 0
        
        with subcol1:
            st.metric(f"{granularidad}s > Costo Var.", f"{registros_encima:,}")
            st.caption(f"{pct_encima:.1f}% del total")
    
    with subcol2:
        st.metric("% Horas Premium/Día", f"{pct_horas_premium_ponderado:.1f}%")
    
    with subcol3:
        st.metric("Prom. Horas Premium", f"${promedio_horas_premium:.2f}")

# GRUPO 2: Estadísticas de Precio
with col_grupo2:
    st.markdown("### Estadísticas de Precio")
    subcol4, subcol5, subcol6 = st.columns(3)
    
    with subcol4:
        promedio = df_agg['Precio'].mean()
        st.metric("Promedio Periodo", f"${promedio:.2f}")
    
    with subcol5:
        minimo = df_agg['Min'].min() if 'Min' in df_agg.columns else df_agg['Precio'].min()
        st.metric("Mínimo Periodo", f"${minimo:.2f}")
    
    with subcol6:
        maximo = df_agg['Max'].max() if 'Max' in df_agg.columns else df_agg['Precio'].max()
        st.metric("Máximo Periodo", f"${maximo:.2f}")

# GRUPO 3: Contribuciones
with col_grupo3:
    st.markdown("### Contribuciones (Millones COP)")
    subcol7, subcol8 = st.columns(2)
    
    with subcol7:
        st.metric("Cont. Contrato", f"${total_cont_contrato/1_000_000:,.0f}M")
    
    with subcol8:
        st.metric("Cont. Excedentes", f"${total_cont_excedentes/1_000_000:,.0f}M")

st.markdown("---")

# ========== GRÁFICA DE LÍNEAS MULTI-AÑO ==========
st.markdown("## Evolución de Precios")

fig_lineas = go.Figure()

años_unicos = sorted(df_agg['Año'].unique())
colores = px.colors.qualitative.Plotly

# Convertir transparencia de porcentaje a valor 0-1
alpha = transparencia / 100

# Primero agregar las áreas sombreadas (para que queden detrás) - SOLO SI ESTÁ ACTIVADO
if mostrar_area:
    for idx, año in enumerate(años_unicos):
        df_año = df_agg[df_agg['Año'] == año].copy()
        
        # Calcular el mínimo entre costo variable y precio bolsa
        df_año['Min_CV_Bolsa'] = df_año['Precio'].apply(lambda x: min(costo_variable, x))
        
        # Calcular contribución por punto (en millones)
        df_año['Contribucion_Punto'] = (precio_contrato - df_año['Min_CV_Bolsa']) * cap_contrato_kw / 1_000_000
        
        if granularidad == "Horario":
            df_año = df_año.sort_values('Fecha')
            df_año['Indice'] = range(len(df_año))
            x_vals = df_año['Indice']
        elif granularidad == "Diario":
            x_vals = df_año['Día_año']
        else:  # Mensual
            x_vals = df_año['Mes']
        
        # Área sombreada: desde Min(CV, Bolsa) hasta Precio Contrato
        fig_lineas.add_trace(go.Scatter(
            x=x_vals,
            y=df_año['Min_CV_Bolsa'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig_lineas.add_trace(go.Scatter(
            x=x_vals,
            y=[precio_contrato] * len(x_vals),
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor=f'rgba(196, 214, 0, {alpha})',  # Color #c4d600 con transparencia
            showlegend=False,
            hoverinfo='skip',
            name=f'Contribución Contrato {año}'
        ))

# Luego agregar las líneas de precio (encima del área)
for idx, año in enumerate(años_unicos):
    df_año = df_agg[df_agg['Año'] == año].copy()
    
    # Calcular contribución para el tooltip
    df_año['Min_CV_Bolsa'] = df_año['Precio'].apply(lambda x: min(costo_variable, x))
    df_año['Contribucion_Punto'] = (precio_contrato - df_año['Min_CV_Bolsa']) * cap_contrato_kw / 1_000_000
    
    if granularidad == "Horario":
        df_año = df_año.sort_values('Fecha')
        df_año['Indice'] = range(len(df_año))
        x_vals = df_año['Indice']
        hover_template = (
            'Fecha: %{customdata[0]}<br>'
            'Precio: $%{y:.2f}<br>'
            'Contribución: $%{customdata[1]:.2f}M<extra></extra>'
        )
        customdata = list(zip(
            df_año['Fecha'].dt.strftime('%Y-%m-%d %H:00'),
            df_año['Contribucion_Punto']
        ))
    elif granularidad == "Diario":
        x_vals = df_año['Día_año']
        hover_template = (
            'Fecha: %{customdata[0]}<br>'
            'Precio: $%{y:.2f}<br>'
            'Contribución: $%{customdata[1]:.2f}M<extra></extra>'
        )
        customdata = list(zip(
            df_año['Fecha'].dt.strftime('%Y-%m-%d'),
            df_año['Contribucion_Punto']
        ))
    else:  # Mensual
        x_vals = df_año['Mes']
        hover_template = (
            'Mes: %{x}<br>'
            'Precio: $%{y:.2f}<br>'
            'Contribución: $%{customdata:.2f}M<extra></extra>'
        )
        customdata = df_año['Contribucion_Punto']
    
    fig_lineas.add_trace(go.Scatter(
        x=x_vals,
        y=df_año['Precio'],
        mode='lines',
        name=f"Precio Bolsa {año}",
        line=dict(width=2, color=colores[idx % len(colores)]),
        hovertemplate=hover_template,
        customdata=customdata
    ))

# Línea de costo variable
fig_lineas.add_hline(y=costo_variable, line_dash="dash", line_color="red", line_width=2,
                     annotation_text=f"Costo Variable: ${costo_variable:.2f}",
                     annotation_position="top right")

# Línea de precio contrato
fig_lineas.add_hline(y=precio_contrato, line_dash="dot", line_color="green", line_width=2,
                     annotation_text=f"Precio Contrato: ${precio_contrato:.2f}",
                     annotation_position="bottom right")

# Configurar eje X
if granularidad == "Mensual":
    fig_lineas.update_xaxes(
        title="Mes",
        tickmode='array',
        tickvals=list(range(1, 13)),
        ticktext=['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                  'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    )
elif granularidad == "Horario":
    fig_lineas.update_xaxes(title="Secuencia Temporal (Horas)")
else:
    fig_lineas.update_xaxes(title="Día del Año")

fig_lineas.update_layout(
    height=500,
    yaxis_title="Precio (COP/kWh)",
    hovermode='x unified',
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_lineas, use_container_width=True)

st.markdown("---")

# ========== ANÁLISIS DE GENERACIÓN ==========
# Solo mostrar si no está en modo Horario
if granularidad in ["Diario", "Mensual"]:
    st.markdown("## Análisis de Generación Eléctrica")
    
    # Filtrar datos de generación por rango de fechas
    df_gen_filtrado = df_generacion[(df_generacion['Fecha'].dt.date >= fecha_inicio) & 
                                     (df_generacion['Fecha'].dt.date <= fecha_fin)].copy()
    
    df_alt_filtrado = df_alternativos[(df_alternativos['Fecha'].dt.date >= fecha_inicio) & 
                                       (df_alternativos['Fecha'].dt.date <= fecha_fin)].copy()
    
    # Combinar datos convencionales y alternativos
    df_gen_completo = pd.concat([df_gen_filtrado, df_alt_filtrado], ignore_index=True)
    
    # Agregar según granularidad
    if granularidad == "Diario":
        df_gen_agg = df_gen_completo.groupby(['Fecha', 'Tipo fuente']).agg({
            'Suma de Generación': 'sum'
        }).reset_index()
        # Agregar columnas para alinear con df_agg de precios
        df_gen_agg['Año'] = df_gen_agg['Fecha'].dt.year
        df_gen_agg['Día_año'] = df_gen_agg['Fecha'].dt.dayofyear
    else:  # Mensual
        df_gen_agg = df_gen_completo.groupby([df_gen_completo['Fecha'].dt.to_period('M'), 'Tipo fuente']).agg({
            'Suma de Generación': 'sum'
        }).reset_index()
        df_gen_agg['Fecha'] = df_gen_agg['Fecha'].dt.to_timestamp()
        df_gen_agg['Año'] = df_gen_agg['Fecha'].dt.year
        df_gen_agg['Mes'] = df_gen_agg['Fecha'].dt.month
    
    # Pivotar para tener columnas por tipo
    df_gen_pivot = df_gen_agg.pivot(index='Fecha', columns='Tipo fuente', values='Suma de Generación').reset_index()
    df_gen_pivot = df_gen_pivot.fillna(0)
    
    # Agregar columnas de año y día para el eje X
    if granularidad == "Diario":
        df_gen_pivot['Año'] = df_gen_pivot['Fecha'].dt.year
        df_gen_pivot['Día_año'] = df_gen_pivot['Fecha'].dt.dayofyear
    else:
        df_gen_pivot['Año'] = df_gen_pivot['Fecha'].dt.year
        df_gen_pivot['Mes'] = df_gen_pivot['Fecha'].dt.month
    
    # Calcular totales y porcentajes
    total_hidraulica = df_gen_pivot['Hidráulica'].sum() if 'Hidráulica' in df_gen_pivot.columns else 0
    total_fosil = df_gen_pivot['Combustible fósil'].sum() if 'Combustible fósil' in df_gen_pivot.columns else 0
    total_alternativos = df_gen_pivot['Alternativos'].sum() if 'Alternativos' in df_gen_pivot.columns else 0
    total_general = total_hidraulica + total_fosil + total_alternativos
    
    pct_hidraulica = (total_hidraulica / total_general * 100) if total_general > 0 else 0
    pct_fosil = (total_fosil / total_general * 100) if total_general > 0 else 0
    pct_alternativos = (total_alternativos / total_general * 100) if total_general > 0 else 0
    
    # Crear gráfico de barras apiladas (ancho completo)
    fig_gen = go.Figure()
    
    # Obtener años únicos para mantener consistencia
    años_gen_unicos = sorted(df_gen_pivot['Año'].unique())
    
    for año in años_gen_unicos:
        df_año_gen = df_gen_pivot[df_gen_pivot['Año'] == año].copy()
        
        # Determinar eje X según granularidad
        if granularidad == "Diario":
            x_vals = df_año_gen['Día_año']
        else:  # Mensual
            x_vals = df_año_gen['Mes']
        
        if 'Hidráulica' in df_gen_pivot.columns:
            fig_gen.add_trace(go.Bar(
                x=x_vals,
                y=df_año_gen['Hidráulica'],
                name=f'Hidráulica {año}',
                marker_color='#4472C4',
                legendgroup=f'{año}',
                hovertemplate='Hidráulica: %{y:.2f} GWh<extra></extra>'
            ))
        
        if 'Combustible fósil' in df_gen_pivot.columns:
            fig_gen.add_trace(go.Bar(
                x=x_vals,
                y=df_año_gen['Combustible fósil'],
                name=f'Fósil {año}',
                marker_color='#ED7D31',
                legendgroup=f'{año}',
                hovertemplate='Fósil: %{y:.2f} GWh<extra></extra>'
            ))
        
        if 'Alternativos' in df_gen_pivot.columns:
            fig_gen.add_trace(go.Bar(
                x=x_vals,
                y=df_año_gen['Alternativos'],
                name=f'Alternativos {año}',
                marker_color='#70AD47',
                legendgroup=f'{año}',
                hovertemplate='Alternativos: %{y:.2f} GWh<extra></extra>'
            ))
    
    # Configurar eje X igual que en precios
    if granularidad == "Mensual":
        fig_gen.update_xaxes(
            title="Mes",
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                      'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        )
    else:  # Diario
        fig_gen.update_xaxes(title="Día del Año")
    
    fig_gen.update_layout(
        barmode='stack',
        height=500,
        yaxis_title="Generación (GWh)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_gen, use_container_width=True)
    
    # Indicadores debajo de la gráfica
    st.markdown("### Composición Generación")
    col_ind1, col_ind2, col_ind3 = st.columns(3)
    
    with col_ind1:
        st.metric("Generación Hídrica", 
                 f"{total_hidraulica:,.0f} GWh",
                 f"{pct_hidraulica:.1f}%")
    
    with col_ind2:
        st.metric("Generación Fósil", 
                 f"{total_fosil:,.0f} GWh",
                 f"{pct_fosil:.1f}%")
    
    with col_ind3:
        st.metric("Generación Alternativos", 
                 f"{total_alternativos:,.0f} GWh",
                 f"{pct_alternativos:.1f}%")
    
    st.markdown("---")

# ========== HISTOGRAMA COMPARATIVO ==========
st.markdown("## Distribución de Precios (Histogramas por Año)")

fig_hist = go.Figure()

# Crear histogramas superpuestos por año
for idx, año in enumerate(años_unicos):
    df_año = df_agg[df_agg['Año'] == año]
    
    fig_hist.add_trace(go.Histogram(
        x=df_año['Precio'],
        name=str(año),
        opacity=0.6,
        xbins=dict(size=50),  # Bins de 50 COP
        marker_color=colores[idx % len(colores)]
    ))
    
    # Calcular % por debajo del costo variable
    total = len(df_año)
    debajo = (df_año['Precio'] <= costo_variable).sum()
    pct_debajo = (debajo / total * 100) if total > 0 else 0
    
    st.write(f"**{año}:** {pct_debajo:.1f}% de los datos estuvieron por debajo del costo variable")

# Línea vertical del costo variable
fig_hist.add_vline(x=costo_variable, line_dash="dash", line_color="red", line_width=2,
                   annotation_text=f"Costo Variable: ${costo_variable:.2f}",
                   annotation_position="top right")

# Línea vertical del precio contrato
fig_hist.add_vline(x=precio_contrato, line_dash="dot", line_color="green", line_width=2,
                   annotation_text=f"Precio Contrato: ${precio_contrato:.2f}",
                   annotation_position="top left")

fig_hist.update_layout(
    barmode='overlay',
    height=400,
    xaxis_title="Precio (COP/kWh)",
    yaxis_title="Frecuencia",
    showlegend=True
)

st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")

# ========== TABLA DE ESTADÍSTICAS ==========
st.markdown("## Tabla de Estadísticas")

if granularidad == "Diario":
    tabla = df_agg[['Fecha', 'Precio', 'Min', 'Max', 'Horas_Premium', 'Cont_Contrato', 'Cont_Excedentes']].copy()
    tabla['Fecha'] = tabla['Fecha'].dt.strftime('%Y-%m-%d')
    tabla['Cont_Contrato'] = tabla['Cont_Contrato'].round(0)
    tabla['Cont_Excedentes'] = tabla['Cont_Excedentes'].round(0)
    tabla.columns = ['Fecha', 'Promedio', 'Mínimo', 'Máximo', 'Horas Premium', 'Cont. Contrato (COP)', 'Cont. Excedentes (COP)']
elif granularidad == "Mensual":
    tabla = df_agg[['Año', 'Mes', 'Precio', 'Min', 'Max', 'Horas_Premium', 'Cont_Contrato', 'Cont_Excedentes']].copy()
    tabla['Mes'] = tabla['Mes'].apply(lambda x: ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                                                   'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'][x-1])
    tabla['Cont_Contrato'] = tabla['Cont_Contrato'].round(0)
    tabla['Cont_Excedentes'] = tabla['Cont_Excedentes'].round(0)
    tabla.columns = ['Año', 'Mes', 'Promedio', 'Mínimo', 'Máximo', 'Horas Premium', 'Cont. Contrato (COP)', 'Cont. Excedentes (COP)']
else:
    tabla = df_agg[['Fecha', 'Hora', 'Precio', 'Cont_Contrato', 'Cont_Excedentes']].copy()
    tabla['Fecha'] = tabla['Fecha'].dt.strftime('%Y-%m-%d')
    tabla['Cont_Contrato'] = tabla['Cont_Contrato'].round(0)
    tabla['Cont_Excedentes'] = tabla['Cont_Excedentes'].round(0)
    tabla.columns = ['Fecha', 'Hora', 'Precio', 'Cont. Contrato (COP)', 'Cont. Excedentes (COP)']

st.dataframe(tabla, use_container_width=True, height=400)

# Botón de descarga
csv = tabla.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Descargar datos como CSV",
    data=csv,
    file_name=f'datos_bolsa_{granularidad.lower()}_{fecha_inicio}_{fecha_fin}.csv',
    mime='text/csv',
)
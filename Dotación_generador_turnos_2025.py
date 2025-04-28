import streamlit as st
import pandas as pd
import numpy as np
import math
import time
from io import BytesIO
import datetime
from pyworkforce.scheduling import MinAbsDifference

# Interfaz
st.set_page_config(page_title="Generador de Turnos 2025", layout="wide")
st.title("🛠️ Generador de Turnos 2025")

# Sidebar - Parámetros
st.sidebar.header("Parámetros de Optimización")
MAX_ITER = st.sidebar.number_input("Iteraciones metaheurística (MAX_ITER)", min_value=1, value=200, step=10)
TIME_SOLVER = st.sidebar.number_input("Tiempo por solver (segundos)", min_value=1.0, value=15.0)
SEED_START = st.sidebar.number_input("Semilla (SEED_START)", min_value=0, value=0)
PERTURB_NOISE = st.sidebar.slider("Ruido inicial (PERTURB_NOISE)", min_value=0.0, max_value=1.0, value=0.40)
MIN_REST_PCT = st.sidebar.slider("Pct mínimo descanso (MIN_REST_PCT)", min_value=0.0, max_value=1.0, value=0.0)
ANNEALING = st.sidebar.checkbox("Usar annealing", value=True)
NOISE_FINAL = st.sidebar.slider("Ruido final (NOISE_FINAL)", min_value=0.0, max_value=1.0, value=0.05)

# Funciones auxiliares
@st.cache_data
def load_data(uploaded_file):
    df_dem = pd.read_excel(uploaded_file, sheet_name=0)
    df_staff = pd.read_excel(uploaded_file, sheet_name=1)
    return df_dem, df_staff

# Distribución de turnos, configuración global
dias_semana = ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']

shifts_coverage = {
    # (aquí pegue su diccionario completo de shifts_coverage)
}
# El usuario debe rellenar shifts_coverage según su configuración

# Subida de archivo
uploaded = st.file_uploader("📂 Carga tu archivo Excel con demanda y personal", type=["xlsx"])
if not uploaded:
    st.info("Por favor, sube un archivo Excel con dos hojas: demanda (sheet1) y staff (sheet2).")
    st.stop()

# Cargar datos
df_dem, df_staff = load_data(uploaded)
required_resources = [[] for _ in range(7)]
for _, row in df_dem.iterrows():
    required_resources[int(row['Día'])-1].append(row['Suma de Agentes Requeridos Erlang'])
assert all(len(day)==24 for day in required_resources), "Cada día debe tener 24 periodos"

employees = df_staff['Nombre'].astype(str).tolist()
base_shifts = df_staff['Horario'].astype(str).tolist()

# Funciones de optimización

def adjust_required(dist):
    return [[math.ceil(req/(1-dist[d])) for req in day]
            for d, day in enumerate(required_resources)]

def greedy_day_off_assignment(n, dist):
    counts = np.zeros(7, int)
    quota = (n * dist).round().astype(int)
    result = []
    for _ in range(n):
        idx = np.argmax(quota - counts)
        result.append(idx)
        counts[idx] += 1
    return result


def mutate_dist(base, it):
    scale = PERTURB_NOISE
    if ANNEALING:
        frac = it / max(1, MAX_ITER-1)
        scale = PERTURB_NOISE*(1-frac) + NOISE_FINAL*frac
    noise = np.random.default_rng(SEED_START+it).normal(0, scale, 7)
    cand = np.clip(base*(1+noise), 1e-9, None)
    cand /= cand.sum()
    mask = cand < MIN_REST_PCT
    deficit = (MIN_REST_PCT - cand[mask]).sum()
    cand[mask] = MIN_REST_PCT
    if deficit>0:
        surplus = ~mask
        cand[surplus] -= deficit/surplus.sum()
    return cand


def coverage_manual(plan):
    diff, total = 0, sum(map(sum, required_resources))
    for d in range(7):
        for h in range(24):
            req = required_resources[d][h]
            work = sum(1 for shift, off in plan
                       if shifts_coverage.get(shift,[0]*24)[h] and off!=d)
            diff += abs(work-req)
    return (1-diff/total)*100

# Ejecución de la metaheurística
st.write("## 🚀 Iniciando optimización...")
status_text = st.empty()
best_cov, best_sol, best_dist = -1,None,None

daily_totals = [sum(d) for d in required_resources]
base_rest = np.array([1/max(1,x) for x in daily_totals])
base_rest /= base_rest.sum()

for it in range(int(MAX_ITER)):
    dist = mutate_dist(base_rest, it) if it else base_rest.copy()
    solver = MinAbsDifference(
        num_days=7, periods=24,
        shifts_coverage=shifts_coverage,
        required_resources=adjust_required(dist),
        max_period_concurrency=5000,
        max_shift_concurrency=300,
        max_search_time=TIME_SOLVER,
        num_search_workers=8,
        random_seed=int(SEED_START+it)
    )
    sol = solver.solve()
    cov = coverage_manual(list(zip(base_shifts, greedy_day_off_assignment(len(base_shifts), dist))))
    status_text.text(f"Iter {it+1}/{int(MAX_ITER)} – cobertura aproximada: {cov:.2f}%")
    if cov>best_cov:
        best_cov, best_sol, best_dist = cov, sol, dist.copy()

st.success(f"✅ Optimización completa. Mejor cobertura: {best_cov:.2f}%")

# Construcción del plan definitivo
days_off = greedy_day_off_assignment(len(employees), best_dist)
plan = []
for i, emp in enumerate(employees):
    best_pat = base_shifts[i] + '_1'
    plan.append((best_pat, days_off[i]))

# Generación de resultados
suf = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Recursos crudos
if best_sol:
    df_raw = pd.DataFrame(best_sol['resources_shifts'])
else:
    df_raw = pd.DataFrame()

# Plan de contratación
summary = pd.DataFrame({
    'Nombre': employees,
    'Horario': [p for p,_ in plan],
    'Día Desc.': [dias_semana[off] for _,off in plan]
})
summary['Tipo con.'] = summary['Horario'].apply(lambda s: '8h' if s.startswith('FT') else '4h')
summary['Personal a Contratar'] = 1
to_con = summary.groupby(['Horario','Tipo con.','Día Desc.'], as_index=False).sum()

# Detalle programación diaria
rows = []
for i, emp in enumerate(employees):
    pat, off = plan[i]
    for d in range(7):
        rows.append({
            'Nombre': emp,
            'Día': dias_semana[d],
            'Horario': 'Descanso' if d==off else pat
        })

df_detail = pd.DataFrame(rows)

# Descarga de archivos
st.write("### 📥 Descarga de resultados")

# Función para creación de botón de descarga
def make_download(df, sheet_name, filename):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    buf.seek(0)
    st.download_button(
        label=f"Descargar {filename}",
        data=buf,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

make_download(df_raw, 'Recursos', f'Result_{suf}.xlsx')
make_download(to_con, 'Contratacion', f'Plan_Contratacion_{suf}.xlsx')
make_download(df_detail, 'Detalle', f'Detalle_Programacion_{suf}.xlsx')

# Verificación cobertura
diff_rows = []
for d in range(7):
    for h in range(24):
        req = required_resources[d][h]
        work = sum(1 for pat,off in plan if off!=d and shifts_coverage.get(pat,[0]*24)[h])
        diff_rows.append({
            'Día Semana': dias_semana[d],
            'Hora': f"{h:02d}:00",
            'Requeridos': req,
            'Asignados': work,
            'Diferencia': work-req
        })

df_cov = pd.DataFrame(diff_rows)
make_download(df_cov, 'Cobertura', f'Verificacion_Cobertura_{suf}.xlsx')

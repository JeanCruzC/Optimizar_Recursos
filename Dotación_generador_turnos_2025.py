import streamlit as st
import pandas as pd
import numpy as np
import math
import time
from pyworkforce.scheduling import MinAbsDifference
import datetime
from io import BytesIO

st.set_page_config(page_title="Generador de Turnos 2025", layout="wide")
st.title("🛠️ Generador de Turnos 2025")

# ——— Sidebar: parámetros ———————————————————————————
st.sidebar.header("Parámetros de Optimización")
MAX_ITER      = st.sidebar.number_input("Iteraciones (MAX_ITER)",    min_value=1,   value=200,  step=10)
TIME_SOLVER   = st.sidebar.number_input("Tiempo por solver (seg)",  min_value=1.0, value=15.0, step=1.0)
SEED_START    = st.sidebar.number_input("Semilla (SEED_START)",     min_value=0,   value=0,    step=1)
PERTURB_NOISE = st.sidebar.slider("Ruido inicial (PERTURB_NOISE)", 0.0, 1.0, 0.40)
MIN_REST_PCT  = st.sidebar.slider("Pct mínimo descanso (MIN_REST_PCT)", 0.0, 1.0, 0.00)
ANNEALING     = st.sidebar.checkbox("Usar annealing", True)
NOISE_FINAL   = st.sidebar.slider("Ruido final (NOISE_FINAL)",        0.0, 1.0, 0.05)

# ——— Carga de datos ———————————————————————————————
uploaded = st.file_uploader("📂 Sube tu Excel (hoja1: demanda, hoja2: staff)", type=["xlsx"])
if not uploaded:
    st.info("Por favor, sube un archivo .xlsx con dos hojas (demanda y staff).")
    st.stop()

# lee directamente desde el buffer de Streamlit
df_dem   = pd.read_excel(uploaded, sheet_name=0)
df_staff = pd.read_excel(uploaded, sheet_name=1)

# reconstruye estructura original
dias_semana = ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']
required_resources = [[] for _ in range(7)]
for _, r in df_dem.iterrows():
    required_resources[int(r['Día'])-1].append(r['Suma de Agentes Requeridos Erlang'])
assert all(len(d)==24 for d in required_resources), "Demanda debe tener 24 periodos por día"

employees   = df_staff['Nombre'].astype(str).tolist()
base_shifts = df_staff['Horario'].astype(str).tolist()

# 2. DEFINICIÓN DE TURNOS ---------------------------------------
# 2. DEFINICIÓN DE TURNOS ---------------------------------------
shifts_coverage = {
    # ----------------------------------------------------------
    # TURNOS FULL‑TIME 8H
    # ----------------------------------------------------------
    "FT_00:00_1":[1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    "FT_00:00_2":[1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    "FT_00:00_3":[1,1,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    "FT_01:00_1":[0,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    "FT_01:00_2":[0,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    "FT_01:00_3":[0,1,1,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    "FT_02:00_1":[0,0,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    "FT_02:00_2":[0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    "FT_02:00_3":[0,0,1,1,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    "FT_03:00_1":[0,0,0,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
    "FT_03:00_2":[0,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
    "FT_03:00_3":[0,0,0,1,1,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
    "FT_04:00_1":[0,0,0,0,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0],
    "FT_04:00_2":[0,0,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0],
    "FT_04:00_3":[0,0,0,0,1,1,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0],
    "FT_05:00_1":[0,0,0,0,0,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0],
    "FT_05:00_2":[0,0,0,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0],
    "FT_05:00_3":[0,0,0,0,0,1,1,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0],
    "FT_06:00_1":[0,0,0,0,0,0,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
    "FT_06:00_2":[0,0,0,0,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0],
    "FT_06:00_3":[0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0],
    "FT_07:00_1":[0,0,0,0,0,0,0,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0],
    "FT_07:00_2":[0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0],
    "FT_07:00_3":[0,0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0],
    "FT_08:00_1":[0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0],
    "FT_08:00_2":[0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,0],
    "FT_08:00_3":[0,0,0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,0,0,0,0,0,0,0],
    "FT_09:00_1":[0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0],
    "FT_09:00_2":[0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0],
    "FT_09:00_3":[0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,0,0,0,0,0,0],
    "FT_10:00_1":[0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,1,1,0,0,0,0,0],
    "FT_10:00_2":[0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,0],
    "FT_10:00_3":[0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,0,0,0,0,0],
    "FT_11:00_1":[0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,1,1,0,0,0,0],
    "FT_11:00_2":[0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0],
    "FT_11:00_3":[0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,0,0,0,0],
    "FT_12:00_1":[0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,1,1,0,0,0],
    "FT_12:00_2":[0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,1,0,0,0],
    "FT_12:00_3":[0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,0,0,0],
    "FT_13:00_1":[0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,1,1,0,0],
    "FT_13:00_2":[0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,1,0,0],
    "FT_13:00_3":[0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,0,0],
    "FT_14:00_1":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,1,1,0],
    "FT_14:00_2":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,1,0],
    "FT_14:00_3":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,0],
    "FT_15:00_1":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,1,1],
    "FT_15:00_2":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,1],
    "FT_15:00_3":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,1,1,1],
    "FT_16:00_1":[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,1],
    "FT_16:00_2":[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1],
    "FT_16:00_3":[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,1,1],
    "FT_17:00_1":[1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1],
    "FT_17:00_2":[1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1],
    "FT_17:00_3":[1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,1],
    "FT_18:00_1":[1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1],
    "FT_18:00_2":[1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1],
    "FT_18:00_3":[1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1],
    "FT_19:00_1":[1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1],
    "FT_19:00_2":[1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1],
    "FT_19:00_3":[0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1],
    "FT_20:00_1":[1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1],
    "FT_20:00_2":[0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1],
    "FT_20:00_3":[1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1],
    "FT_21:00_1":[0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1],
    "FT_21:00_2":[1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1],
    "FT_21:00_3":[1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1],
    "FT_22:00_1":[1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    "FT_22:00_2":[1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    "FT_22:00_3":[1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    "FT_23:00_1":[1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    "FT_23:00_2":[1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    "FT_23:00_3":[1,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],

    # ----------------------------------------------------------
    # TURNOS PART‑TIME 4H
    # ----------------------------------------------------------
    "00_4":[1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    "01_4":[0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    "02_4":[0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    "03_4":[0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    "04_4":[0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    "05_4":[0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    "06_4":[0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    "07_4":[0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    "08_4":[0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
    "09_4":[0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0],
    "10_4":[0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0],
    "11_4":[0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0],
    "12_4":[0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0],
    "13_4":[0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0],
    "14_4":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0],
    "15_4":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0],
    "16_4":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0],
    "17_4":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0],
    "18_4":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0],
    "19_4":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0],
    "20_4":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1],
    "21_4":[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1],
    "22_4":[1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    "23_4":[1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
}

# ——— Funciones auxiliares ——————————————————————————
def adjust_required(dist):
    return [[math.ceil(req/(1-dist[d])) for req in day]
            for d, day in enumerate(required_resources)]

def greedy_day_off_assignment(n, dist):
    counts = np.zeros(7, int)
    quota  = (n * dist).round().astype(int)
    result = []
    for _ in range(n):
        idx = np.argmax(quota - counts)
        result.append(idx)
        counts[idx] += 1
    return result

def mutate_dist(base, it):
    scale = PERTURB_NOISE
    if ANNEALING:
        frac = it/max(1,MAX_ITER-1)
        scale = PERTURB_NOISE*(1-frac) + NOISE_FINAL*frac
    noise = np.random.default_rng(SEED_START+it).normal(0, scale, 7)
    cand = np.clip(base*(1+noise), 1e-9, None)
    cand /= cand.sum()
    mask = cand<MIN_REST_PCT
    deficit = (MIN_REST_PCT-cand[mask]).sum()
    cand[mask] = MIN_REST_PCT
    if deficit>0:
        surplus = ~mask
        cand[surplus] -= deficit/surplus.sum()
    return cand

def coverage_pct(sol, dist):
    if sol.get('status') not in ('OPTIMAL','FEASIBLE'):
        return 0.0
    offs = greedy_day_off_assignment(len(shifts_coverage), dist)
    day_map = {s:dias_semana[d] for s,d in zip(shifts_coverage, offs)}
    diff, total = 0, sum(map(sum, required_resources))
    for d in range(7):
        for h in range(24):
            req = required_resources[d][h]
            work = sum(
                row.get('resources',1)
                for row in sol.get('resources_shifts',[])
                if row['day']==d
                and shifts_coverage[row['shift']][h]
                and day_map[row['shift']]!=dias_semana[d]
            )
            diff += abs(work-req)
    return (1-diff/total)*100

def coverage_manual(plan):
    diff, total = 0, sum(map(sum, required_resources))
    for d in range(7):
        for h in range(24):
            req = required_resources[d][h]
            work = sum(
                1 for shift,off in plan
                if shifts_coverage.get(shift,[0]*24)[h] and off!=d
            )
            diff += abs(work-req)
    return (1-diff/total)*100

# ——— Botón de ejecución ———————————————————————————
if st.button("🚀 Ejecutar Optimización"):
    progress = st.empty()
    best_cov, best_sol, best_dist = -1, None, None

    daily_totals = [sum(d) for d in required_resources]
    base_rest    = np.array([1/max(1,x) for x in daily_totals])
    base_rest   /= base_rest.sum()

    for it in range(int(MAX_ITER)):
        start = time.time()
        dist  = mutate_dist(base_rest, it) if it else base_rest.copy()

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
        cov = coverage_pct(sol, dist)

        progress.text(f"Iter {it+1}/{int(MAX_ITER)} — cobertura: {cov:.2f}%  (t={time.time()-start:.1f}s)")
        if cov > best_cov:
            best_cov, best_sol, best_dist = cov, sol, dist.copy()

    st.success(f"✅ Optimización completa. Mejor cobertura: {best_cov:.2f}%")

    # 6. Asignación greedy final
    days_off = greedy_day_off_assignment(len(employees), best_dist)
    plan = []
    for i, emp in enumerate(employees):
        best_cov2, best_pat = -1, None
        for suf in [1,2,3]:
            p = f"{base_shifts[i]}_{suf}"
            cov2 = coverage_manual(plan+[(p, days_off[i])])
            if cov2 > best_cov2:
                best_cov2, best_pat = cov2, p
        plan.append((best_pat, days_off[i]))

    # 7. Preparar buffers de descarga en sesión
    suf = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 7.1 Raw
    df_raw = pd.DataFrame(best_sol.get('resources_shifts',[]))
    buf1 = BytesIO()
    with pd.ExcelWriter(buf1, engine="openpyxl") as w:
        df_raw.to_excel(w, sheet_name="Raw", index=False)
    buf1.seek(0)
    st.session_state["down_raw"] = (buf1, f"Result_{suf}.xlsx")
    # 7.2 Contratación
    summary = pd.DataFrame({
        'Nombre': employees,
        'Horario': [p for p,_ in plan],
        'Día Desc.': [dias_semana[off] for _,off in plan]
    })
    summary['Tipo con.'] = summary['Horario'].apply(lambda s: '8h' if s.startswith('FT') else '4h')
    summary['Personal a Contratar'] = 1
    plan_con = summary.groupby(['Horario','Tipo con.','Día Desc.'], as_index=False).sum()
    plan_con['Refrig'] = plan_con['Horario'].apply(lambda s: f"Refrigerio {s.split('_')[-1]}" if s.startswith('FT') else '-')
    buf2 = BytesIO()
    with pd.ExcelWriter(buf2, engine="openpyxl") as w2:
        plan_con.to_excel(w2, sheet_name="Contratacion", index=False)
    buf2.seek(0)
    st.session_state["down_con"] = (buf2, f"Plan_Contratacion_{suf}.xlsx")
    # 7.3 Detalle diaria
    rows = []
    for i, emp in enumerate(employees):
        pat, off = plan[i]
        for d in range(7):
            rows.append({
                'Nombre': emp,
                'Día': dias_semana[d],
                'Horario': 'Descanso' if d==off else pat,
                'Refrig': '-' if d==off else (f"Refrigerio {pat.split('_')[-1]}" if pat.startswith('FT') else '-')
            })
    df_det = pd.DataFrame(rows)
    buf3 = BytesIO()
    with pd.ExcelWriter(buf3, engine="openpyxl") as w3:
        df_det.to_excel(w3, sheet_name="Detalle", index=False)
    buf3.seek(0)
    st.session_state["down_det"] = (buf3, f"Detalle_Programacion_{suf}.xlsx")
    # 7.4 Cobertura
    cov_rows = []
    for d, dia in enumerate(dias_semana):
        for h in range(24):
            req = required_resources[d][h]
            work = sum(
                1 for pat,off in plan
                if off!=d and shifts_coverage.get(pat,[0]*24)[h]
            )
            cov_rows.append({
                'Día Semana': dia,
                'Hora': f"{h:02d}:00",
                'Requeridos': req,
                'Asignados': work,
                'Diferencia': work-req
            })
    df_cov = pd.DataFrame(cov_rows)
    buf4 = BytesIO()
    with pd.ExcelWriter(buf4, engine="openpyxl") as w4:
        df_cov.to_excel(w4, sheet_name="Cobertura", index=False)
    buf4.seek(0)
    st.session_state["down_cov"] = (buf4, f"Verificacion_Cobertura_{suf}.xlsx")

# ——— Botones de descarga persistentes —————————————————————
if "down_raw" in st.session_state:
    buf, fname = st.session_state["down_raw"]
    st.download_button("📥 Descargar Resultados crudos", buf, file_name=fname,
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if "down_con" in st.session_state:
    buf, fname = st.session_state["down_con"]
    st.download_button("📥 Descargar Plan de Contratación", buf, file_name=fname,
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if "down_det" in st.session_state:
    buf, fname = st.session_state["down_det"]
    st.download_button("📥 Descargar Detalle de Programación", buf, file_name=fname,
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if "down_cov" in st.session_state:
    buf, fname = st.session_state["down_cov"]
    st.download_button("📥 Descargar Verificación de Cobertura", buf, file_name=fname,
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

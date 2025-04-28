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

# ——— Definición de turnos (copiar tu dict completo) ——————————
shifts_coverage = {
    # ... tu diccionario completo de shifts_coverage ...
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
    # debug: mostrar estado de solución
    st.write(f"🐞 coverage_pct: status={sol.get('status')}, len(resources_shifts)={len(sol.get('resources_shifts',[]))}")
    if sol.get('status') not in ('OPTIMAL','FEASIBLE'):
        return 0.0
    offs = greedy_day_off_assignment(len(shifts_coverage), dist)
    day_map = {s:dias_semana[d] for s,d in zip(shifts_coverage, offs)}
    diff, total = 0, sum(map(sum, required_resources))
    st.write(f"🐞 coverage_pct: total_demand={total}, dist={dist[:3]}...")
    for d, day in enumerate(dias_semana):
        for h in range(24):
            req = required_resources[d][h]
            work = 0
            for row in sol['resources_shifts']:
                if (row['day']==d 
                   and shifts_coverage[row['shift']][h]
                   and day_map[row['shift']]!=day):
                    work += row.get('resources',1)
            diff += abs(work-req)
    coverage = (1-diff/total)*100
    st.write(f"🐞 coverage_pct: diff={diff}, coverage={coverage:.2f}%")
    return coverage


def coverage_manual(plan):
    diff, total = 0, sum(map(sum, required_resources))
    for d in range(7):
        for h in range(24):
            req = required_resources[d][h]
            work = sum(1 for shift,off in plan
                       if shifts_coverage.get(shift,[0]*24)[h] and off!=d)
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
        st.write(f"🐞 Iter {it+1}: dist sample={dist[:3]}...")

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
        st.write(f"🐞 Iter {it+1}: sol.status={sol.get('status')}")

        cov = coverage_pct(sol, dist)
        progress.text(f"Iter {it+1}/{int(MAX_ITER)} — cobertura: {cov:.2f}%  (t={time.time()-start:.1f}s)")
        st.write(f"🐞 Iter {it+1}: coverage returned={cov:.2f}%")

        if cov > best_cov:
            best_cov, best_sol, best_dist = cov, sol, dist.copy()
            st.write(f"🐞 Nuevo mejor: {best_cov:.2f}% en iter {it+1}")

    st.success(f"✅ Optimización completa. Mejor cobertura: {best_cov:.2f}%")

    # … resto del código de asignación y descarga igual que antes …

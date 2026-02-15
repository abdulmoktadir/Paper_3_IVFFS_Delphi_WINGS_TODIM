import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------
# Custom CSS for better appearance
# ------------------------------
st.set_page_config(
    page_title="IVFFS Decision Toolkit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS
st.markdown("""
<style>
    /* Main container */
    .main > div {
        padding: 0 2rem;
    }
    /* Headers */
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Segoe UI', sans-serif;
    }
    /* Sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    /* Buttons */
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    /* Data editor */
    .stDataFrame {
        border: 1px solid #ddd;
        border-radius: 8px;
        overflow: hidden;
    }
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f1f3f4;
        border-radius: 8px;
        font-weight: 600;
    }
    /* Metrics */
    .css-1xarl3l {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    /* Footer */
    .footer {
        text-align: center;
        color: #7f8c8d;
        padding: 2rem 0;
        font-size: 0.9rem;
        border-top: 1px solid #ecf0f1;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

EPS = 1e-12

# =========================================================
# IVFFS REPRESENTATION
# =========================================================

def clamp01(x: float) -> float:
    return float(min(1.0 - EPS, max(EPS, x)))

def make_ivffs(muL, muU, nuL, nuU):
    muL, muU, nuL, nuU = map(float, (muL, muU, nuL, nuU))
    muL, muU = min(muL, muU), max(muL, muU)
    nuL, nuU = min(nuL, nuU), max(nuL, nuU)
    return (clamp01(muL), clamp01(muU), clamp01(nuL), clamp01(nuU))

def format_ivffs(v):
    muL, muU, nuL, nuU = v
    return f"([{muL:.3f},{muU:.3f}],[{nuL:.3f},{nuU:.3f}])"  # shorter precision for display

# =========================================================
# IVFFDWA AGGREGATION
# =========================================================

def _safe_pow(x, p):
    return float(x) ** float(p)

def agg_membership_bound(x_list, w_list, alpha, power=3.0):
    alpha = max(float(alpha), EPS)
    s = 0.0
    for x, w in zip(x_list, w_list):
        x = clamp01(float(x))
        w = float(w)
        xp = _safe_pow(x, power)
        frac = xp / max(EPS, (1.0 - xp))
        s += w * _safe_pow(frac, alpha)
    inner = _safe_pow(s, 1.0 / alpha)
    val = 1.0 - (1.0 / (1.0 + inner))
    return _safe_pow(val, 1.0 / power)

def agg_nonmembership_bound(y_list, w_list, alpha, power=3.0):
    alpha = max(float(alpha), EPS)
    s = 0.0
    for y, w in zip(y_list, w_list):
        y = clamp01(float(y))
        w = float(w)
        yp = _safe_pow(y, power)
        frac = (1.0 - yp) / max(EPS, yp)
        s += w * _safe_pow(frac, alpha)
    inner = _safe_pow(s, 1.0 / alpha)
    denom = _safe_pow((1.0 + inner), 1.0 / power)
    return 1.0 / max(EPS, denom)

def ivffdwa_aggregate(ivffs_list, w_list, alpha):
    muL_list = [v[0] for v in ivffs_list]
    muU_list = [v[1] for v in ivffs_list]
    nuL_list = [v[2] for v in ivffs_list]
    nuU_list = [v[3] for v in ivffs_list]

    a = agg_membership_bound(muL_list, w_list, alpha, power=3.0)
    b = agg_membership_bound(muU_list, w_list, alpha, power=3.0)
    c = agg_nonmembership_bound(nuL_list, w_list, alpha, power=3.0)
    d = agg_nonmembership_bound(nuU_list, w_list, alpha, power=3.0)

    return make_ivffs(a, b, c, d)

# =========================================================
# IVFFS SCORE FUNCTION Ψ
# =========================================================
def ivffs_score(v):
    muL, muU, nuL, nuU = v
    return 0.5 * (0.5 * (muL**3 + muU**3 - nuL**3 - nuU**3) + 1.0)

# =========================================================
# LINGUISTIC SCALES
# =========================================================

# Strength (WINGS)
WINGS_STRENGTH = {
    "VLI": make_ivffs(0.10, 0.20, 0.80, 0.90),
    "LI":  make_ivffs(0.20, 0.50, 0.70, 0.80),
    "MI":  make_ivffs(0.50, 0.70, 0.50, 0.70),
    "HI":  make_ivffs(0.70, 0.80, 0.20, 0.50),
    "VHI": make_ivffs(0.80, 0.90, 0.10, 0.20),
}
WINGS_STRENGTH_FULL = {
    "VLI": "Very Low Important",
    "LI":  "Low Important",
    "MI":  "Medium Important",
    "HI":  "High Important",
    "VHI": "Very High Important",
}

# Influence (WINGS)
WINGS_INFLUENCE = {
    "ELI": make_ivffs(0.05, 0.15, 0.85, 0.95),
    "VLI": make_ivffs(0.15, 0.25, 0.75, 0.85),
    "LI":  make_ivffs(0.25, 0.35, 0.65, 0.75),
    "MI":  make_ivffs(0.50, 0.50, 0.50, 0.50),
    "HI":  make_ivffs(0.65, 0.75, 0.25, 0.35),
    "VHI": make_ivffs(0.75, 0.85, 0.15, 0.25),
    "EHI": make_ivffs(0.85, 0.95, 0.05, 0.15),
}
WINGS_INFLUENCE_FULL = {
    "ELI": "Extremely Low Influence",
    "VLI": "Very Low Influence",
    "LI":  "Low Influence",
    "MI":  "Medium Influence",
    "HI":  "High Influence",
    "VHI": "Very High Influence",
    "EHI": "Extremely High Influence",
}

# TODIM linguistic
TODIM_LINGUISTIC = {
    "VP": make_ivffs(0.10, 0.15, 0.90, 0.95),
    "P":  make_ivffs(0.20, 0.25, 0.80, 0.85),
    "MP": make_ivffs(0.30, 0.35, 0.70, 0.75),
    "F":  make_ivffs(0.50, 0.55, 0.40, 0.45),
    "MG": make_ivffs(0.70, 0.75, 0.30, 0.35),
    "G":  make_ivffs(0.80, 0.85, 0.20, 0.25),
    "VG": make_ivffs(0.90, 0.95, 0.10, 0.15),
}
TODIM_FULL = {
    "VP": "Very Poor",
    "P":  "Poor",
    "MP": "Medium Poor",
    "F":  "Fair",
    "MG": "Medium Good",
    "G":  "Good",
    "VG": "Very Good",
}

# =========================================================
# Helpers
# =========================================================

def parse_csv_names(s):
    return [x.strip() for x in (s or "").split(",") if x.strip()]

def normalize_ivffs_todim(v, crit_type: str):
    crit_type = (crit_type or "").strip().lower()
    if crit_type.startswith("c"):  # cost
        muL, muU, nuL, nuU = v
        return make_ivffs(nuL, nuU, muL, muU)
    return v

def round_df(df: pd.DataFrame, d=6):
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(d)
    return out

def sync_strength_df(old_df: pd.DataFrame, comps: list[str]):
    old_df = old_df.copy()
    if "Component" not in old_df.columns:
        old_df["Component"] = old_df.index.astype(str)
    if "Strength" not in old_df.columns:
        old_df["Strength"] = "MI"
    lookup = dict(zip(old_df["Component"].astype(str), old_df["Strength"].astype(str)))
    new_strength = [lookup.get(c, "MI") for c in comps]
    return pd.DataFrame({"Component": comps, "Strength": new_strength})

def sync_influence_df(old_df: pd.DataFrame, comps: list[str]):
    old_df = old_df.copy()
    old_df.index = old_df.index.astype(str)
    old_df.columns = old_df.columns.astype(str)
    new_df = pd.DataFrame("ELI", index=comps, columns=comps)
    common_r = [c for c in comps if c in old_df.index]
    common_c = [c for c in comps if c in old_df.columns]
    if common_r and common_c:
        new_df.loc[common_r, common_c] = old_df.loc[common_r, common_c].values
    for i in range(len(comps)):
        new_df.iloc[i, i] = "—"
    return new_df

# =========================================================
# WINGS MODULE
# =========================================================
def ivffs_wings_module():
    st.header("📌 IVFFS‑WINGS: Influence Analysis")
    st.markdown("Assess component strength and influence using interval-valued fermatean fuzzy sets.")

    with st.expander("ℹ️ Linguistic scales used", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Strength (Table A1)**")
            df_str = pd.DataFrame([{"Code": k, "Meaning": WINGS_STRENGTH_FULL[k], "IVFFS": format_ivffs(v)}
                                    for k, v in WINGS_STRENGTH.items()])
            st.dataframe(df_str, hide_index=True, use_container_width=True)
        with col2:
            st.markdown("**Influence (Table A4)**")
            df_inf = pd.DataFrame([{"Code": k, "Meaning": WINGS_INFLUENCE_FULL[k], "IVFFS": format_ivffs(v)}
                                    for k, v in WINGS_INFLUENCE.items()])
            st.dataframe(df_inf, hide_index=True, use_container_width=True)

    # Step 1: Basic parameters
    with st.container(border=True):
        st.subheader("Step 1: Define components and experts")
        c1, c2, c3 = st.columns(3)
        n = c1.number_input("Number of components", min_value=2, max_value=30, value=5, step=1, key="w_n")
        k = c2.number_input("Number of experts", min_value=1, max_value=20, value=4, step=1, key="w_k")
        alpha = c3.number_input("Dombi α", min_value=0.01, max_value=50.0, value=0.5, step=0.01, key="w_alpha")

        comp_names_in = st.text_input("Component names (comma‑separated)", value="C1,C2,C3,C4,C5", key="w_comps_in")
        comps = parse_csv_names(comp_names_in)
        if len(comps) != n:
            comps = [f"C{i+1}" for i in range(n)]
            st.info(f"Using default component names: {', '.join(comps)}")

        st.markdown("**Expert weights (must sum to 1.0)**")
        if k == 1:
            w_exp = [1.0]
            st.success("Single expert → weight = 1.0")
        else:
            cols = st.columns(k)
            w_exp = []
            total = 0.0
            for i in range(k):
                with cols[i]:
                    w = st.number_input(f"E{i+1}", min_value=0.0, max_value=1.0,
                                         value=round(1/k, 6), step=0.00001, format="%.6f",
                                         key=f"w_exp_{i}")
                    w_exp.append(w)
                    total += w
            if not np.isclose(total, 1.0):
                st.error(f"❌ Expert weights must sum to 1.0 (current sum = {total:.6f})")
                st.progress(min(total, 1.0))
                return
            else:
                st.success(f"✅ Sum = {total:.6f}")

    # Session state initialisation
    sk_strength = "w_strength_terms"
    sk_infl = "w_infl_terms"
    if sk_strength not in st.session_state or len(st.session_state[sk_strength]) != k:
        st.session_state[sk_strength] = [pd.DataFrame({"Component": comps, "Strength": ["MI"]*n}) for _ in range(k)]
    if sk_infl not in st.session_state or len(st.session_state[sk_infl]) != k:
        st.session_state[sk_infl] = [pd.DataFrame("ELI", index=comps, columns=comps) for _ in range(k)]
        for e in range(k):
            for i in range(n):
                st.session_state[sk_infl][e].iloc[i, i] = "—"

    # Sync with current comps
    for e in range(k):
        st.session_state[sk_strength][e] = sync_strength_df(st.session_state[sk_strength][e], comps)
        st.session_state[sk_infl][e] = sync_influence_df(st.session_state[sk_infl][e], comps)

    # Step 2: Expert inputs
    with st.container(border=True):
        st.subheader("Step 2: Enter expert evaluations")
        tabs = st.tabs([f"👤 Expert {i+1}" for i in range(k)])
        for e, tab in enumerate(tabs):
            with tab:
                st.markdown("##### (A) Strength of components")
                dfS = st.data_editor(
                    st.session_state[sk_strength][e],
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Component": st.column_config.TextColumn("Component", disabled=True),
                        "Strength": st.column_config.SelectboxColumn("Strength", options=list(WINGS_STRENGTH.keys()))
                    },
                    key=f"w_strength_editor_{e}"
                )
                st.session_state[sk_strength][e] = sync_strength_df(dfS, comps)

                st.markdown("##### (B) Influence matrix (row → column)")
                dfI = st.session_state[sk_infl][e]
                options = ["—"] + list(WINGS_INFLUENCE.keys())
                col_cfg = {c: st.column_config.SelectboxColumn(c, options=options) for c in comps}
                editedI = st.data_editor(
                    dfI,
                    use_container_width=True,
                    column_config=col_cfg,
                    key=f"w_infl_editor_{e}"
                )
                # enforce diagonal
                for i in range(n):
                    editedI.iloc[i, i] = "—"
                st.session_state[sk_infl][e] = sync_influence_df(editedI, comps)

    # Run button
    if st.button("🚀 Run IVFFS‑WINGS Analysis", type="primary", use_container_width=True, key="w_run_btn"):
        with st.spinner("Computing aggregated SIDRM and results..."):
            # Build SIDRM per expert
            expert_sidrm = []
            for e in range(k):
                strength_terms = list(st.session_state[sk_strength][e]["Strength"])
                infl_terms = st.session_state[sk_infl][e]
                mat = [[None]*n for _ in range(n)]
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            mat[i][j] = WINGS_STRENGTH[strength_terms[i]]
                        else:
                            term = infl_terms.iloc[i, j]
                            if term == "—":
                                term = "ELI"
                            mat[i][j] = WINGS_INFLUENCE[term]
                expert_sidrm.append(mat)

            # Aggregate
            agg = [[None]*n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    cell_list = [expert_sidrm[e][i][j] for e in range(k)]
                    agg[i][j] = ivffdwa_aggregate(cell_list, w_exp, alpha)

            # Score matrix C
            C = np.array([[ivffs_score(agg[i][j]) for j in range(n)] for i in range(n)], dtype=float)
            sC = float(C.sum())
            if abs(sC) < EPS:
                st.error("Normalization failed: sum of scores is zero.")
                return
            N = C / sC
            I = np.eye(n)
            T = N @ np.linalg.pinv(I - N)

            # Compute metrics
            TI = T.sum(axis=1)
            TR = T.sum(axis=0)
            ENG = TI + TR
            ROLE = TI - TR
            EV = np.sqrt(ENG**2 + ROLE**2)
            W = EV / max(EPS, float(EV.sum()))
            comp_type = np.where(ROLE >= 0, "Cause", "Effect")

            # Display results
            st.subheader("📊 Results")
            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.metric("Number of components", n)
            with col_res2:
                st.metric("Number of experts", k)
            with col_res3:
                st.metric("Dombi α", f"{alpha:.3f}")

            # Main results table
            out = pd.DataFrame({
                "Component": comps,
                "TI (Influence from)": TI,
                "TR (Influence on)": TR,
                "Engagement (TI+TR)": ENG,
                "Role (TI-TR)": ROLE,
                "Type": comp_type,
                "Weight": W
            }).round(6)
            st.dataframe(out, hide_index=True, use_container_width=True)

            # Cause-effect plot
            fig = px.scatter(
                out, x="Engagement", y="Role", text="Component",
                color="Type", color_discrete_map={"Cause": "#2E86C1", "Effect": "#E74C3C"},
                title="Cause–Effect Diagram",
                labels={"Engagement": "Engagement (TI+TR)", "Role": "Role (TI−TR)"}
            )
            fig.update_traces(textposition="top center", marker=dict(size=12))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)

            # Optional: show intermediate matrices in expanders
            with st.expander("📋 View intermediate matrices"):
                st.markdown("**Aggregated IVFFS‑SIDRM**")
                st.dataframe(pd.DataFrame([[format_ivffs(agg[i][j]) for j in range(n)] for i in range(n)],
                                          index=comps, columns=comps), use_container_width=True)
                st.markdown("**Score Matrix C (Ψ)**")
                st.dataframe(pd.DataFrame(C, index=comps, columns=comps).round(6), use_container_width=True)
                st.markdown("**Normalized Matrix N**")
                st.dataframe(pd.DataFrame(N, index=comps, columns=comps).round(6), use_container_width=True)
                st.markdown("**Total Relation Matrix T**")
                st.dataframe(pd.DataFrame(T, index=comps, columns=comps).round(6), use_container_width=True)

# =========================================================
# TODIM MODULE
# =========================================================
def ivffs_todim_module():
    st.header("📌 IVFFS‑TODIM: Multi‑Criteria Decision Making")
    st.markdown("Rank alternatives based on linguistic evaluations under multiple criteria.")

    with st.expander("ℹ️ TODIM linguistic scale (VP … VG)"):
        df_todim = pd.DataFrame([{"Code": k, "Meaning": TODIM_FULL[k], "IVFFS": format_ivffs(v)}
                                  for k, v in TODIM_LINGUISTIC.items()])
        st.dataframe(df_todim, hide_index=True, use_container_width=True)

    # Step 1: Alternatives, criteria, types, weights
    with st.container(border=True):
        st.subheader("Step 1: Define decision problem")
        c1, c2 = st.columns(2)
        alts_in = c1.text_input("Alternatives (comma‑separated)", "S1,S2,S3,S4", key="t_alts_in")
        crits_in = c2.text_input("Criteria (comma‑separated)", "C1,C2,C3", key="t_crits_in")
        alts = parse_csv_names(alts_in)
        crits = parse_csv_names(crits_in)

        if len(alts) < 2:
            st.warning("At least two alternatives required.")
            return
        if len(crits) < 1:
            st.warning("At least one criterion required.")
            return

        cfg_key = "t_cfg"
        if cfg_key not in st.session_state or list(st.session_state[cfg_key]["Criterion"]) != crits:
            w0 = [round(1/len(crits), 5)] * len(crits)
            if len(crits) > 1:
                w0[-1] = round(1.0 - sum(w0[:-1]), 5)
            st.session_state[cfg_key] = pd.DataFrame({
                "Criterion": crits,
                "Type": ["Benefit"] * len(crits),
                "Weight": w0
            })

        cfg = st.data_editor(
            st.session_state[cfg_key],
            hide_index=True,
            use_container_width=True,
            column_config={
                "Type": st.column_config.SelectboxColumn("Type", options=["Benefit", "Cost"]),
                "Weight": st.column_config.NumberColumn("Weight", format="%.5f", min_value=0.0, max_value=1.0, step=0.00001),
            },
            key="t_cfg_editor"
        )
        st.session_state[cfg_key] = cfg

        types = cfg["Type"].tolist()
        w_crit = cfg["Weight"].astype(float).tolist()
        if not np.isclose(sum(w_crit), 1.0):
            st.error(f"Criterion weights must sum to 1.0 (current sum = {sum(w_crit):.5f})")
            return
        else:
            st.success(f"✅ Sum of criterion weights = {sum(w_crit):.5f}")

    # Step 2: Experts
    with st.container(border=True):
        st.subheader("Step 2: Define experts")
        c1, c2 = st.columns(2)
        k = c1.number_input("Number of experts", min_value=1, max_value=30, value=4, step=1, key="t_k")
        alpha = c2.number_input("Dombi α", min_value=0.01, max_value=50.0, value=0.5, step=0.01, key="t_alpha")

        st.markdown("**Expert weights (sum = 1.0)**")
        if k == 1:
            w_exp = [1.0]
            st.success("Single expert → weight = 1.0")
        else:
            cols = st.columns(k)
            w_exp = []
            total_exp = 0.0
            for i in range(k):
                with cols[i]:
                    w = st.number_input(f"E{i+1}", min_value=0.0, max_value=1.0,
                                         value=round(1/k, 6), step=0.00001, format="%.6f",
                                         key=f"t_wexp_{i}")
                    w_exp.append(w)
                    total_exp += w
            if not np.isclose(total_exp, 1.0):
                st.error(f"❌ Expert weights must sum to 1.0 (current sum = {total_exp:.6f})")
                return
            else:
                st.success(f"✅ Sum = {total_exp:.6f}")

    # Step 3: Evaluation matrices
    mat_key = "t_terms"
    need_reset = (mat_key not in st.session_state) or (len(st.session_state[mat_key]) != k)
    if not need_reset:
        df0 = st.session_state[mat_key][0]
        need_reset = (list(df0.index) != alts) or (list(df0.columns) != crits)
    if need_reset:
        st.session_state[mat_key] = [pd.DataFrame("F", index=alts, columns=crits) for _ in range(k)]

    with st.container(border=True):
        st.subheader("Step 3: Enter expert evaluations")
        tabs = st.tabs([f"👤 Expert {i+1}" for i in range(k)])
        for i, tab in enumerate(tabs):
            with tab:
                st.session_state[mat_key][i] = st.data_editor(
                    st.session_state[mat_key][i],
                    use_container_width=True,
                    column_config={c: st.column_config.SelectboxColumn(c, options=list(TODIM_LINGUISTIC.keys()))
                                   for c in crits},
                    key=f"t_editor_{i}"
                )

    # Step 4: TODIM parameters
    with st.container(border=True):
        st.subheader("Step 4: TODIM parameters")
        c1, c2 = st.columns(2)
        theta = c1.number_input("Loss attenuation factor θ", min_value=0.01, max_value=50.0, value=1.0, step=0.01, key="t_theta")
        ref_mode = c2.selectbox("Reference criterion (ωr)", options=["Max weight (default)"] + crits, index=0, key="t_ref")

    # Run
    if st.button("🚀 Run IVFFS‑TODIM Analysis", type="primary", use_container_width=True, key="t_run_btn"):
        with st.spinner("Computing dominance and ranking..."):
            # Aggregate
            agg = {}
            for a in alts:
                for c in crits:
                    vals = []
                    for e in range(k):
                        term = st.session_state[mat_key][e].loc[a, c]
                        vals.append(TODIM_LINGUISTIC[term])
                    agg[(a, c)] = ivffdwa_aggregate(vals, w_exp, alpha)

            # Normalize
            norm = {}
            for j, c in enumerate(crits):
                for a in alts:
                    norm[(a, c)] = normalize_ivffs_todim(agg[(a, c)], types[j])

            # Score matrix
            tau = np.array([[ivffs_score(norm[(a, c)]) for c in crits] for a in alts], dtype=float)

            # Reference weight
            if ref_mode == "Max weight (default)":
                w_r = max(w_crit)
            else:
                w_r = float(w_crit[crits.index(ref_mode)])
            if abs(w_r) < EPS:
                st.error("Reference weight ωr is zero.")
                return

            w_rel = np.array([w / w_r for w in w_crit], dtype=float)
            sum_w_rel = float(w_rel.sum())
            if sum_w_rel <= EPS:
                st.error("Sum of relative weights is zero.")
                return

            # Dominance matrix
            nA, nC = len(alts), len(crits)
            dominance = np.zeros((nA, nA), dtype=float)
            for i in range(nA):
                for q in range(nA):
                    if i == q:
                        continue
                    s = 0.0
                    for j in range(nC):
                        diff = tau[i, j] - tau[q, j]
                        ad = abs(diff)
                        if ad <= 0:
                            continue
                        if diff > 0:
                            s += math.sqrt((w_rel[j] * ad) / sum_w_rel)
                        else:
                            s += (-1.0 / theta) * math.sqrt((sum_w_rel * ad) / max(EPS, w_rel[j]))
                    dominance[i, q] = s

            # Overall superiority
            Phi = dominance.sum(axis=1)
            ranking = pd.DataFrame({
                "Alternative": alts,
                "Φ (superiority)": Phi,
                "Rank": Phi.rank(ascending=False, method="min").astype(int)
            }).sort_values("Rank").reset_index(drop=True)

            # Display results
            st.subheader("📊 Ranking Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Alternatives", nA)
            with col2:
                st.metric("Criteria", nC)
            with col3:
                st.metric("θ", f"{theta:.3f}")

            st.dataframe(ranking.round(6), hide_index=True, use_container_width=True)

            # Bar chart
            fig = px.bar(ranking, x="Alternative", y="Φ (superiority)",
                         color="Φ (superiority)", color_continuous_scale="Blues",
                         title="Overall Superiority of Alternatives")
            st.plotly_chart(fig, use_container_width=True)

            # Optional expanders for intermediate matrices
            with st.expander("📋 View intermediate matrices"):
                st.markdown("**Aggregated IVFFS matrix**")
                st.dataframe(pd.DataFrame([[format_ivffs(agg[(a, c)]) for c in crits] for a in alts],
                                          index=alts, columns=crits), use_container_width=True)
                st.markdown("**Normalized IVFFS matrix**")
                st.dataframe(pd.DataFrame([[format_ivffs(norm[(a, c)]) for c in crits] for a in alts],
                                          index=alts, columns=crits), use_container_width=True)
                st.markdown("**Score matrix τ**")
                st.dataframe(pd.DataFrame(tau, index=alts, columns=crits).round(6), use_container_width=True)
                st.markdown("**Dominance matrix δ**")
                st.dataframe(pd.DataFrame(dominance, index=alts, columns=alts).round(6), use_container_width=True)

# =========================================================
# MAIN APP
# =========================================================
def main():
    st.sidebar.image("https://via.placeholder.com/150x50?text=IVFFS+Toolkit", use_container_width=True)  # placeholder logo
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a module", ["IVFFS‑WINGS", "IVFFS‑TODIM"])

    if page == "IVFFS‑WINGS":
        ivffs_wings_module()
    else:
        ivffs_todim_module()

    st.markdown('<div class="footer">Developed with Streamlit • Interval‑Valued Fermatean Fuzzy Sets</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

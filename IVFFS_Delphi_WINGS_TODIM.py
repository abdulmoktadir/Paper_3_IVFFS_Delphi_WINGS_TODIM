import math
import numpy as np
import pandas as pd
import streamlit as st

EPS = 1e-12

# =========================================================
# IVFFS REPRESENTATION
#   IVFFS = ([mu_L, mu_U], [nu_L, nu_U])
#   We'll store as (mu_L, mu_U, nu_L, nu_U)
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
    return f"([{muL:.6f},{muU:.6f}],[{nuL:.6f},{nuU:.6f}])"

# =========================================================
# IVFFDWA (Dombi-based) AGGREGATION (Excel-matching)
# Membership (mu bounds a/b):
#   a = ( 1 - 1/(1 + ( Σ λi * ( (x_i^3/(1-x_i^3))^α ) )^(1/α) ) ) )^(1/3)
# Nonmembership (nu bounds c/d):
#   c = 1 / ( (1 + ( Σ λi * ( ((1-y_i^3)/y_i^3)^α ) )^(1/α) ) )^(1/3) )
# =========================================================

def _safe_pow(x, p):
    return float(x) ** float(p)

def agg_membership_bound(x_list, w_list, alpha, power=3.0):
    alpha = float(alpha)
    alpha = max(alpha, EPS)

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
    alpha = float(alpha)
    alpha = max(alpha, EPS)

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
# IVFFS SCORE FUNCTION Ψ (used in WINGS + TODIM)
# Ψ(β) = 1/2 * ( 1/2*(muL^3 + muU^3 - nuL^3 - nuU^3) + 1 )
# =========================================================

def ivffs_score(v):
    muL, muU, nuL, nuU = v
    return 0.5 * (0.5 * (muL**3 + muU**3 - nuL**3 - nuU**3) + 1.0)

# =========================================================
# WINGS LINGUISTIC SCALES (YOUR REQUIRED TABLES)
# =========================================================

# Table A1 (Biswas et al., 2023): Strength evaluation
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

# Table A4 (Seikh & Mandal, 2023): Causal influence evaluation
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

# =========================================================
# TODIM LINGUISTIC SCALE (VP…VG) (your earlier requirement)
# =========================================================

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
# TODIM normalization (paper): Benefit same, Cost swap μ ↔ ν
# =========================================================

def normalize_ivffs(v, crit_type: str):
    crit_type = (crit_type or "").strip().lower()
    if crit_type.startswith("c"):  # cost
        muL, muU, nuL, nuU = v
        return make_ivffs(nuL, nuU, muL, muU)
    return v

# =========================================================
# Helpers
# =========================================================

def parse_csv_names(s):
    return [x.strip() for x in (s or "").split(",") if x.strip()]

def df_numeric_round(df: pd.DataFrame, decimals=6):
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="ignore")
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(decimals)
    return out

# =========================================================
# MODULE 1: IVFFS-WINGS
#   Inputs:
#     strengths per component (Table A1)
#     influence matrix (Table A4)
#   Aggregation: IVFFDWA (Excel)
# =========================================================

def ivffs_wings_module():
    st.header("📌 IVFFS-WINGS")
    st.caption("Strength scale (Biswas et al., 2023) + Influence scale (Seikh & Mandal, 2023) + IVFFDWA aggregation (Excel)")

    with st.expander("Strength linguistic scale (Table A1)", expanded=False):
        df1 = pd.DataFrame(
            [{"Code": k, "Meaning": WINGS_STRENGTH_FULL[k], "IVFFS": format_ivffs(v)} for k, v in WINGS_STRENGTH.items()]
        )
        st.dataframe(df1, hide_index=True, use_container_width=True)

    with st.expander("Influence linguistic scale (Table A4)", expanded=False):
        df2 = pd.DataFrame(
            [{"Code": k, "Meaning": WINGS_INFLUENCE_FULL[k], "IVFFS": format_ivffs(v)} for k, v in WINGS_INFLUENCE.items()]
        )
        st.dataframe(df2, hide_index=True, use_container_width=True)

    st.subheader("Step 1: Components and Experts")
    n = st.number_input("Number of components", min_value=2, max_value=30, value=5, step=1, key="w_n")
    k = st.number_input("Number of experts", min_value=1, max_value=20, value=4, step=1, key="w_k")

    colA, colB = st.columns(2)
    comp_names_in = colA.text_input("Component names (comma-separated)", value="C1,C2,C3,C4,C5", key="w_comps_in")
    alpha = colB.number_input("Dombi parameter (α)", min_value=0.01, max_value=50.0, value=0.5, step=0.01, key="w_alpha")

    comps = parse_csv_names(comp_names_in)
    if len(comps) != int(n):
        comps = [f"C{i+1}" for i in range(int(n))]

    st.markdown("**Expert weights (must sum to 1.0)**")
    if int(k) == 1:
        w_exp = [1.0]
        st.info("Single expert → weight = 1.0")
    else:
        cols = st.columns(int(k))
        w_exp = []
        for i in range(int(k)):
            with cols[i]:
                w_exp.append(
                    st.number_input(
                        f"E{i+1}", min_value=0.0, max_value=1.0,
                        value=round(1/int(k), 6),
                        step=0.00001, format="%.6f",
                        key=f"w_exp_{i}"
                    )
                )
        if not np.isclose(sum(w_exp), 1.0):
            st.error(f"Expert weights must sum to 1.0 (now {sum(w_exp):.6f}).")
            return

    # session state containers
    sk_strength = "w_strength_terms"
    sk_infl = "w_infl_terms"

    if sk_strength not in st.session_state or len(st.session_state[sk_strength]) != int(k):
        st.session_state[sk_strength] = [
            pd.DataFrame({"Component": comps, "Strength": ["MI"] * int(n)}) for _ in range(int(k))
        ]
    if sk_infl not in st.session_state or len(st.session_state[sk_infl]) != int(k):
        st.session_state[sk_infl] = [
            pd.DataFrame("ELI", index=comps, columns=comps) for _ in range(int(k))
        ]
        for e in range(int(k)):
            for i in range(int(n)):
                st.session_state[sk_infl][e].iloc[i, i] = "—"

    st.subheader("Step 2: Expert inputs")

    tabs = st.tabs([f"Expert {i+1}" for i in range(int(k))])
    for e, tab in enumerate(tabs):
        with tab:
            st.markdown("### (A) Strength of each component (diagonal) — Table A1")
            dfS = st.session_state[sk_strength][e].copy()
            dfS["Component"] = comps
            dfS = st.data_editor(
                dfS,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Component": st.column_config.TextColumn("Component", disabled=True),
                    "Strength": st.column_config.SelectboxColumn("Strength", options=list(WINGS_STRENGTH.keys()))
                },
                key=f"w_strength_editor_{e}"
            )
            st.session_state[sk_strength][e] = dfS

            st.markdown("### (B) Influence matrix (row → column) — Table A4 (off-diagonal only)")
            dfI = st.session_state[sk_infl][e].copy()

            # ensure correct indices/cols
            dfI = dfI.reindex(index=comps, columns=comps, fill_value="ELI")
            for i in range(int(n)):
                dfI.iloc[i, i] = "—"

            # build column config: all influence codes allowed, but diagonal visually fixed as "—"
            col_cfg = {c: st.column_config.SelectboxColumn(c, options=["—"] + list(WINGS_INFLUENCE.keys()))
                       for c in comps}

            editedI = st.data_editor(
                dfI,
                use_container_width=True,
                column_config=col_cfg,
                key=f"w_infl_editor_{e}"
            )

            # enforce diagonal stays "—"
            for i in range(int(n)):
                editedI.iloc[i, i] = "—"

            st.session_state[sk_infl][e] = editedI

    if st.button("✅ Run IVFFS-WINGS", type="primary", use_container_width=True, key="w_run_btn"):
        with st.spinner("Computing IVFFS-WINGS..."):
            # Build SIDRM per expert in IVFFS values
            expert_sidrm = []
            for e in range(int(k)):
                strength_terms = list(st.session_state[sk_strength][e]["Strength"])
                infl_terms = st.session_state[sk_infl][e]

                mat = [[None]*int(n) for _ in range(int(n))]
                for i in range(int(n)):
                    for j in range(int(n)):
                        if i == j:
                            mat[i][j] = WINGS_STRENGTH[strength_terms[i]]
                        else:
                            term = infl_terms.iloc[i, j]
                            if term == "—":
                                term = "ELI"
                            mat[i][j] = WINGS_INFLUENCE[term]
                expert_sidrm.append(mat)

            # Aggregate experts cell-wise using IVFFDWA
            agg = [[None]*int(n) for _ in range(int(n))]
            for i in range(int(n)):
                for j in range(int(n)):
                    cell_list = [expert_sidrm[e][i][j] for e in range(int(k))]
                    agg[i][j] = ivffdwa_aggregate(cell_list, w_exp, alpha)

            df_agg = pd.DataFrame([[format_ivffs(agg[i][j]) for j in range(int(n))] for i in range(int(n))],
                                  index=comps, columns=comps)
            st.subheader("Aggregated IVFFS-SIDRM (IVFFDWA)")
            st.dataframe(df_agg, use_container_width=True)

            # Score matrix C using Ψ
            C = np.array([[ivffs_score(agg[i][j]) for j in range(int(n))] for i in range(int(n))], dtype=float)
            dfC = pd.DataFrame(C, index=comps, columns=comps)
            st.subheader("Score Matrix C (Ψ)")
            st.dataframe(df_numeric_round(dfC, 6), use_container_width=True)

            # Normalize: N = C / sum(C)
            varsigma = float(C.sum())
            if abs(varsigma) < EPS:
                st.error("Normalization failed: sum(C) is zero.")
                return
            N = C / varsigma
            dfN = pd.DataFrame(N, index=comps, columns=comps)
            st.subheader("Normalized Score Matrix N")
            st.dataframe(df_numeric_round(dfN, 6), use_container_width=True)

            # Total relation: T = N @ pinv(I - N)
            I = np.eye(int(n))
            T = N @ np.linalg.pinv(I - N)
            dfT = pd.DataFrame(T, index=comps, columns=comps)
            st.subheader("Total Relation Matrix T")
            st.dataframe(df_numeric_round(dfT, 6), use_container_width=True)

            # TI/TR/Engagement/Role/Expected value/Weight
            TI = T.sum(axis=1)
            TR = T.sum(axis=0)
            ENG = TI + TR
            ROLE = TI - TR
            EV = np.sqrt(ENG**2 + ROLE**2)
            W = EV / max(EPS, EV.sum())

            out = pd.DataFrame({
                "Component": comps,
                "TI": TI,
                "TR": TR,
                "Engagement": ENG,
                "Role": ROLE,
                "Expected value": EV,
                "Weight": W
            })
            out["Type"] = np.where(out["Role"] >= 0, "Cause", "Effect")

            st.subheader("IVFFS-WINGS Results")
            st.dataframe(df_numeric_round(out, 6), use_container_width=True, hide_index=True)

# =========================================================
# MODULE 2: IVFFS-TODIM (kept with your VP..VG scale)
# =========================================================

def ivffs_todim_module():
    st.header("📌 IVFFS-TODIM")
    st.caption("VP…VG TODIM scale + IVFFDWA aggregation (Excel) + normalization (Benefit keep, Cost swap μ↔ν)")

    with st.expander("TODIM Linguistic scale (VP…VG)", expanded=False):
        scale_df = pd.DataFrame(
            [{"Code": k, "Meaning": TODIM_FULL[k], "IVFFS": format_ivffs(v)} for k, v in TODIM_LINGUISTIC.items()]
        )
        st.dataframe(scale_df, hide_index=True, use_container_width=True)

    st.subheader("Step 1: Alternatives, Criteria, Types, Weights")
    col1, col2 = st.columns(2)
    alts_in = col1.text_input("Alternatives (comma-separated)", "S1,S2,S3,S4", key="t_alts_in")
    crits_in = col2.text_input("Criteria (comma-separated)", "C1,C2,C3", key="t_crits_in")

    alts = parse_csv_names(alts_in)
    crits = parse_csv_names(crits_in)

    if len(alts) < 2 or len(crits) < 1:
        st.warning("Provide at least 2 alternatives and at least 1 criterion.")
        return

    cfg_key = "t_crit_cfg"
    if cfg_key not in st.session_state or list(st.session_state[cfg_key]["Criterion"]) != crits:
        w0 = [round(1/len(crits), 5)] * len(crits)
        if len(crits) > 1:
            w0[-1] = round(1.0 - sum(w0[:-1]), 5)
        st.session_state[cfg_key] = pd.DataFrame({
            "Criterion": crits,
            "Type": ["Benefit"] * len(crits),
            "Weight": w0
        })

    edited_cfg = st.data_editor(
        st.session_state[cfg_key],
        hide_index=True,
        use_container_width=True,
        column_config={
            "Type": st.column_config.SelectboxColumn("Type", options=["Benefit", "Cost"]),
            "Weight": st.column_config.NumberColumn("Weight", format="%.5f", min_value=0.0, max_value=1.0, step=0.00001),
        },
        key="t_cfg_editor"
    )

    types = edited_cfg["Type"].tolist()
    w_crit = edited_cfg["Weight"].astype(float).tolist()

    if not np.isclose(sum(w_crit), 1.0):
        st.error(f"Criteria weights must sum to 1.0 (now {sum(w_crit):.5f}).")
        return

    st.subheader("Step 2: Experts, weights, and linguistic evaluations")
    k = st.number_input("Number of experts", min_value=1, max_value=30, value=4, step=1, key="t_k")
    alpha = st.number_input("Dombi parameter (α)", min_value=0.01, max_value=50.0, value=0.5, step=0.01, key="t_alpha")

    st.markdown("**Expert weights (sum = 1.0)**")
    if int(k) == 1:
        w_exp = [1.0]
        st.info("Single expert → weight = 1.0")
    else:
        cols = st.columns(int(k))
        w_exp = []
        for i in range(int(k)):
            with cols[i]:
                w_exp.append(
                    st.number_input(f"E{i+1}", min_value=0.0, max_value=1.0,
                                    value=round(1/int(k), 6),
                                    step=0.00001, format="%.6f", key=f"t_wexp_{i}")
                )
        if not np.isclose(sum(w_exp), 1.0):
            st.error(f"Expert weights must sum to 1.0 (now {sum(w_exp):.6f}).")
            return

    mat_key = "t_expert_terms"
    need_reset = (mat_key not in st.session_state) or (len(st.session_state[mat_key]) != int(k))
    if not need_reset:
        df0 = st.session_state[mat_key][0]
        need_reset = (list(df0.index) != alts) or (list(df0.columns) != crits)

    if need_reset:
        st.session_state[mat_key] = [pd.DataFrame("F", index=alts, columns=crits) for _ in range(int(k))]

    tabs = st.tabs([f"Expert {i+1}" for i in range(int(k))])
    for i, tab in enumerate(tabs):
        with tab:
            st.session_state[mat_key][i] = st.data_editor(
                st.session_state[mat_key][i],
                use_container_width=True,
                column_config={c: st.column_config.SelectboxColumn(c, options=list(TODIM_LINGUISTIC.keys()))
                               for c in crits},
                key=f"t_editor_{i}"
            )

    st.subheader("Step 3: TODIM parameters")
    theta = st.number_input("Loss attenuation factor (θ)", min_value=0.01, max_value=50.0, value=1.0, step=0.01, key="t_theta")
    ref_mode = st.selectbox("Reference criterion (ωr)", options=["Max weight (default)"] + crits, index=0, key="t_ref")

    if st.button("✅ Run IVFFS-TODIM", type="primary", use_container_width=True, key="t_run_btn"):
        with st.spinner("Computing IVFFS-TODIM..."):
            # aggregate
            agg = {}
            for a in alts:
                for c in crits:
                    vals = []
                    for e in range(int(k)):
                        term = st.session_state[mat_key][e].loc[a, c]
                        vals.append(TODIM_LINGUISTIC[term])
                    agg[(a, c)] = ivffdwa_aggregate(vals, w_exp, alpha)

            df_agg = pd.DataFrame(
                [[format_ivffs(agg[(a, c)]) for c in crits] for a in alts],
                index=alts, columns=crits
            )
            st.subheader("Aggregated IVFFS Decision Matrix (IVFFDWA)")
            st.dataframe(df_agg, use_container_width=True)

            # normalize
            norm = {}
            for j, c in enumerate(crits):
                for a in alts:
                    norm[(a, c)] = normalize_ivffs(agg[(a, c)], types[j])

            df_norm = pd.DataFrame(
                [[format_ivffs(norm[(a, c)]) for c in crits] for a in alts],
                index=alts, columns=crits
            )
            st.subheader("Normalized IVFFS Matrix (Benefit: same, Cost: swap μ ↔ ν)")
            st.dataframe(df_norm, use_container_width=True)

            # score matrix
            tau = np.array([[ivffs_score(norm[(a, c)]) for c in crits] for a in alts], dtype=float)
            df_tau = pd.DataFrame(tau, index=alts, columns=crits)
            st.subheader("Score Matrix τ (Ψ)")
            st.dataframe(df_numeric_round(df_tau, 6), use_container_width=True)

            # reference weight
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

            df_w = pd.DataFrame({"Criterion": crits, "Weight ω": w_crit, "Relative weight ω'": w_rel})
            st.subheader("Criteria weights (ω) and relative weights (ω')")
            st.dataframe(df_numeric_round(df_w, 6), use_container_width=True, hide_index=True)

            # dominance
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

            df_dom = pd.DataFrame(dominance, index=alts, columns=alts)
            st.subheader("Dominance matrix δ(i,q)")
            st.dataframe(df_numeric_round(df_dom, 6), use_container_width=True)

            # overall superiority
            Phi = dominance.sum(axis=1)
            df_phi = pd.DataFrame({"Alternative": alts, "Φ (overall superiority)": Phi})
            df_phi["Rank"] = df_phi["Φ (overall superiority)"].rank(ascending=False, method="min").astype(int)
            df_phi = df_phi.sort_values("Rank").reset_index(drop=True)

            st.subheader("Final IVFFS-TODIM Ranking")
            st.dataframe(df_numeric_round(df_phi, 6), use_container_width=True, hide_index=True)

# =========================================================
# MAIN APP (Two Modules)
# =========================================================

def main():
    st.set_page_config(page_title="IVFFS Toolkit (WINGS + TODIM)", layout="wide")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a Module", ["IVFFS-WINGS", "IVFFS-TODIM"])

    if page == "IVFFS-WINGS":
        ivffs_wings_module()
    else:
        ivffs_todim_module()

if __name__ == "__main__":
    main()

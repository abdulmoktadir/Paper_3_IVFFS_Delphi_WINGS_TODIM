import math
import numpy as np
import pandas as pd
import streamlit as st

# Optional exports / visuals
import io
import base64
import matplotlib.pyplot as plt

# =========================================================
# IVFFS REPRESENTATION
#   IVFFS = ([mu_L, mu_U], [nu_L, nu_U])
#   We'll store as (mu_L, mu_U, nu_L, nu_U)
# =========================================================

EPS = 1e-12

def clamp01(x: float) -> float:
    return float(min(1.0 - EPS, max(EPS, x)))

def format_ivffs(v):
    muL, muU, nuL, nuU = v
    return f"([{muL:.6f},{muU:.6f}],[{nuL:.6f},{nuU:.6f}])"

def make_ivffs(muL, muU, nuL, nuU):
    muL, muU, nuL, nuU = map(float, (muL, muU, nuL, nuU))
    # keep order
    muL, muU = min(muL, muU), max(muL, muU)
    nuL, nuU = min(nuL, nuU), max(nuL, nuU)
    # clamp to (0,1) open interval to avoid div by zero in Dombi
    return (clamp01(muL), clamp01(muU), clamp01(nuL), clamp01(nuU))

# =========================================================
# YOUR IVFFS-TODIM LINGUISTIC SCALE (VP…VG)
#   Very Poor  VP ([0.10,0.15],[0.90,0.95])
#   Poor       P  ([0.20,0.25],[0.80,0.85])
#   MediumPoor MP ([0.30,0.35],[0.70,0.75])
#   Fair       F  ([0.50,0.55],[0.40,0.45])
#   Med Good   MG ([0.70,0.75],[0.30,0.35])
#   Good       G  ([0.80,0.85],[0.20,0.25])
#   Very Good  VG ([0.90,0.95],[0.10,0.15])
# =========================================================

IVFFS_LINGUISTIC = {
    "VP": make_ivffs(0.10, 0.15, 0.90, 0.95),
    "P":  make_ivffs(0.20, 0.25, 0.80, 0.85),
    "MP": make_ivffs(0.30, 0.35, 0.70, 0.75),
    "F":  make_ivffs(0.50, 0.55, 0.40, 0.45),
    "MG": make_ivffs(0.70, 0.75, 0.30, 0.35),
    "G":  make_ivffs(0.80, 0.85, 0.20, 0.25),
    "VG": make_ivffs(0.90, 0.95, 0.10, 0.15),
}

IVFFS_FULL = {
    "VP": "Very Poor",
    "P":  "Poor",
    "MP": "Medium Poor",
    "F":  "Fair",
    "MG": "Medium Good",
    "G":  "Good",
    "VG": "Very Good",
}

# =========================================================
# IVFFDWA (DTN/DTCN-based) AGGREGATION (Excel-matching)
#
# Your Excel membership aggregation for a/b:
#   a = ( 1 - 1/(1 + ( Σ λi * ( (x_i^3/(1-x_i^3))^α ) )^(1/α) ) ) )^(1/3)
#
# Your Excel nonmembership aggregation for c/d:
#   c = 1 / ( (1 + ( Σ λi * ( ((1-y_i^3)/y_i^3)^α ) )^(1/α) ) )^(1/3) )
#
# Where:
#   x = membership bound (mu_L or mu_U)
#   y = nonmembership bound (nu_L or nu_U)
#   α = Dombi parameter (your Excel uses a cell like $B$96)
#   λi = expert weight
#   power = 3 (fixed in your formulas)
# =========================================================

def _safe_pow(x, p):
    return float(x) ** float(p)

def agg_membership_bound(x_list, w_list, alpha, power=3.0):
    """
    Excel-matching for mu bounds (a,b).
    """
    alpha = float(alpha)
    alpha = max(alpha, EPS)

    s = 0.0
    for x, w in zip(x_list, w_list):
        x = clamp01(float(x))
        w = float(w)
        xp = _safe_pow(x, power)
        # (x^p)/(1-x^p)
        frac = xp / max(EPS, (1.0 - xp))
        s += w * _safe_pow(frac, alpha)

    inner = _safe_pow(s, 1.0/alpha)
    val = 1.0 - (1.0 / (1.0 + inner))
    return _safe_pow(val, 1.0/power)

def agg_nonmembership_bound(y_list, w_list, alpha, power=3.0):
    """
    Excel-matching for nu bounds (c,d).
    """
    alpha = float(alpha)
    alpha = max(alpha, EPS)

    s = 0.0
    for y, w in zip(y_list, w_list):
        y = clamp01(float(y))
        w = float(w)
        yp = _safe_pow(y, power)
        # ((1-y^p)/(y^p))
        frac = (1.0 - yp) / max(EPS, yp)
        s += w * _safe_pow(frac, alpha)

    inner = _safe_pow(s, 1.0/alpha)
    denom = _safe_pow((1.0 + inner), 1.0/power)
    return 1.0 / max(EPS, denom)

def ivffdwa_aggregate(ivffs_list, w_list, alpha):
    """
    Aggregates a list of IVFFS tuples using your Excel DTN/DTCN formulas.
    Returns aggregated IVFFS (muL, muU, nuL, nuU).
    """
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
# IVFFS SCORE FUNCTION Ψ (paper)
# Ψ(β) = 1/2 * ( 1/2*(muL^3 + muU^3 - nuL^3 - nuU^3) + 1 )
# Used in BOTH WINGS and TODIM
# =========================================================

def ivffs_score(v):
    muL, muU, nuL, nuU = v
    muL3 = muL**3
    muU3 = muU**3
    nuL3 = nuL**3
    nuU3 = nuU**3
    return 0.5 * (0.5 * (muL3 + muU3 - nuL3 - nuU3) + 1.0)

# =========================================================
# NORMALIZATION for IVFFS-TODIM (paper Definition 7)
# Benefit: keep same
# Cost: swap membership & nonmembership intervals:
#   ( [muL,muU],[nuL,nuU] ) -> ( [nuL,nuU],[muL,muU] )
# =========================================================

def normalize_ivffs(v, crit_type: str):
    crit_type = (crit_type or "").strip().lower()
    if crit_type.startswith("c"):  # cost
        muL, muU, nuL, nuU = v
        return make_ivffs(nuL, nuU, muL, muU)
    return v  # benefit

# =========================================================
# Utilities
# =========================================================

def df_numeric_round(df: pd.DataFrame, decimals=6):
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="ignore")
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(decimals)
    return out

def parse_csv_names(s):
    return [x.strip() for x in (s or "").split(",") if x.strip()]

# =========================================================
# MODULE 1: IVFFS-WINGS
# =========================================================

def ivffs_wings_module():
    st.header("📌 IVFFS-WINGS")
    st.caption("IVFFS-WINGS using IVFFDWA (Excel aggregation) + score Ψ + total relation matrix + TI/TR/Engagement/Role/Expected value/Weight")

    with st.expander("IVFFS Linguistic Scale (VP…VG)", expanded=False):
        scale_df = pd.DataFrame(
            [{"Abbr": k, "Meaning": IVFFS_FULL[k], "IVFFS": format_ivffs(v)} for k, v in IVFFS_LINGUISTIC.items()]
        )
        st.dataframe(scale_df, hide_index=True, use_container_width=True)

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
    if k == 1:
        w_exp = [1.0]
        st.info("Single expert → weight = 1.0")
    else:
        cols = st.columns(int(k))
        w_exp = []
        for i in range(int(k)):
            with cols[i]:
                w_exp.append(
                    st.number_input(f"E{i+1}", min_value=0.0, max_value=1.0, value=round(1/int(k), 6),
                                    step=0.00001, format="%.6f", key=f"w_exp_{i}")
                )
        if not np.isclose(sum(w_exp), 1.0):
            st.error(f"Expert weights must sum to 1.0 (now {sum(w_exp):.6f}).")
            return

    st.subheader("Step 2: Expert SIDRM (Diagonal = Strength; Off-diagonal = Influence)")
    st.caption("Use VP..VG for both strength and influence to match your Excel encoding.")

    # session state for expert matrices
    state_key = "w_sidrm_terms"
    need_reset = (state_key not in st.session_state) or (len(st.session_state[state_key]) != int(k))
    if not need_reset:
        # shape check
        m0 = st.session_state[state_key][0]
        need_reset = (len(m0) != int(n)) or (len(m0[0]) != int(n))

    if need_reset:
        st.session_state[state_key] = []
        for _ in range(int(k)):
            mat = [["VP"] * int(n) for __ in range(int(n))]
            for i in range(int(n)):
                mat[i][i] = "MG"
            st.session_state[state_key].append(mat)

    tabs = st.tabs([f"Expert {i+1}" for i in range(int(k))])
    for e, tab in enumerate(tabs):
        with tab:
            st.write("Diagonal = Strength; Off-diagonal = Influence (row → column)")
            df_terms = pd.DataFrame(st.session_state[state_key][e], index=comps, columns=comps)
            edited = st.data_editor(
                df_terms,
                use_container_width=True,
                column_config={c: st.column_config.SelectboxColumn(c, options=list(IVFFS_LINGUISTIC.keys()))
                               for c in df_terms.columns},
                key=f"w_editor_{e}"
            )
            st.session_state[state_key][e] = edited.values.tolist()

    if st.button("✅ Run IVFFS-WINGS", type="primary", use_container_width=True, key="w_run_btn"):
        with st.spinner("Computing IVFFS-WINGS..."):
            # 2.1 Build IVFFS matrices for each expert
            expert_ivffs = []
            for e in range(int(k)):
                mat = [[None]*int(n) for _ in range(int(n))]
                for i in range(int(n)):
                    for j in range(int(n)):
                        term = st.session_state[state_key][e][i][j]
                        mat[i][j] = IVFFS_LINGUISTIC[term]
                expert_ivffs.append(mat)

            # 2.2 Aggregate experts cell-wise using IVFFDWA
            agg = [[None]*int(n) for _ in range(int(n))]
            for i in range(int(n)):
                for j in range(int(n)):
                    cell_list = [expert_ivffs[e][i][j] for e in range(int(k))]
                    agg[i][j] = ivffdwa_aggregate(cell_list, w_exp, alpha)

            # Show aggregated IVFFS SIDRM
            df_agg = pd.DataFrame([[format_ivffs(agg[i][j]) for j in range(int(n))] for i in range(int(n))],
                                  index=comps, columns=comps)
            st.subheader("Aggregated IVFFS-SIDRM (IVFFDWA)")
            st.dataframe(df_agg, use_container_width=True)

            # 2.3 Score matrix C using Ψ
            C = np.array([[ivffs_score(agg[i][j]) for j in range(int(n))] for i in range(int(n))], dtype=float)

            dfC = pd.DataFrame(C, index=comps, columns=comps)
            st.subheader("Score Matrix C (Ψ)")
            st.dataframe(df_numeric_round(dfC, 6), use_container_width=True)

            # 2.4 Normalize C -> N by varsigma (sum of all elements)
            varsigma = float(C.sum())
            if abs(varsigma) < EPS:
                st.error("Normalization failed: sum(C) is zero.")
                return
            N = C / varsigma

            dfN = pd.DataFrame(N, index=comps, columns=comps)
            st.subheader("Normalized Score Matrix N")
            st.dataframe(df_numeric_round(dfN, 6), use_container_width=True)

            # 2.5 Total relation matrix T = N @ pinv(I - N)
            I = np.eye(int(n))
            T = N @ np.linalg.pinv(I - N)

            dfT = pd.DataFrame(T, index=comps, columns=comps)
            st.subheader("Total Relation Matrix T")
            st.dataframe(df_numeric_round(dfT, 6), use_container_width=True)

            # 2.6 TI/TR/Engagement/Role/Expected value/Weight
            TI = T.sum(axis=1)          # row sum
            TR = T.sum(axis=0)          # col sum
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
# MODULE 2: IVFFS-TODIM
# =========================================================

def ivffs_todim_module():
    st.header("📌 IVFFS-TODIM")
    st.caption("Aggregation by IVFFDWA (Excel formulas) + normalization (paper Definition 7) + TODIM dominance (Excel-like with negative loss)")

    with st.expander("IVFFS Linguistic Scale (VP…VG)", expanded=False):
        scale_df = pd.DataFrame(
            [{"Abbr": k, "Meaning": IVFFS_FULL[k], "IVFFS": format_ivffs(v)} for k, v in IVFFS_LINGUISTIC.items()]
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

    # Criteria config table (types + weights)
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
            "Type": st.column_config.SelectboxColumn("Type", options=["Benefit","Cost"]),
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
    if k == 1:
        w_exp = [1.0]
        st.info("Single expert → weight = 1.0")
    else:
        cols = st.columns(int(k))
        w_exp = []
        for i in range(int(k)):
            with cols[i]:
                w_exp.append(
                    st.number_input(f"E{i+1}", min_value=0.0, max_value=1.0,
                                    value=round(1/int(k), 6), step=0.00001,
                                    format="%.6f", key=f"t_wexp_{i}")
                )
        if not np.isclose(sum(w_exp), 1.0):
            st.error(f"Expert weights must sum to 1.0 (now {sum(w_exp):.6f}).")
            return

    # per expert decision matrices of linguistic terms
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
                column_config={c: st.column_config.SelectboxColumn(c, options=list(IVFFS_LINGUISTIC.keys()))
                               for c in crits},
                key=f"t_editor_{i}"
            )

    st.subheader("Step 3: TODIM parameters")
    theta = st.number_input("Loss attenuation factor (θ)", min_value=0.01, max_value=50.0, value=1.0, step=0.01, key="t_theta")
    ref_mode = st.selectbox("Reference criterion (ωr)", options=["Max weight (default)"] + crits, index=0, key="t_ref")

    if st.button("✅ Run IVFFS-TODIM", type="primary", use_container_width=True, key="t_run_btn"):
        with st.spinner("Computing IVFFS-TODIM..."):
            # 3.1 Aggregate expert matrices -> IVFFS per (alt, crit)
            agg = {}
            for a in alts:
                for c in crits:
                    vals = []
                    for e in range(int(k)):
                        term = st.session_state[mat_key][e].loc[a, c]
                        vals.append(IVFFS_LINGUISTIC[term])
                    agg[(a, c)] = ivffdwa_aggregate(vals, w_exp, alpha)

            df_agg = pd.DataFrame(
                [[format_ivffs(agg[(a,c)]) for c in crits] for a in alts],
                index=alts, columns=crits
            )
            st.subheader("Aggregated IVFFS Decision Matrix (IVFFDWA)")
            st.dataframe(df_agg, use_container_width=True)

            # 3.2 Normalize (paper Definition 7)
            norm = {}
            for j, c in enumerate(crits):
                for a in alts:
                    norm[(a, c)] = normalize_ivffs(agg[(a, c)], types[j])

            df_norm = pd.DataFrame(
                [[format_ivffs(norm[(a,c)]) for c in crits] for a in alts],
                index=alts, columns=crits
            )
            st.subheader("Normalized IVFFS Matrix (Benefit: same, Cost: swap μ ↔ ν)")
            st.dataframe(df_norm, use_container_width=True)

            # 3.3 Score matrix τ using Ψ
            tau = np.array([[ivffs_score(norm[(a,c)]) for c in crits] for a in alts], dtype=float)
            df_tau = pd.DataFrame(tau, index=alts, columns=crits)
            st.subheader("Score Matrix τ (Ψ)")
            st.dataframe(df_numeric_round(df_tau, 6), use_container_width=True)

            # 3.4 Relative weights ω' (TODIM)
            if ref_mode == "Max weight (default)":
                w_r = max(w_crit)
            else:
                idx_r = crits.index(ref_mode)
                w_r = float(w_crit[idx_r])

            if abs(w_r) < EPS:
                st.error("Reference weight ωr is zero.")
                return

            w_rel = np.array([w / w_r for w in w_crit], dtype=float)
            sum_w_rel = float(w_rel.sum())
            if sum_w_rel <= EPS:
                st.error("Sum of relative weights is zero.")
                return

            df_w = pd.DataFrame({
                "Criterion": crits,
                "Weight ω": w_crit,
                "Relative weight ω'": w_rel
            })
            st.subheader("Criteria weights (ω) and relative weights (ω')")
            st.dataframe(df_numeric_round(df_w, 6), use_container_width=True, hide_index=True)

            # 3.5 TODIM dominance ψj(Si,Sq)  (Excel-like)
            # gain:  sqrt( (ω'j * |diff|) / Σω' )
            # loss: -1/θ * sqrt( (Σω' * |diff|) / ω'j )
            nA = len(alts)
            nC = len(crits)

            dominance = np.zeros((nA, nA), dtype=float)  # δ(i,q)
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
                            s += (-1.0/theta) * math.sqrt((sum_w_rel * ad) / max(EPS, w_rel[j]))
                    dominance[i, q] = s

            df_dom = pd.DataFrame(dominance, index=alts, columns=alts)
            st.subheader("Dominance matrix δ(i,q)")
            st.dataframe(df_numeric_round(df_dom, 6), use_container_width=True)

            # 3.6 Overall superiority Φ_i = Σ_q δ(i,q)
            Phi = dominance.sum(axis=1)
            df_phi = pd.DataFrame({
                "Alternative": alts,
                "Φ (overall superiority)": Phi
            })
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

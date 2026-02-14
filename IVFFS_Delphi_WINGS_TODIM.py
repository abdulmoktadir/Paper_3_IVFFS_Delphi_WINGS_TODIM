import streamlit as st
import numpy as np
import pandas as pd
import math
import graphviz
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
import io
import base64

# =========================================================
# IVFFS REPRESENTATION
#   IVFFS = ([muL, muU], [nuL, nuU])
# Fermatean-style power r = 3 is used in aggregation.
# =========================================================

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def format_ivffs(v):
    (muL, muU), (nuL, nuU) = v
    return f"([{muL:.6f},{muU:.6f}],[{nuL:.6f},{nuU:.6f}])"

def ivffs_to_row(v):
    (muL, muU), (nuL, nuU) = v
    return {"muL": muL, "muU": muU, "nuL": nuL, "nuU": nuU}

def parse_ivffs(s: str):
    # expects like: ([0.10,0.15],[0.90,0.95])
    try:
        ss = s.strip().replace(" ", "")
        ss = ss.replace("(", "").replace(")", "")
        # split into two bracket groups
        # ([a,b],[c,d])
        left = ss.split("],[")[0].replace("[", "").replace("]", "")
        right = ss.split("],[")[1].replace("[", "").replace("]", "")
        a, b = [float(x) for x in left.split(",")]
        c, d = [float(x) for x in right.split(",")]
        return ([clamp01(a), clamp01(b)], [clamp01(c), clamp01(d)])
    except Exception:
        return None

# =========================================================
# Dombi Aggregation (matches your Excel-style structure)
#
# Membership (S_Dombi):
#   mu = ( 1 - 1/(1 + ( Σ wi * ((mu_i^r)/(1-mu_i^r))^p )^(1/p) )) )^(1/r)
#
# Non-membership (T_Dombi):
#   nu = ( 1/(1 + ( Σ wi * (((1-nu_i^r)/(nu_i^r))^p ) )^(1/p) )) )^(1/r)
#
# with r = 3 (Fermatean) and p = Dombi parameter.
# =========================================================

def safe_pow(x: float, a: float) -> float:
    # protects 0^0
    if x == 0.0 and a == 0.0:
        return 1.0
    return float(x) ** float(a)

def dombi_S_membership(values, weights, p: float, r: float = 3.0) -> float:
    # values in [0,1]
    eps = 1e-15
    p = float(p)
    acc = 0.0
    for x, w in zip(values, weights):
        x = clamp01(x)
        w = float(w)
        xr = safe_pow(x, r)
        denom = max(eps, 1.0 - xr)
        ratio = xr / denom  # (x^r)/(1-x^r)
        acc += w * safe_pow(ratio, p)
    inner = safe_pow(max(acc, 0.0), 1.0 / p)
    res = 1.0 - 1.0 / (1.0 + inner)
    return safe_pow(clamp01(res), 1.0 / r)

def dombi_T_nonmembership(values, weights, p: float, r: float = 3.0) -> float:
    eps = 1e-15
    p = float(p)
    acc = 0.0
    for x, w in zip(values, weights):
        x = clamp01(x)
        w = float(w)
        xr = safe_pow(x, r)
        denom = max(eps, xr)
        ratio = (1.0 - xr) / denom  # (1-x^r)/(x^r)
        acc += w * safe_pow(ratio, p)
    inner = safe_pow(max(acc, 0.0), 1.0 / p)
    res = 1.0 / (1.0 + inner)
    return safe_pow(clamp01(res), 1.0 / r)

def aggregate_ivffs_dombi(ivffs_list, weights, p: float, r: float = 3.0):
    # ivffs_list: [ ([muL,muU],[nuL,nuU]) ... ]
    muL_list = [v[0][0] for v in ivffs_list]
    muU_list = [v[0][1] for v in ivffs_list]
    nuL_list = [v[1][0] for v in ivffs_list]
    nuU_list = [v[1][1] for v in ivffs_list]

    muL = dombi_S_membership(muL_list, weights, p, r)
    muU = dombi_S_membership(muU_list, weights, p, r)
    nuL = dombi_T_nonmembership(nuL_list, weights, p, r)
    nuU = dombi_T_nonmembership(nuU_list, weights, p, r)

    return ([muL, muU], [nuL, nuU])

# =========================================================
# Defuzz / Score for IVFFS (for matrix computations)
# You can swap this if your paper uses a different score.
# This one is consistent with Fermatean idea:
#   score = avg(mu)^r - avg(nu)^r
# =========================================================

def ivffs_score(v, r: float = 3.0) -> float:
    (muL, muU), (nuL, nuU) = v
    mu = (muL + muU) / 2.0
    nu = (nuL + nuU) / 2.0
    return safe_pow(mu, r) - safe_pow(nu, r)

# =========================================================
# IVFFS–TODIM Linguistic Scale (YOUR PROVIDED VALUES)
# =========================================================

TODIM_LINGUISTIC = {
    "VP": ([0.10, 0.15], [0.90, 0.95]),
    "P":  ([0.20, 0.25], [0.80, 0.85]),
    "MP": ([0.30, 0.35], [0.70, 0.75]),
    "F":  ([0.50, 0.55], [0.40, 0.45]),
    "MG": ([0.70, 0.75], [0.30, 0.35]),
    "G":  ([0.80, 0.85], [0.20, 0.25]),
    "VG": ([0.90, 0.95], [0.10, 0.15]),
}

TODIM_FULL = {
    "VP": "Very Poor",
    "P": "Poor",
    "MP": "Medium Poor",
    "F": "Fair",
    "MG": "Medium Good",
    "G": "Good",
    "VG": "Very Good",
}

# =========================================================
# Helper: safe dataframe display (prevents Streamlit styler crash)
# =========================================================

def dataframe_numeric_format(df: pd.DataFrame, precision: int = 6):
    df2 = df.copy()
    num_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) == 0:
        st.dataframe(df2, use_container_width=True, hide_index=True)
        return
    fmt = {c: f"{{:.{precision}f}}" for c in num_cols}
    st.dataframe(df2.style.format(fmt), use_container_width=True, hide_index=True)

# =========================================================
# IVFFS–WINGS Module (DEMATEL-style core for TI/TR)
# Uses the SAME Dombi aggregation as TODIM.
# =========================================================

WINGS_STRENGTH_TERMS = {
    # you can replace with your paper’s IVFFS scale for strengths
    "VLR": ([0.10, 0.15], [0.80, 0.85]),
    "LR":  ([0.20, 0.25], [0.70, 0.75]),
    "MR":  ([0.50, 0.55], [0.40, 0.45]),
    "HR":  ([0.70, 0.75], [0.30, 0.35]),
    "VHR": ([0.90, 0.95], [0.10, 0.15]),
}

WINGS_INFLUENCE_TERMS = {
    # you can replace with your paper’s IVFFS scale for influence
    "ELI": ([0.10, 0.15], [0.80, 0.85]),
    "VLI": ([0.20, 0.25], [0.70, 0.75]),
    "LI":  ([0.30, 0.35], [0.60, 0.65]),
    "MI":  ([0.50, 0.55], [0.40, 0.45]),
    "HI":  ([0.70, 0.75], [0.30, 0.35]),
    "VHI": ([0.80, 0.85], [0.20, 0.25]),
    "EHI": ([0.90, 0.95], [0.10, 0.15]),
}

WINGS_FULL = {
    "VLR": "Very Low Relevance",
    "LR": "Low Relevance",
    "MR": "Medium Relevance",
    "HR": "High Relevance",
    "VHR": "Very High Relevance",
    "ELI": "Extremely Low Influence",
    "VLI": "Very Low Influence",
    "LI": "Low Influence",
    "MI": "Medium Influence",
    "HI": "High Influence",
    "VHI": "Very High Influence",
    "EHI": "Extremely High Influence",
}

def wings_compute_total_relation(Z: np.ndarray) -> np.ndarray:
    n = Z.shape[0]
    I = np.eye(n)
    try:
        T = Z @ np.linalg.inv(I - Z)
    except np.linalg.LinAlgError:
        T = Z @ np.linalg.pinv(I - Z)
    return T

def ivffs_wings_module():
    st.header("📌 IVFFS–WINGS")
    st.caption("Dombi aggregation (same as your Excel-style formula) + DEMATEL-like TI/TR outputs.")

    with st.expander("IVFFS Linguistic Terms (WINGS)"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Strength terms**")
            dfS = pd.DataFrame([
                {"Abbr": k, "Meaning": WINGS_FULL.get(k, ""), "IVFFS": format_ivffs(WINGS_STRENGTH_TERMS[k])}
                for k in WINGS_STRENGTH_TERMS
            ])
            st.dataframe(dfS, use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**Influence terms**")
            dfI = pd.DataFrame([
                {"Abbr": k, "Meaning": WINGS_FULL.get(k, ""), "IVFFS": format_ivffs(WINGS_INFLUENCE_TERMS[k])}
                for k in WINGS_INFLUENCE_TERMS
            ])
            st.dataframe(dfI, use_container_width=True, hide_index=True)

    with st.sidebar:
        st.subheader("⚙️ WINGS Settings")
        n_components = st.number_input("Number of Components", min_value=2, max_value=25, value=4, step=1, key="w_ncomp2")
        n_experts = st.number_input("Number of Experts", min_value=1, max_value=15, value=2, step=1, key="w_nexp2")
        dombi_p = st.number_input("Dombi parameter p", min_value=1.0, value=96.0, step=1.0, key="w_dombi_p")

        comps = []
        for i in range(n_components):
            comps.append(st.text_input(f"Component {i+1}", value=f"C{i+1}", key=f"w_compname_{i}"))

        expert_w = None
        if n_experts == 1:
            expert_w = [1.0]
        else:
            st.markdown("---")
            st.markdown("**Expert weights (sum=1)**")
            ws = []
            for i in range(n_experts):
                ws.append(st.number_input(f"E{i+1}", min_value=0.0, max_value=1.0,
                                          value=round(1/n_experts, 6), step=0.01,
                                          format="%.6f", key=f"w_expw_{i}"))
            if not np.isclose(sum(ws), 1.0):
                st.error(f"Expert weights must sum to 1.0 (now {sum(ws):.6f})")
                st.stop()
            expert_w = ws

    # session init
    if "wings_data" not in st.session_state:
        st.session_state.wings_data = {}

    for e in range(n_experts):
        if e not in st.session_state.wings_data:
            st.session_state.wings_data[e] = {
                "strengths": ["MR"] * n_components,
                "influences": [["ELI"] * n_components for _ in range(n_components)]
            }

    tabs = st.tabs([f"Expert {i+1}" for i in range(n_experts)])
    for e in range(n_experts):
        with tabs[e]:
            st.markdown("### Strengths (diagonal)")
            cols = st.columns(n_components)
            for i in range(n_components):
                cur = st.session_state.wings_data[e]["strengths"][i]
                with cols[i]:
                    st.session_state.wings_data[e]["strengths"][i] = st.selectbox(
                        comps[i],
                        options=list(WINGS_STRENGTH_TERMS.keys()),
                        index=list(WINGS_STRENGTH_TERMS.keys()).index(cur),
                        key=f"w_str_{e}_{i}"
                    )

            st.markdown("### Influence matrix (row influences column)")
            for i in range(n_components):
                row_cols = st.columns(n_components)
                for j in range(n_components):
                    with row_cols[j]:
                        if i == j:
                            st.markdown("—")
                        else:
                            cur = st.session_state.wings_data[e]["influences"][i][j]
                            st.session_state.wings_data[e]["influences"][i][j] = st.selectbox(
                                f"{comps[i]}→{comps[j]}",
                                options=list(WINGS_INFLUENCE_TERMS.keys()),
                                index=list(WINGS_INFLUENCE_TERMS.keys()).index(cur),
                                key=f"w_inf_{e}_{i}_{j}"
                            )

    if st.button("🚀 Run IVFFS–WINGS", type="primary", use_container_width=True):
        with st.spinner("Computing WINGS..."):
            # 1) Aggregate IVFFS SIDRM using Dombi (same aggregation for all cells)
            agg_sidrm = [[None for _ in range(n_components)] for _ in range(n_components)]
            for i in range(n_components):
                for j in range(n_components):
                    iv_list = []
                    for e in range(n_experts):
                        if i == j:
                            term = st.session_state.wings_data[e]["strengths"][i]
                            iv_list.append(WINGS_STRENGTH_TERMS[term])
                        else:
                            term = st.session_state.wings_data[e]["influences"][i][j]
                            iv_list.append(WINGS_INFLUENCE_TERMS[term])
                    agg_sidrm[i][j] = aggregate_ivffs_dombi(iv_list, expert_w, p=dombi_p, r=3.0)

            # show aggregated IVFFS matrix
            df_agg = pd.DataFrame(index=comps, columns=comps, dtype=object)
            for i in range(n_components):
                for j in range(n_components):
                    df_agg.iloc[i, j] = format_ivffs(agg_sidrm[i][j])

            st.subheader("Aggregated IVFFS SIDRM (Dombi)")
            st.dataframe(df_agg, use_container_width=True)

            # 2) Convert to crisp matrix using score
            D = np.zeros((n_components, n_components))
            for i in range(n_components):
                for j in range(n_components):
                    D[i, j] = ivffs_score(agg_sidrm[i][j], r=3.0)

            # 3) Normalize (DEMATEL standard): Z = D / max(row_sum)
            row_sums = np.sum(np.abs(D), axis=1)
            s = np.max(row_sums) if np.max(row_sums) != 0 else 1.0
            Z = D / s

            # 4) Total relation
            T = wings_compute_total_relation(Z)

            TI = np.sum(T, axis=1)
            TR = np.sum(T, axis=0)
            ENG = TI + TR
            ROLE = TI - TR
            EV = np.sqrt(ENG**2 + ROLE**2)
            W = EV / np.sum(EV) if np.sum(EV) != 0 else np.zeros_like(EV)

            out = pd.DataFrame({
                "Component": comps,
                "TI": TI,
                "TR": TR,
                "Engagement": ENG,
                "Role": ROLE,
                "Expected value": EV,
                "Weight": W
            })

            st.subheader("TI / TR / Engagement / Role / Expected value / Weight")
            dataframe_numeric_format(out, precision=6)

            # optional: cause/effect
            out2 = out.copy()
            out2["Type"] = np.where(out2["Role"] > 0, "Cause", "Effect")
            st.subheader("Cause–Effect")
            st.dataframe(out2, use_container_width=True, hide_index=True)

# =========================================================
# IVFFS–TODIM Module
# Uses the SAME Dombi aggregation as WINGS.
# =========================================================

def todim_normalize(X: pd.DataFrame, crit_types: list[str]) -> pd.DataFrame:
    """
    TODIM usually works on normalized crisp scores.
    We'll normalize each criterion to [0,1] by min-max on crisp values.
    For cost criteria: invert after normalization.
    """
    Xn = X.copy()
    for j, col in enumerate(X.columns):
        v = X[col].values.astype(float)
        vmin, vmax = np.min(v), np.max(v)
        if vmax - vmin == 0:
            nv = np.zeros_like(v)
        else:
            nv = (v - vmin) / (vmax - vmin)

        if crit_types[j].lower().startswith("c"):
            nv = 1.0 - nv
        Xn[col] = nv
    return Xn

def todim_dominance(Xn: pd.DataFrame, weights: list[float], theta: float = 1.0, alpha: float = 1.0):
    """
    Classic TODIM dominance with prospect theory shape.
    For each pair (i,k):
      δ(i,k)= Σ_j φ_j(i,k)
    where:
      if xij >= xkj:  φ =  (w_j/w_r) * (xij-xkj)^alpha
      else:           φ = -(1/theta) * (w_j/w_r) * (xkj-xij)^alpha
    """
    A = Xn.index.tolist()
    m = len(A)
    n = Xn.shape[1]

    w = np.array(weights, dtype=float)
    wr = np.max(w) if np.max(w) != 0 else 1.0
    wrel = w / wr

    Delta = np.zeros((m, m))
    for i in range(m):
        for k in range(m):
            if i == k:
                continue
            s = 0.0
            for j in range(n):
                d = Xn.iloc[i, j] - Xn.iloc[k, j]
                if d >= 0:
                    s += wrel[j] * (abs(d) ** alpha)
                else:
                    s -= (1.0 / max(theta, 1e-12)) * wrel[j] * (abs(d) ** alpha)
            Delta[i, k] = s

    # global value
    Phi = np.sum(Delta, axis=1)
    # normalize Phi to [0,1]
    pmin, pmax = np.min(Phi), np.max(Phi)
    if pmax - pmin == 0:
        V = np.zeros_like(Phi)
    else:
        V = (Phi - pmin) / (pmax - pmin)
    return pd.DataFrame({"Alternative": A, "Phi": Phi, "Value": V}).sort_values("Value", ascending=False).reset_index(drop=True)

def ivffs_todim_module():
    st.header("📌 IVFFS–TODIM")
    st.caption("Expert aggregation uses your Dombi Fermatean formula (same as WINGS).")

    with st.expander("TODIM Linguistic Scale (VP…VG)"):
        df_scale = pd.DataFrame([
            {"Abbr": k, "Meaning": TODIM_FULL[k], "IVFFS": format_ivffs(TODIM_LINGUISTIC[k])}
            for k in TODIM_LINGUISTIC
        ])
        st.dataframe(df_scale, use_container_width=True, hide_index=True)

    st.subheader("Step 1: Alternatives, Criteria, Types, Weights")
    c1, c2 = st.columns(2)
    alts_in = c1.text_input("Alternatives (comma-separated)", "A1, A2, A3", key="t_alts")
    crits_in = c2.text_input("Criteria (comma-separated)", "C1, C2, C3", key="t_crits")

    alts = [x.strip() for x in alts_in.split(",") if x.strip()]
    crits = [x.strip() for x in crits_in.split(",") if x.strip()]
    if len(alts) == 0 or len(crits) == 0:
        st.warning("Enter at least 1 alternative and 1 criterion.")
        return

    if "todim_crit_df" not in st.session_state or set(st.session_state.todim_crit_df["Criterion"]) != set(crits):
        w = [round(1/len(crits), 5)] * len(crits)
        if len(crits) > 0:
            w[-1] = 1.0 - sum(w[:-1])
        st.session_state.todim_crit_df = pd.DataFrame({
            "Criterion": crits,
            "Type": ["Benefit"] * len(crits),
            "Weight": w
        })

    edited = st.data_editor(
        st.session_state.todim_crit_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Type": st.column_config.SelectboxColumn("Type", options=["Benefit", "Cost"]),
            "Weight": st.column_config.NumberColumn("Weight", format="%.5f", min_value=0.0, max_value=1.0, step=0.00001),
        },
        key="todim_crit_editor"
    )
    crit_types = edited["Type"].tolist()
    crit_w = edited["Weight"].astype(float).tolist()

    if not np.isclose(sum(crit_w), 1.0):
        st.error(f"Criteria weights must sum to 1.0 (now {sum(crit_w):.5f}).")
        return

    st.subheader("Step 2: Expert evaluations (linguistic)")
    n_exp = st.number_input("Number of experts", min_value=1, max_value=30, value=2, step=1, key="t_nexp")

    with st.sidebar:
        st.subheader("⚙️ TODIM Settings")
        dombi_p = st.number_input("Dombi parameter p (same aggregation)", min_value=1.0, value=96.0, step=1.0, key="t_dombi_p")
        theta = st.number_input("Loss attenuation θ", min_value=0.01, value=1.0, step=0.05, key="t_theta")
        alpha = st.number_input("Shape α", min_value=0.1, value=1.0, step=0.1, key="t_alpha")

    st.markdown("**Expert weights (sum=1)**")
    exp_w = []
    if n_exp == 1:
        exp_w = [1.0]
        st.info("Single expert → weight = 1.0")
    else:
        cols = st.columns(n_exp)
        for i in range(n_exp):
            with cols[i]:
                exp_w.append(st.number_input(
                    f"E{i+1}",
                    min_value=0.0, max_value=1.0,
                    value=round(1/n_exp, 6),
                    step=0.01,
                    format="%.6f",
                    key=f"t_ew_{i}"
                ))
        if not np.isclose(sum(exp_w), 1.0):
            st.error(f"Expert weights must sum to 1.0 (now {sum(exp_w):.6f})")
            return

    # expert decision matrices
    if "todim_expert_dfs" not in st.session_state:
        st.session_state.todim_expert_dfs = {}

    need_reset = (
        len(st.session_state.todim_expert_dfs) != n_exp
        or (n_exp > 0 and (
            set(st.session_state.todim_expert_dfs.get(0, pd.DataFrame()).index) != set(alts)
            or set(st.session_state.todim_expert_dfs.get(0, pd.DataFrame()).columns) != set(crits)
        ))
    )
    if need_reset:
        st.session_state.todim_expert_dfs = {i: pd.DataFrame("F", index=alts, columns=crits) for i in range(n_exp)}

    tabs = st.tabs([f"Expert {i+1}" for i in range(n_exp)])
    for i, tab in enumerate(tabs):
        with tab:
            st.session_state.todim_expert_dfs[i] = st.data_editor(
                st.session_state.todim_expert_dfs[i],
                use_container_width=True,
                column_config={c: st.column_config.SelectboxColumn(c, options=list(TODIM_LINGUISTIC.keys())) for c in crits},
                key=f"t_editor_{i}"
            )

    if st.button("✅ Run IVFFS–TODIM", type="primary", use_container_width=True):
        with st.spinner("Computing TODIM..."):
            # 1) Aggregate IVFFS decision matrix using Dombi
            agg = {}
            for a in alts:
                for c in crits:
                    vals = []
                    for e in range(n_exp):
                        term = st.session_state.todim_expert_dfs[e].loc[a, c]
                        vals.append(TODIM_LINGUISTIC[term])
                    agg[(a, c)] = aggregate_ivffs_dombi(vals, exp_w, p=dombi_p, r=3.0)

            # show aggregated IVFFS matrix
            df_agg = pd.DataFrame(index=alts, columns=crits, dtype=object)
            for a in alts:
                for c in crits:
                    df_agg.loc[a, c] = format_ivffs(agg[(a, c)])
            st.subheader("Aggregated IVFFS Decision Matrix (Dombi)")
            st.dataframe(df_agg, use_container_width=True)

            # 2) Crisp score matrix
            X = pd.DataFrame(index=alts, columns=crits, dtype=float)
            for a in alts:
                for c in crits:
                    X.loc[a, c] = ivffs_score(agg[(a, c)], r=3.0)

            st.subheader("Crisp Score Matrix (from IVFFS)")
            dataframe_numeric_format(X.reset_index().rename(columns={"index":"Alternative"}), precision=6)

            # 3) Normalize for TODIM
            Xn = todim_normalize(X, crit_types)
            st.subheader("Normalized Matrix (TODIM)")
            dataframe_numeric_format(Xn.reset_index().rename(columns={"index":"Alternative"}), precision=6)

            # 4) TODIM dominance + ranking
            res = todim_dominance(Xn, crit_w, theta=theta, alpha=alpha)
            res["Rank"] = np.arange(1, len(res) + 1)

            st.subheader("TODIM Results")
            dataframe_numeric_format(res, precision=6)

# =========================================================
# MAIN APP NAVIGATION (TWO MODULES)
# =========================================================

def main():
    st.set_page_config(page_title="IVFFS Toolkit (WINGS + TODIM)", layout="wide")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a Module", ["IVFFS–WINGS", "IVFFS–TODIM"])

    if page == "IVFFS–WINGS":
        ivffs_wings_module()
    else:
        ivffs_todim_module()

if __name__ == "__main__":
    main()

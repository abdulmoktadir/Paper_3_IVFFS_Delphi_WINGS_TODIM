import streamlit as st
import numpy as np
import pandas as pd
import graphviz
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
import io
import base64
import matplotlib.pyplot as plt

# =========================================================
# Helpers: safe numeric styling (fixes Streamlit Styler crash)
# =========================================================
def st_dataframe_numeric_format(df: pd.DataFrame, precision: int = 6, **kwargs):
    """Safely format only numeric cols to avoid Streamlit pandas Styler ValueError."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) == 0:
        st.dataframe(df, **kwargs)
        return
    fmt = {c: f"{{:.{precision}f}}" for c in num_cols}
    st.dataframe(df.style.format(fmt, na_rep=""), **kwargs)

# =========================================================
# IVFFS representation (Interval-Valued Fermatean Fuzzy Set)
#   We store: mu = [mu_l, mu_u], nu = [nu_l, nu_u]
#   Fermatean constraint: mu^3 + nu^3 <= 1 (typically)
# =========================================================
def ivffs(mu_l, mu_u, nu_l, nu_u):
    return {"mu_l": float(mu_l), "mu_u": float(mu_u), "nu_l": float(nu_l), "nu_u": float(nu_u)}

def ivffs_str(x, nd=6):
    return f"([ {x['mu_l']:.{nd}f}, {x['mu_u']:.{nd}f} ], [ {x['nu_l']:.{nd}f}, {x['nu_u']:.{nd}f} ])"

# =========================================================
# Excel-matching IVFFS Dombi aggregation (Fermatean power = 3)
#
# Your Excel pattern for membership (for one bound):
#   (1 - (1/(1 + ( Σ w_i * ((μ_i^3/(1-μ_i^3))^λ) )^(1/λ) ))) )^(1/3)
#
# and for non-membership:
#   (1/(1 + ( Σ w_i * (((1-ν_i^3)/ν_i^3)^λ) )^(1/λ) )))^(1/3)
#
# This matches how your sheet aggregates each parameter.
# =========================================================
_EPS = 1e-12

def _safe_clip01(x):
    return float(min(1.0 - _EPS, max(_EPS, float(x))))

def dombi_mu_agg(mu_list, w_list, lam):
    lam = float(lam)
    s = 0.0
    for mu, w in zip(mu_list, w_list):
        mu = _safe_clip01(mu)
        t = (mu**3) / max(_EPS, (1.0 - mu**3))
        s += float(w) * (t**lam)
    core = (s ** (1.0 / lam)) if s > 0 else 0.0
    val = 1.0 - (1.0 / (1.0 + core))
    val = max(0.0, min(1.0, val))
    return val ** (1.0 / 3.0)

def dombi_nu_agg(nu_list, w_list, lam):
    lam = float(lam)
    s = 0.0
    for nu, w in zip(nu_list, w_list):
        nu = _safe_clip01(nu)
        t = (1.0 - nu**3) / max(_EPS, (nu**3))
        s += float(w) * (t**lam)
    core = (s ** (1.0 / lam)) if s > 0 else 0.0
    val = 1.0 / (1.0 + core)
    val = max(0.0, min(1.0, val))
    return val ** (1.0 / 3.0)

def ivffs_dombi_aggregate(expert_ivffs_list, expert_weights, lam):
    """Aggregate IVFFS across experts (Excel-matching) for one alt-criterion cell."""
    mu_l_list = [x["mu_l"] for x in expert_ivffs_list]
    mu_u_list = [x["mu_u"] for x in expert_ivffs_list]
    nu_l_list = [x["nu_l"] for x in expert_ivffs_list]
    nu_u_list = [x["nu_u"] for x in expert_ivffs_list]

    agg_mu_l = dombi_mu_agg(mu_l_list, expert_weights, lam)
    agg_mu_u = dombi_mu_agg(mu_u_list, expert_weights, lam)
    agg_nu_l = dombi_nu_agg(nu_l_list, expert_weights, lam)
    agg_nu_u = dombi_nu_agg(nu_u_list, expert_weights, lam)

    return ivffs(agg_mu_l, agg_mu_u, agg_nu_l, agg_nu_u)

# =========================================================
# Expected value (matches your Excel pattern seen in sheet)
# Example you shared / same structure in your workbook:
#   EV = ( ((mu_l^3 + mu_u^3 - nu_l^3 - nu_u^3)/2) + 1 ) / 2
# =========================================================
def ivffs_expected_value(x):
    mu_l, mu_u = x["mu_l"], x["mu_u"]
    nu_l, nu_u = x["nu_l"], x["nu_u"]
    core = (mu_l**3 + mu_u**3 - nu_l**3 - nu_u**3) / 2.0
    ev = (core + 1.0) / 2.0
    return float(ev)

# =========================================================
# TODIM (crisp on EV after aggregation)
# - Normalize per criterion
#   Benefit: x / max(x)
#   Cost:    min(x) / x
# =========================================================
def normalize_matrix_todim(ev_df: pd.DataFrame, crit_types: list):
    norm = ev_df.copy()
    for j, c in enumerate(ev_df.columns):
        col = ev_df[c].astype(float).values
        if crit_types[j].lower().startswith("b"):  # Benefit
            mx = np.max(col) if np.max(col) != 0 else 1.0
            norm[c] = col / mx
        else:  # Cost
            mn = np.min(col)
            # avoid division by zero
            norm[c] = np.array([mn / v if v != 0 else 0.0 for v in col], dtype=float)
    return norm

def todim_rank(norm_df: pd.DataFrame, weights: list, theta: float = 1.0):
    """
    Classic TODIM dominance:
      reference criterion r = argmax(w)
      phi_j(i,k) = (w_j/w_r)*(d)               if d >= 0
                 = -(w_r/w_j)*(|d|)/theta      if d < 0
      delta(i,k) = Σ_j phi_j(i,k)
      xi_i = (Σ_k delta(i,k) - min) / (max - min)
    """
    W = np.array(weights, dtype=float)
    W = W / W.sum() if W.sum() != 0 else W
    r = int(np.argmax(W))
    wr = W[r] if W[r] != 0 else 1.0

    X = norm_df.values.astype(float)
    n_alt, n_crit = X.shape

    delta = np.zeros((n_alt, n_alt), dtype=float)
    for i in range(n_alt):
        for k in range(n_alt):
            if i == k:
                continue
            s = 0.0
            for j in range(n_crit):
                d = X[i, j] - X[k, j]
                wj = W[j] if W[j] != 0 else _EPS
                if d >= 0:
                    s += (wj / wr) * d
                else:
                    s += - (wr / wj) * (abs(d) / max(_EPS, theta))
            delta[i, k] = s

    S = delta.sum(axis=1)
    mn, mx = float(S.min()), float(S.max())
    if abs(mx - mn) < 1e-12:
        xi = np.ones_like(S)
    else:
        xi = (S - mn) / (mx - mn)

    return delta, S, xi

# =========================================================
# Linguistic scale for IVFFS-TODIM (you can edit here)
# If your Excel scale differs, paste your exact values below.
# Format: "TERM": ivffs(mu_l, mu_u, nu_l, nu_u)
# =========================================================
IVFFS_TODIM_SCALE = {
    "VP": ivffs(0.00, 0.10, 0.80, 0.90),
    "P":  ivffs(0.10, 0.30, 0.70, 0.80),
    "MP": ivffs(0.30, 0.50, 0.50, 0.70),
    "F":  ivffs(0.50, 0.70, 0.30, 0.50),
    "MG": ivffs(0.70, 0.85, 0.15, 0.30),
    "G":  ivffs(0.85, 0.95, 0.05, 0.15),
    "VG": ivffs(0.95, 1.00, 0.00, 0.05),
}
IVFFS_TODIM_FULL = {
    "VP": "Very Poor", "P": "Poor", "MP": "Medium Poor", "F": "Fair",
    "MG": "Medium Good", "G": "Good", "VG": "Very Good"
}

# =========================================================
# MODULE 1: IVFFS-WINGS (simple expected-value WINGS-style)
# NOTE: If your paper uses a different IVFFS-WINGS pipeline,
# tell me the exact sheet name + key formulas and I’ll match it.
# =========================================================
def ivffs_wings_module():
    st.header("🪽 IVFFS–WINGS (Expected-Value Based)")
    st.caption("This module computes TI/TR/Engagement/Role from an EV matrix (crisp after IVFFS expected value).")

    n = st.number_input("Number of factors", min_value=2, max_value=25, value=5, step=1)
    names = [st.text_input(f"Factor {i+1} name", value=f"F{i+1}") for i in range(int(n))]

    st.markdown("### Direct-influence matrix (Expected values, 0–1)")
    st.info("If you want full IVFFS (interval) inputs for WINGS too, tell me your Excel sheet formulas and I will update it.")

    Z = pd.DataFrame(0.0, index=names, columns=names)
    Z = st.data_editor(Z, use_container_width=True, key="wings_Z")

    # Normalize like DEMATEL/WINGS: Z / s where s = max(max row sum, max col sum)
    row_sum = Z.sum(axis=1).values
    col_sum = Z.sum(axis=0).values
    s = float(max(row_sum.max(), col_sum.max(), 1e-12))
    ZN = Z / s

    I = np.eye(len(names))
    try:
        T = ZN.values @ np.linalg.inv(I - ZN.values)
    except np.linalg.LinAlgError:
        st.error("Matrix inversion failed (I - Z is singular). Try smaller values or adjust inputs.")
        return

    TI = T.sum(axis=1)
    TR = T.sum(axis=0)
    engagement = TI + TR
    role = TI - TR

    out = pd.DataFrame({
        "Factor": names,
        "TI": TI,
        "TR": TR,
        "Engagement": engagement,
        "Role": role,
    })

    # Example “Expected value” and “Weight” (normalize Engagement)
    out["Expected value"] = (out["Engagement"] - out["Engagement"].min()) / (out["Engagement"].max() - out["Engagement"].min() + 1e-12)
    out["Weight"] = out["Expected value"] / (out["Expected value"].sum() + 1e-12)

    st.markdown("### Results")
    st_dataframe_numeric_format(out, precision=6, use_container_width=True, hide_index=True)

# =========================================================
# MODULE 2: IVFFS–TODIM (Excel-matching aggregation)
# =========================================================
def ivffs_todim_module():
    st.header("📊 IVFFS–TODIM (Excel-matching Dombi aggregation)")
    st.caption("Aggregation is Dombi + Fermatean cube (matches your Excel formula). Defuzzification (EV) happens after aggregation.")

    with st.expander("Linguistic Scale (edit in code if needed)", expanded=False):
        scale_df = pd.DataFrame([
            {"Abbr": k, "Meaning": IVFFS_TODIM_FULL.get(k, ""), "IVFFS": ivffs_str(v)}
            for k, v in IVFFS_TODIM_SCALE.items()
        ])
        st.dataframe(scale_df, hide_index=True, use_container_width=True)

    c1, c2 = st.columns(2)
    alts_in = c1.text_input("Alternatives (comma-separated)", "A1, A2, A3")
    crits_in = c2.text_input("Criteria (comma-separated)", "C1, C2, C3")

    alts = [a.strip() for a in alts_in.split(",") if a.strip()]
    crits = [c.strip() for c in crits_in.split(",") if c.strip()]
    if not alts or not crits:
        st.warning("Please input at least one alternative and one criterion.")
        return

    # Criteria table
    if "todim_crit_df" not in st.session_state or set(st.session_state.todim_crit_df["Criterion"]) != set(crits):
        w = [round(1 / len(crits), 6)] * len(crits)
        w[-1] = 1.0 - sum(w[:-1])
        st.session_state.todim_crit_df = pd.DataFrame({
            "Criterion": crits,
            "Type": ["Benefit"] * len(crits),
            "Weight": w
        })

    st.markdown("### Criteria types & weights")
    crit_df = st.data_editor(
        st.session_state.todim_crit_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Type": st.column_config.SelectboxColumn("Type", options=["Benefit", "Cost"]),
            # ✅ allows 5-digit weights
            "Weight": st.column_config.NumberColumn("Weight", format="%.5f", min_value=0.0, max_value=1.0, step=0.00001),
        },
        key="todim_crit_editor"
    )

    crit_types = crit_df["Type"].tolist()
    crit_w = crit_df["Weight"].astype(float).tolist()

    if not np.isclose(sum(crit_w), 1.0):
        st.error(f"Criteria weights must sum to 1.0 (now: {sum(crit_w):.6f})")
        return

    st.markdown("### Experts")
    n_exp = st.number_input("Number of experts", min_value=1, max_value=30, value=3, step=1)

    exp_weights = []
    if n_exp == 1:
        exp_weights = [1.0]
        st.info("Single expert → weight=1.0")
    else:
        cols = st.columns(int(n_exp))
        for i in range(int(n_exp)):
            with cols[i]:
                exp_weights.append(
                    st.number_input(
                        f"E{i+1}",
                        min_value=0.0, max_value=1.0,
                        value=round(1 / n_exp, 6),
                        format="%.6f",
                        step=0.01
                    )
                )
        if not np.isclose(sum(exp_weights), 1.0):
            st.error(f"Expert weights must sum to 1.0 (now: {sum(exp_weights):.6f})")
            return

    lam = st.number_input("Dombi parameter (λ)", min_value=0.01, value=3.0, step=0.1)
    theta = st.number_input("TODIM attenuation factor (θ)", min_value=0.01, value=1.0, step=0.1)

    # Expert decision tables
    if "todim_exp_tables" not in st.session_state:
        st.session_state.todim_exp_tables = {}

    need_reset = (
        len(st.session_state.todim_exp_tables) != int(n_exp)
        or (int(n_exp) > 0 and (
            set(st.session_state.todim_exp_tables.get(0, pd.DataFrame()).index) != set(alts)
            or set(st.session_state.todim_exp_tables.get(0, pd.DataFrame()).columns) != set(crits)
        ))
    )
    if need_reset:
        st.session_state.todim_exp_tables = {
            i: pd.DataFrame("F", index=alts, columns=crits) for i in range(int(n_exp))
        }

    st.markdown("### Expert evaluations (linguistic)")
    tabs = st.tabs([f"Expert {i+1}" for i in range(int(n_exp))])
    for i, tab in enumerate(tabs):
        with tab:
            st.session_state.todim_exp_tables[i] = st.data_editor(
                st.session_state.todim_exp_tables[i],
                use_container_width=True,
                column_config={c: st.column_config.SelectboxColumn(c, options=list(IVFFS_TODIM_SCALE.keys())) for c in crits},
                key=f"todim_editor_{i}"
            )

    if st.button("✅ Run IVFFS–TODIM", type="primary", use_container_width=True):
        # 1) Aggregate IVFFS per alt-crit (Excel-matching)
        agg = {}
        for a in alts:
            for c in crits:
                cell_ivffs = []
                for e in range(int(n_exp)):
                    term = st.session_state.todim_exp_tables[e].loc[a, c]
                    cell_ivffs.append(IVFFS_TODIM_SCALE[term])
                agg[(a, c)] = ivffs_dombi_aggregate(cell_ivffs, exp_weights, lam)

        # show aggregated
        agg_df = pd.DataFrame(index=alts, columns=crits, dtype=object)
        for a in alts:
            for c in crits:
                agg_df.loc[a, c] = ivffs_str(agg[(a, c)], nd=6)

        st.markdown("#### 1) Aggregated IVFFS matrix (NO defuzz yet)")
        st.dataframe(agg_df, use_container_width=True)

        # 2) Expected values
        ev_df = pd.DataFrame(index=alts, columns=crits, dtype=float)
        for a in alts:
            for c in crits:
                ev_df.loc[a, c] = ivffs_expected_value(agg[(a, c)])

        st.markdown("#### 2) Expected value matrix (after aggregation)")
        st_dataframe_numeric_format(ev_df.reset_index().rename(columns={"index": "Alternative"}), precision=6, use_container_width=True, hide_index=True)

        # 3) Normalize for TODIM
        norm_df = normalize_matrix_todim(ev_df, crit_types)
        st.markdown("#### 3) Normalized matrix (Benefit: x/max, Cost: min/x)")
        st_dataframe_numeric_format(norm_df.reset_index().rename(columns={"index": "Alternative"}), precision=6, use_container_width=True, hide_index=True)

        # 4) TODIM dominance + final score
        delta, S, xi = todim_rank(norm_df, crit_w, theta=theta)

        dom_df = pd.DataFrame(delta, index=alts, columns=alts)
        st.markdown("#### 4) Dominance matrix δ(i,k)")
        st_dataframe_numeric_format(dom_df.reset_index().rename(columns={"index": "Alternative"}), precision=6, use_container_width=True, hide_index=True)

        res = pd.DataFrame({
            "Alternative": alts,
            "Sum dominance": S,
            "Final TODIM value (ξ)": xi
        }).sort_values("Final TODIM value (ξ)", ascending=False).reset_index(drop=True)
        res["Rank"] = np.arange(1, len(res) + 1)

        st.markdown("#### 5) Final ranking")
        st_dataframe_numeric_format(res, precision=6, use_container_width=True, hide_index=True)

# =========================================================
# Main app (two modules)
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

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
# IVFFS REPRESENTATION
#   IVFFS = ([a,b], [c,d])
#   [a,b] = interval membership
#   [c,d] = interval non-membership
# =========================================================

def clamp01(x: float, eps: float = 1e-12) -> float:
    # keep away from 0 and 1 to avoid division blowups like x^3/(1-x^3)
    return float(min(1.0 - eps, max(eps, x)))

def fmt_ivffs(v):
    (ab, cd) = v
    return f"([{ab[0]:.5f},{ab[1]:.5f}],[{cd[0]:.5f},{cd[1]:.5f}])"

# =========================================================
# Linguistic scale for IVFFS-TODIM (YOUR PROVIDED SCALE)
# =========================================================
IVFFS_LINGUISTIC = {
    "VP": ([0.10, 0.15], [0.90, 0.95]),
    "P" : ([0.20, 0.25], [0.80, 0.85]),
    "MP": ([0.30, 0.35], [0.70, 0.75]),
    "F" : ([0.50, 0.55], [0.40, 0.45]),
    "MG": ([0.70, 0.75], [0.30, 0.35]),
    "G" : ([0.80, 0.85], [0.20, 0.25]),
    "VG": ([0.90, 0.95], [0.10, 0.15]),
}

IVFFS_FULL = {
    "VP":"Very Poor","P":"Poor","MP":"Medium Poor","F":"Fair",
    "MG":"Medium Good","G":"Good","VG":"Very Good"
}

# =========================================================
# IVFFS expected/crisp value (MATCHES YOUR EXCEL "Crisp Value")
# Excel: =(((a^3+b^3-c^3-d^3)/2+1))/2
# =========================================================
def ivffs_expected_value(v, r=3.0) -> float:
    (ab, cd) = v
    a, b = ab
    c, d = cd
    return (((a**r + b**r - c**r - d**r) / 2.0) + 1.0) / 2.0

# =========================================================
# Dombi-weighted IVFFS aggregation (MATCH YOUR EXCEL FORMULAS)
# r = 3 fixed (Excel uses ^3)
# p = dombi parameter (cell like $C$149)
# weights = expert weights (λ_i)
#
# Membership endpoints (a,b):
#   x = ( 1 - 1/(1 + ( Σ λ_i * ( (x_i^r)/(1-x_i^r) )^p )^(1/p) ) ) )^(1/r)
#
# Non-membership endpoints (c,d):
#   y = ( 1 / (1 + ( Σ λ_i * ( ((1-y_i^r)/(y_i^r)) )^p )^(1/p) ) ) )^(1/r)
# =========================================================
def dombi_aggregate_membership_endpoint(xs, weights, p, r=3.0):
    p = float(p)
    r = float(r)
    eps = 1e-12

    acc = 0.0
    for x, w in zip(xs, weights):
        x = clamp01(x, eps)
        xr = x**r
        ratio = xr / max(eps, (1.0 - xr))              # (x^r)/(1-x^r)
        acc += float(w) * (ratio ** p)                 # Σ λ_i * ratio^p

    inner = acc ** (1.0 / p)                           # (...)^(1/p)
    core = 1.0 - 1.0 / (1.0 + inner)                   # 1 - 1/(1+inner)
    return core ** (1.0 / r)                           # ^(1/r)

def dombi_aggregate_nonmembership_endpoint(ys, weights, p, r=3.0):
    p = float(p)
    r = float(r)
    eps = 1e-12

    acc = 0.0
    for y, w in zip(ys, weights):
        y = clamp01(y, eps)
        yr = y**r
        ratio = (1.0 - yr) / max(eps, yr)              # (1-y^r)/(y^r)
        acc += float(w) * (ratio ** p)                 # Σ λ_i * ratio^p

    inner = acc ** (1.0 / p)                           # (...)^(1/p)
    core = 1.0 / (1.0 + inner)                         # 1/(1+inner)
    return core ** (1.0 / r)                           # ^(1/r)

def ivffs_dombi_weighted_aggregate(ivffs_list, weights, p, r=3.0):
    # ivffs_list: list of ([a,b],[c,d]) from experts
    a_list = [v[0][0] for v in ivffs_list]
    b_list = [v[0][1] for v in ivffs_list]
    c_list = [v[1][0] for v in ivffs_list]
    d_list = [v[1][1] for v in ivffs_list]

    a = dombi_aggregate_membership_endpoint(a_list, weights, p, r=r)
    b = dombi_aggregate_membership_endpoint(b_list, weights, p, r=r)
    c = dombi_aggregate_nonmembership_endpoint(c_list, weights, p, r=r)
    d = dombi_aggregate_nonmembership_endpoint(d_list, weights, p, r=r)

    # keep consistency: a<=b and c<=d
    ab = [min(a, b), max(a, b)]
    cd = [min(c, d), max(c, d)]
    return (ab, cd)

# =========================================================
# Helpers
# =========================================================
def safe_df(df: pd.DataFrame):
    # Avoid Styler problems on Streamlit Cloud.
    st.dataframe(df, use_container_width=True, hide_index=True)

def word_download_link(doc: Document, filename: str):
    file_stream = io.BytesIO()
    doc.save(file_stream)
    file_stream.seek(0)
    b64 = base64.b64encode(file_stream.read()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{b64}" download="{filename}">Download Word Report</a>'
    return href

# =========================================================
# IVFFS-WINGS (crisp DEMATEL-style WINGS based on expected values)
# - Build expert SIDRM with IVFFS terms
# - Aggregate using SAME Dombi operator
# - Convert aggregated IVFFS matrix to crisp using Excel expected value
# - Run DEMATEL: Z = A / max(max_row_sum, max_col_sum), T = Z(I-Z)^-1
# - Output TI/TR/Engagement/Role + normalized Weight
# =========================================================
def ivffs_wings_module():
    st.header("📌 IVFFS–WINGS (Dombi aggregation + Excel expected value)")
    st.caption("Aggregation uses your Excel Dombi-based IVFFS operator (r=3, p = Dombi parameter).")

    with st.expander("IVFFS linguistic scale (VP…VG)"):
        scale_df = pd.DataFrame([
            {"Abbr": k, "Meaning": IVFFS_FULL[k], "IVFFS": fmt_ivffs(IVFFS_LINGUISTIC[k])}
            for k in IVFFS_LINGUISTIC
        ])
        safe_df(scale_df)

    st.subheader("Step 1: Components + Experts")
    n = st.number_input("Number of components", min_value=2, max_value=25, value=5, step=1)
    comps = []
    cols = st.columns(5) if n >= 5 else st.columns(n)
    for i in range(n):
        with cols[i % len(cols)]:
            comps.append(st.text_input(f"Component {i+1}", value=f"ESG{i+1}", key=f"ivw_c_{i}"))

    m = st.number_input("Number of experts", min_value=1, max_value=15, value=4, step=1)
    st.markdown("**Expert weights (sum = 1.00000)**")
    ew = []
    if m == 1:
        ew = [1.0]
        st.info("Single expert → weight = 1.0")
    else:
        ew_cols = st.columns(m)
        for i in range(m):
            with ew_cols[i]:
                ew.append(st.number_input(f"E{i+1}", min_value=0.0, max_value=1.0,
                                          value=round(1/m, 5), step=0.00001, format="%.5f",
                                          key=f"ivw_ew_{i}"))
        if not np.isclose(sum(ew), 1.0):
            st.error(f"Expert weights must sum to 1.00000 (now {sum(ew):.5f}).")
            return

    p = st.number_input("Dombi parameter (p)", min_value=0.01, max_value=50.0, value=0.5, step=0.01)
    st.caption("Excel uses r=3 (power 3) in the aggregation formulas; this app fixes r=3.")

    st.subheader("Step 2: Expert linguistic SIDRM (diagonal = strength, off-diagonal = influence)")
    # store per expert: n x n with abbreviations
    if "ivw_sidrm" not in st.session_state:
        st.session_state.ivw_sidrm = {}

    reset = (
        len(st.session_state.ivw_sidrm) != m
        or (m > 0 and (
            st.session_state.ivw_sidrm.get(0, pd.DataFrame()).shape != (n, n)
        ))
    )
    if reset:
        st.session_state.ivw_sidrm = {
            e: pd.DataFrame("F", index=comps, columns=comps) for e in range(m)
        }

    tabs = st.tabs([f"Expert {i+1}" for i in range(m)])
    for e, tab in enumerate(tabs):
        with tab:
            st.session_state.ivw_sidrm[e] = st.data_editor(
                st.session_state.ivw_sidrm[e],
                use_container_width=True,
                column_config={
                    c: st.column_config.SelectboxColumn(c, options=list(IVFFS_LINGUISTIC.keys()))
                    for c in comps
                },
                key=f"ivw_mat_{e}"
            )

    if st.button("✅ Run IVFFS–WINGS", type="primary", use_container_width=True):
        # 1) Aggregate IVFFS cellwise using Dombi operator
        agg_ivffs = np.empty((n, n), dtype=object)
        for i in range(n):
            for j in range(n):
                vals = []
                for e in range(m):
                    term = st.session_state.ivw_sidrm[e].iloc[i, j]
                    vals.append(IVFFS_LINGUISTIC[term])
                agg_ivffs[i, j] = ivffs_dombi_weighted_aggregate(vals, ew, p, r=3.0)

        # 2) Expected value matrix (crisp)
        A = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                A[i, j] = ivffs_expected_value(agg_ivffs[i, j], r=3.0)

        df_agg_show = pd.DataFrame([[fmt_ivffs(agg_ivffs[i, j]) for j in range(n)] for i in range(n)],
                                   index=comps, columns=comps).reset_index().rename(columns={"index":"Component"})
        st.markdown("#### Aggregated IVFFS SIDRM (Dombi)")
        safe_df(df_agg_show)

        df_A = pd.DataFrame(A, index=comps, columns=comps).reset_index().rename(columns={"index":"Component"})
        st.markdown("#### Expected value SIDRM (Excel crisp)")
        safe_df(df_A)

        # 3) DEMATEL normalization
        row_sums = np.sum(A, axis=1)
        col_sums = np.sum(A, axis=0)
        s = max(row_sums.max(), col_sums.max())
        if s == 0:
            st.error("Normalization scale is 0 (all entries are zero). Check input.")
            return
        Z = A / s

        # 4) Total relation matrix
        I = np.eye(n)
        try:
            T = Z @ np.linalg.inv(I - Z)
        except np.linalg.LinAlgError:
            # fallback to pseudo-inverse
            T = Z @ np.linalg.pinv(I - Z)

        TI = T.sum(axis=1)
        TR = T.sum(axis=0)
        ENG = TI + TR
        ROLE = TI - TR

        # 5) Weight from engagement (common in DEMATEL-style weighting)
        # you can change to abs(ROLE) if your paper uses that; engagement matches your column list.
        w = ENG / ENG.sum() if ENG.sum() != 0 else np.zeros_like(ENG)

        out = pd.DataFrame({
            "Component": comps,
            "TI": TI,
            "TR": TR,
            "Engagement (TI+TR)": ENG,
            "Role (TI-TR)": ROLE,
            "Weight": w
        })

        st.markdown("#### Results (TI / TR / Engagement / Role / Weight)")
        safe_df(out.round(6))

# =========================================================
# IVFFS-TODIM
# - Expert linguistic decision matrix
# - Aggregate with SAME Dombi operator
# - Defuzz to expected value (Excel)
# - Normalize benefit/cost
# - TODIM dominance + final value
# =========================================================
def todim_module():
    st.header("📌 IVFFS–TODIM (Dombi aggregation + Excel expected value)")
    st.caption("Uses the same IVFFS Dombi aggregation as the WINGS module.")

    with st.expander("IVFFS linguistic scale (VP…VG)"):
        scale_df = pd.DataFrame([
            {"Abbr": k, "Meaning": IVFFS_FULL[k], "IVFFS": fmt_ivffs(IVFFS_LINGUISTIC[k])}
            for k in IVFFS_LINGUISTIC
        ])
        safe_df(scale_df)

    st.subheader("Step 1: Alternatives, Criteria, Types, Weights")
    c1, c2 = st.columns(2)
    alts_in = c1.text_input("Alternatives (comma-separated)", "A1, A2, A3", key="td_alts")
    crits_in = c2.text_input("Criteria (comma-separated)", "C1, C2, C3, C4", key="td_crits")

    alts = [x.strip() for x in alts_in.split(",") if x.strip()]
    crits = [x.strip() for x in crits_in.split(",") if x.strip()]
    if len(alts) < 2 or len(crits) < 1:
        st.warning("Provide at least 2 alternatives and 1 criterion.")
        return

    if "td_crit_df" not in st.session_state or list(st.session_state.td_crit_df["Criterion"]) != crits:
        w0 = [round(1/len(crits), 5)] * len(crits)
        w0[-1] = 1.0 - sum(w0[:-1])
        st.session_state.td_crit_df = pd.DataFrame({
            "Criterion": crits,
            "Type": ["Benefit"] * len(crits),
            "Weight": w0
        })

    crit_df = st.data_editor(
        st.session_state.td_crit_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Type": st.column_config.SelectboxColumn("Type", options=["Benefit","Cost"]),
            "Weight": st.column_config.NumberColumn("Weight", format="%.5f", min_value=0.0, max_value=1.0, step=0.00001),
        },
        key="td_crit_editor"
    )

    ctype = crit_df["Type"].tolist()
    wj = crit_df["Weight"].astype(float).tolist()
    if not np.isclose(sum(wj), 1.0):
        st.error(f"Criteria weights must sum to 1.00000 (now {sum(wj):.5f}).")
        return

    st.subheader("Step 2: Experts + weights + Dombi parameter")
    m = st.number_input("Number of experts", min_value=1, max_value=30, value=4, step=1, key="td_m")
    ew = []
    if m == 1:
        ew = [1.0]
        st.info("Single expert → weight = 1.0")
    else:
        cols = st.columns(m)
        for i in range(m):
            with cols[i]:
                ew.append(st.number_input(f"E{i+1}", min_value=0.0, max_value=1.0,
                                          value=round(1/m, 5), step=0.00001, format="%.5f",
                                          key=f"td_ew_{i}"))
        if not np.isclose(sum(ew), 1.0):
            st.error(f"Expert weights must sum to 1.00000 (now {sum(ew):.5f}).")
            return

    p = st.number_input("Dombi parameter (p)", min_value=0.01, max_value=50.0, value=0.5, step=0.01, key="td_p")
    theta = st.number_input("TODIM attenuation factor (θ)", min_value=0.01, max_value=50.0, value=1.0, step=0.05, key="td_theta")
    st.caption("Aggregation uses r=3 and your Excel Dombi equations. TODIM is performed on the expected values.")

    # Expert evaluation matrices: alts x crits with linguistic keys
    if "td_exp_mats" not in st.session_state:
        st.session_state.td_exp_mats = {}

    reset = (
        len(st.session_state.td_exp_mats) != m
        or (m > 0 and (
            st.session_state.td_exp_mats.get(0, pd.DataFrame()).shape != (len(alts), len(crits))
        ))
    )
    if reset:
        st.session_state.td_exp_mats = {
            e: pd.DataFrame("F", index=alts, columns=crits) for e in range(m)
        }

    st.subheader("Step 3: Expert linguistic evaluations")
    tabs = st.tabs([f"Expert {i+1}" for i in range(m)])
    for e, tab in enumerate(tabs):
        with tab:
            st.session_state.td_exp_mats[e] = st.data_editor(
                st.session_state.td_exp_mats[e],
                use_container_width=True,
                column_config={c: st.column_config.SelectboxColumn(c, options=list(IVFFS_LINGUISTIC.keys())) for c in crits},
                key=f"td_mat_{e}"
            )

    if st.button("✅ Run IVFFS–TODIM", type="primary", use_container_width=True, key="td_run"):
        # 1) Aggregate per (alt, crit)
        agg = {}
        for a in alts:
            for c in crits:
                vals = []
                for e in range(m):
                    term = st.session_state.td_exp_mats[e].loc[a, c]
                    vals.append(IVFFS_LINGUISTIC[term])
                agg[(a, c)] = ivffs_dombi_weighted_aggregate(vals, ew, p, r=3.0)

        df_agg = pd.DataFrame({
            "Alternative": alts,
            **{c: [fmt_ivffs(agg[(a, c)]) for a in alts] for c in crits}
        })
        st.markdown("#### Aggregated IVFFS Decision Matrix (Dombi)")
        safe_df(df_agg)

        # 2) Expected/crisp matrix X (Excel)
        X = np.zeros((len(alts), len(crits)), dtype=float)
        for i, a in enumerate(alts):
            for j, c in enumerate(crits):
                X[i, j] = ivffs_expected_value(agg[(a, c)], r=3.0)

        df_X = pd.DataFrame(X, index=alts, columns=crits).reset_index().rename(columns={"index":"Alternative"})
        st.markdown("#### Expected value matrix (Excel crisp)")
        safe_df(df_X.round(6))

        # 3) Normalize per criterion
        Xn = X.copy()
        for j, c in enumerate(crits):
            col = X[:, j]
            mx = col.max()
            mn = col.min()
            if ctype[j].lower().startswith("b"):  # Benefit
                denom = mx if mx != 0 else 1.0
                Xn[:, j] = col / denom
            else:  # Cost
                denom = col.copy()
                denom[denom == 0] = 1e-12
                Xn[:, j] = (mn / denom)

        df_Xn = pd.DataFrame(Xn, index=alts, columns=crits).reset_index().rename(columns={"index":"Alternative"})
        st.markdown("#### Normalized expected values (benefit/cost)")
        safe_df(df_Xn.round(6))

        # 4) TODIM dominance
        w = np.array(wj, dtype=float)
        ref = int(np.argmax(w))
        wrel = w / w[ref] if w[ref] != 0 else w

        # range per criterion for scaling
        ranges = (Xn.max(axis=0) - Xn.min(axis=0))
        ranges[ranges == 0] = 1.0

        nA = len(alts)
        delta = np.zeros((nA, nA), dtype=float)

        for i in range(nA):
            for k in range(nA):
                if i == k:
                    continue
                s = 0.0
                for j in range(len(crits)):
                    diff = (Xn[i, j] - Xn[k, j]) / ranges[j]
                    if diff >= 0:
                        s += np.sqrt(wrel[j] * diff)
                    else:
                        s -= (1.0 / theta) * np.sqrt(wrel[j] * (-diff))
                delta[i, k] = s

        # overall value
        phi = delta.sum(axis=1)
        # normalize to [0,1]
        phi_min, phi_max = phi.min(), phi.max()
        score = (phi - phi_min) / (phi_max - phi_min) if (phi_max - phi_min) != 0 else np.zeros_like(phi)

        out = pd.DataFrame({
            "Alternative": alts,
            "Phi (sum dominance)": phi,
            "TODIM value": score
        })
        out["Rank"] = out["TODIM value"].rank(ascending=False, method="min").astype(int)
        out = out.sort_values("Rank").reset_index(drop=True)

        st.markdown("#### TODIM result")
        safe_df(out.round(6))

# =========================================================
# MAIN NAVIGATION
# =========================================================
def main():
    st.set_page_config(page_title="IVFFS Toolkit (WINGS + TODIM)", layout="wide")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a Module", ["IVFFS-WINGS", "IVFFS-TODIM"])

    if page == "IVFFS-WINGS":
        ivffs_wings_module()
    else:
        todim_module()

if __name__ == "__main__":
    main()

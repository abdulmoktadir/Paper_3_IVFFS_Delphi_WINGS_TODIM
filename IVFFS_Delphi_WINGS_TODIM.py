import streamlit as st
import numpy as np
import pandas as pd
import math
import io

# =========================================================
# Interval-Valued Fermatean Fuzzy Set (IVFFS)
#  A = ([muL, muU], [nuL, nuU]), q=3 (Fermatean)
# =========================================================

EPS = 1e-12
Q = 3  # Fermatean

def clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))

def sanitize_ivffs(muL, muU, nuL, nuU):
    """Clamp to [0,1], enforce ordering, and (soft) Fermatean feasibility mu^3 + nu^3 <= 1."""
    muL, muU = clamp01(muL), clamp01(muU)
    nuL, nuU = clamp01(nuL), clamp01(nuU)
    if muL > muU:
        muL, muU = muU, muL
    if nuL > nuU:
        nuL, nuU = nuU, nuL

    # Soft feasibility: ensure upper bounds don't violate mu^q + nu^q <= 1.
    # If violated, shrink nuU (then nuL if needed).
    def fix(mu, nu):
        s = mu**Q + nu**Q
        if s <= 1.0 + 1e-10:
            return mu, nu
        # reduce nu to satisfy: nu^Q <= 1 - mu^Q
        target = max(0.0, 1.0 - mu**Q)
        nu_new = target ** (1.0 / Q)
        return mu, nu_new

    muU, nuU = fix(muU, nuU)
    muL, nuL = fix(muL, nuL)

    # re-order again after fix
    muL, muU = min(muL, muU), max(muL, muU)
    nuL, nuU = min(nuL, nuU), max(nuL, nuU)

    return (muL, muU, nuL, nuU)

def ivffs_str(a):
    muL, muU, nuL, nuU = a
    return f"([μL={muL:.5f}, μU={muU:.5f}], [νL={nuL:.5f}, νU={nuU:.5f}])"

# =========================================================
# Dombi-type IVFFS weighted aggregation (IVFFDWA-style)
# (We implement a stable, q=3 version widely used for q-rung/fermatean operators.)
# =========================================================

def dombi_agg_mu(values, weights, alpha=2.0):
    """
    Membership aggregation:
      mu = [1 / (1 + (Σ w_k * ((1 - v^q)/(v^q))^alpha)^(1/alpha))]^(1/q)
    """
    alpha = float(alpha)
    s = 0.0
    for v, w in zip(values, weights):
        v = max(EPS, min(1.0 - EPS, float(v)))
        t = ((1.0 - v**Q) / (v**Q)) ** alpha
        s += float(w) * t
    s = max(EPS, s)
    inner = s ** (1.0 / alpha)
    out = (1.0 / (1.0 + inner)) ** (1.0 / Q)
    return clamp01(out)

def dombi_agg_nu(values, weights, alpha=2.0):
    """
    Non-membership aggregation:
      nu = [( (Σ w_k * ((v^q)/(1 - v^q))^alpha)^(1/alpha) ) / (1 + (Σ...)^(1/alpha)) ]^(1/q)
    """
    alpha = float(alpha)
    s = 0.0
    for v, w in zip(values, weights):
        v = max(EPS, min(1.0 - EPS, float(v)))
        t = ((v**Q) / (1.0 - v**Q)) ** alpha
        s += float(w) * t
    s = max(EPS, s)
    inner = s ** (1.0 / alpha)
    out = (inner / (1.0 + inner)) ** (1.0 / Q)
    return clamp01(out)

def ivffdwa_aggregate(ivffs_list, weights, alpha=2.0):
    """
    Aggregate a list of IVFFSs using Dombi-type operator, applied to each bound.
    """
    wsum = sum(weights)
    if wsum <= 0:
        weights = [1.0 / len(weights)] * len(weights)
    else:
        weights = [w / wsum for w in weights]

    muL_list = [x[0] for x in ivffs_list]
    muU_list = [x[1] for x in ivffs_list]
    nuL_list = [x[2] for x in ivffs_list]
    nuU_list = [x[3] for x in ivffs_list]

    muL = dombi_agg_mu(muL_list, weights, alpha)
    muU = dombi_agg_mu(muU_list, weights, alpha)
    nuL = dombi_agg_nu(nuL_list, weights, alpha)
    nuU = dombi_agg_nu(nuU_list, weights, alpha)

    return sanitize_ivffs(muL, muU, nuL, nuU)

# =========================================================
# Score function (paper-consistent form; yields [0,1])
#   Ψ(A) = 0.5 * ( 0.5*(muL^3 + muU^3 - nuL^3 - nuU^3) + 1 )
# =========================================================

def ivffs_score(a):
    muL, muU, nuL, nuU = a
    val = 0.5 * (0.5 * (muL**Q + muU**Q - nuL**Q - nuU**Q) + 1.0)
    return float(val)

# =========================================================
# Normalization for benefit vs cost criteria
#   Benefit: keep IVFFS as-is
#   Cost: swap membership and non-membership intervals
#         ([muL,muU],[nuL,nuU]) -> ([nuL,nuU],[muL,muU])
# =========================================================

def normalize_ivffs_benefit_cost(a, crit_type: str):
    muL, muU, nuL, nuU = a
    if crit_type.lower().startswith("c"):  # Cost
        return sanitize_ivffs(nuL, nuU, muL, muU)
    return sanitize_ivffs(muL, muU, nuL, nuU)

# =========================================================
# Default Linguistic Scales (editable in UI)
# NOTE: You can overwrite these to match Appendix tables exactly.
# =========================================================

DEFAULT_WINGS_STRENGTH = {
    "VLR": (0.00, 0.10, 0.85, 0.95),
    "LR" : (0.10, 0.25, 0.65, 0.80),
    "MR" : (0.25, 0.45, 0.45, 0.65),
    "HR" : (0.45, 0.70, 0.25, 0.45),
    "VHR": (0.70, 0.90, 0.05, 0.25),
}

DEFAULT_WINGS_INFLUENCE = {
    "ELI": (0.00, 0.10, 0.85, 0.95),
    "VLI": (0.10, 0.20, 0.70, 0.85),
    "LI" : (0.20, 0.35, 0.55, 0.70),
    "MI" : (0.35, 0.55, 0.40, 0.55),
    "HI" : (0.55, 0.70, 0.25, 0.40),
    "VHI": (0.70, 0.82, 0.12, 0.25),
    "EHI": (0.82, 0.92, 0.05, 0.12),
}

DEFAULT_TODIM_SCALE = {
    "VP": (0.00, 0.10, 0.85, 0.95),
    "P" : (0.10, 0.25, 0.65, 0.80),
    "MP": (0.25, 0.40, 0.50, 0.65),
    "F" : (0.40, 0.55, 0.35, 0.50),
    "MG": (0.55, 0.70, 0.20, 0.35),
    "G" : (0.70, 0.82, 0.10, 0.20),
    "VG": (0.82, 0.92, 0.05, 0.12),
}

# =========================================================
# Utilities: Scale editor + parsing
# =========================================================

def scale_dict_to_df(scale: dict):
    rows = []
    for k, v in scale.items():
        muL, muU, nuL, nuU = v
        rows.append({"Term": k, "muL": muL, "muU": muU, "nuL": nuL, "nuU": nuU})
    return pd.DataFrame(rows)

def df_to_scale_dict(df: pd.DataFrame):
    out = {}
    for _, r in df.iterrows():
        term = str(r["Term"]).strip()
        if not term:
            continue
        out[term] = sanitize_ivffs(float(r["muL"]), float(r["muU"]), float(r["nuL"]), float(r["nuU"]))
    return out

def ensure_square_matrix(n, fill="ELI"):
    return [[fill for _ in range(n)] for _ in range(n)]

# =========================================================
# IVFFS-WINGS Module
# =========================================================

def ivffs_wings_module():
    st.header("📊 IVFFS–WINGS")

    st.caption("Workflow: aggregate IVFFS-SIM → score matrix C → normalized SIM N → total SIM T → TI/TR → weights.")
    c1, c2 = st.columns(2)
    n = c1.number_input("Number of components", min_value=2, max_value=30, value=5, step=1)
    mexp = c2.number_input("Number of experts", min_value=1, max_value=20, value=4, step=1)

    # component names
    comp_names = []
    with st.expander("Component names", expanded=True):
        cols = st.columns(min(5, n))
        for i in range(n):
            with cols[i % len(cols)]:
                comp_names.append(st.text_input(f"C{i+1}", value=f"ESG{i+1}", key=f"w_comp_{i}"))

    # expert weights
    st.subheader("Expert weights (sum = 1)")
    if mexp == 1:
        exp_w = [1.0]
        st.info("Single expert → weight = 1.0")
    else:
        exp_w = []
        cols = st.columns(min(6, mexp))
        for k in range(mexp):
            with cols[k % len(cols)]:
                exp_w.append(st.number_input(f"E{k+1}", min_value=0.0, max_value=1.0, value=round(1/mexp, 4), step=0.01, key=f"w_expw_{k}"))
        if not np.isclose(sum(exp_w), 1.0):
            st.error(f"Expert weights must sum to 1.0 (now {sum(exp_w):.6f}).")
            st.stop()

    # Scale editors
    st.subheader("Linguistic scales (editable)")
    st.write("If you want 1:1 Excel match, paste the exact IVFFS values from your Appendix tables here.")
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("**Strength/Relevance scale**")
        df_strength = st.data_editor(
            scale_dict_to_df(st.session_state.get("w_strength_scale", DEFAULT_WINGS_STRENGTH)),
            hide_index=True,
            use_container_width=True,
            key="w_strength_scale_editor"
        )
        strength_scale = df_to_scale_dict(df_strength)
        st.session_state["w_strength_scale"] = strength_scale
    with sc2:
        st.markdown("**Influence scale**")
        df_infl = st.data_editor(
            scale_dict_to_df(st.session_state.get("w_infl_scale", DEFAULT_WINGS_INFLUENCE)),
            hide_index=True,
            use_container_width=True,
            key="w_infl_scale_editor"
        )
        infl_scale = df_to_scale_dict(df_infl)
        st.session_state["w_infl_scale"] = infl_scale

    alpha = st.number_input("Dombi parameter α", min_value=0.1, max_value=20.0, value=2.0, step=0.1)

    # Expert inputs
    st.subheader("Expert inputs (Strengths + Influence Matrix)")
    if "w_experts" not in st.session_state or len(st.session_state["w_experts"]) != mexp or st.session_state.get("w_n_cached") != n:
        st.session_state["w_n_cached"] = n
        st.session_state["w_experts"] = []
        for e in range(mexp):
            st.session_state["w_experts"].append({
                "strength_terms": ["HR"] * n,
                "influence_terms": ensure_square_matrix(n, fill="ELI"),
            })

    tabs = st.tabs([f"Expert {e+1}" for e in range(mexp)]) if mexp > 1 else [st.container()]
    for e in range(mexp):
        with tabs[e] if mexp > 1 else tabs[0]:
            st.markdown("**Diagonal strengths (relevance)**")
            cols = st.columns(min(6, n))
            for i in range(n):
                with cols[i % len(cols)]:
                    cur = st.session_state["w_experts"][e]["strength_terms"][i]
                    term = st.selectbox(
                        comp_names[i],
                        options=list(strength_scale.keys()),
                        index=list(strength_scale.keys()).index(cur) if cur in strength_scale else 0,
                        key=f"w_strength_{e}_{i}"
                    )
                    st.session_state["w_experts"][e]["strength_terms"][i] = term

            st.markdown("**Influence matrix** (row → column). Diagonal is fixed by strengths.")
            for i in range(n):
                row = st.columns(n+1)
                row[0].markdown(f"**{comp_names[i]}**")
                for j in range(n):
                    if i == j:
                        row[j+1].markdown("—")
                        continue
                    cur = st.session_state["w_experts"][e]["influence_terms"][i][j]
                    term = row[j+1].selectbox(
                        f"{i}-{j}",
                        options=list(infl_scale.keys()),
                        index=list(infl_scale.keys()).index(cur) if cur in infl_scale else 0,
                        key=f"w_infl_{e}_{i}_{j}"
                    )
                    st.session_state["w_experts"][e]["influence_terms"][i][j] = term

    if st.button("✅ Run IVFFS–WINGS", type="primary", use_container_width=True):
        # Build each expert IVFFS-SIM as IVFFS matrix (n x n)
        expert_mats = []
        for e in range(mexp):
            mat = [[None]*n for _ in range(n)]
            # diagonal strengths
            for i in range(n):
                mat[i][i] = strength_scale[st.session_state["w_experts"][e]["strength_terms"][i]]
            for i in range(n):
                for j in range(n):
                    if i == j: 
                        continue
                    mat[i][j] = infl_scale[st.session_state["w_experts"][e]["influence_terms"][i][j]]
            expert_mats.append(mat)

        # Aggregate to eβ (aggregated IVFFS-SIM)
        agg = [[None]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                cell_list = [expert_mats[e][i][j] for e in range(mexp)]
                agg[i][j] = ivffdwa_aggregate(cell_list, exp_w, alpha=alpha)

        # Score matrix C (Ceij)
        C = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                C[i, j] = ivffs_score(agg[i][j])

        # scaling factor ς and normalized SIM N
        zeta = float(np.sum(C))
        if abs(zeta) < EPS:
            zeta = 1.0
        N = C / zeta

        # Total relation matrix T = N (I - N)^(-1)
        I = np.eye(n)
        try:
            T = N @ np.linalg.inv(I - N)
        except np.linalg.LinAlgError:
            T = N @ np.linalg.pinv(I - N)

        TI = np.sum(T, axis=1)   # Eq (18)
        TR = np.sum(T, axis=0)   # Eq (19)

        engagement = TI + TR
        role = TI - TR

        # Eq (20)-(21): expected value and normalized weights
        E = np.sqrt(engagement**2 + role**2)
        w = E / (np.sum(E) if np.sum(E) > EPS else 1.0)

        # persist for TODIM
        st.session_state["wings_weights"] = w
        st.session_state["wings_component_names"] = comp_names

        st.success("WINGS computed.")

        with st.expander("Aggregated IVFFS-SIM (eβ)", expanded=False):
            df_agg = pd.DataFrame([[ivffs_str(agg[i][j]) for j in range(n)] for i in range(n)], index=comp_names, columns=comp_names)
            st.dataframe(df_agg, use_container_width=True)

        st.subheader("Score matrix C and normalized SIM N")
        st.write("C = score(eβ), N = C / ς")
        st.dataframe(pd.DataFrame(C, index=comp_names, columns=comp_names).style.format("{:.6f}"), use_container_width=True)
        st.dataframe(pd.DataFrame(N, index=comp_names, columns=comp_names).style.format("{:.6f}"), use_container_width=True)

        st.subheader("Total relation matrix T")
        st.dataframe(pd.DataFrame(T, index=comp_names, columns=comp_names).style.format("{:.6f}"), use_container_width=True)

        st.subheader("TI / TR / Engagement / Role / Expected value / Weight")
        out = pd.DataFrame({
            "Component": comp_names,
            "TI": TI,
            "TR": TR,
            "TI+TR (Engagement)": engagement,
            "TI-TR (Role)": role,
            "E(ω)": E,
            "ω (weight)": w
        })
        out["Group"] = np.where(out["TI-TR (Role)"] > 0, "Cause", "Effect")
        st.dataframe(out.style.format("{:.6f}"), use_container_width=True, hide_index=True)

        # download CSV
        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download WINGS results (CSV)", data=csv, file_name="ivffs_wings_results.csv", mime="text/csv")


# =========================================================
# IVFFS-TODIM Module
# =========================================================

def todim_distance(a, b):
    # τ values are crisp scores; paper uses d(τij, τqj) (we use absolute difference)
    return abs(float(a) - float(b))

def ivffs_todim_module():
    st.header("📈 IVFFS–TODIM")

    st.caption("Workflow: (optional) normalize IVFFS by benefit/cost → score matrix τ → relative weights → dominance → superiority → rank.")

    c1, c2 = st.columns(2)
    alts_in = c1.text_input("Alternatives (comma-separated)", "S1, S2, S3, S4, S5", key="t_alts")
    crits_in = c2.text_input("Criteria (comma-separated)", "ESG1, ESG2, ESG3, ESG4", key="t_crits")
    alternatives = [x.strip() for x in alts_in.split(",") if x.strip()]
    criteria = [x.strip() for x in crits_in.split(",") if x.strip()]
    if len(alternatives) < 2 or len(criteria) < 1:
        st.warning("Need at least 2 alternatives and 1 criterion.")
        return

    # Criteria types + weights
    st.subheader("Criteria types & weights")
    if "t_crit_df" not in st.session_state or set(st.session_state["t_crit_df"]["Criterion"]) != set(criteria):
        w0 = [round(1/len(criteria), 6)] * len(criteria)
        w0[-1] = 1.0 - sum(w0[:-1]) if len(criteria) > 0 else 1.0
        st.session_state["t_crit_df"] = pd.DataFrame({
            "Criterion": criteria,
            "Type": ["Benefit"] * len(criteria),
            "Weight": w0
        })

    edited = st.data_editor(
        st.session_state["t_crit_df"],
        hide_index=True,
        use_container_width=True,
        column_config={
            "Type": st.column_config.SelectboxColumn("Type", options=["Benefit","Cost"]),
            "Weight": st.column_config.NumberColumn("Weight", min_value=0.0, max_value=1.0, step=0.0001, format="%.6f")
        },
        key="t_crit_editor"
    )
    crit_types = edited["Type"].tolist()
    crit_w = edited["Weight"].astype(float).tolist()

    # Option: use weights from WINGS
    use_wings = st.checkbox("Use weights computed from IVFFS–WINGS (if available)", value=False)
    if use_wings and "wings_weights" in st.session_state:
        wings_w = st.session_state["wings_weights"]
        if len(wings_w) == len(criteria):
            crit_w = list(map(float, wings_w))
            st.info("Using WINGS weights for TODIM.")
        else:
            st.warning("WINGS weights exist, but their length != number of criteria here. Using manual weights.")

    if not np.isclose(sum(crit_w), 1.0):
        st.error(f"Criteria weights must sum to 1.0 (now {sum(crit_w):.6f}).")
        return

    # TODIM linguistic scale editor
    st.subheader("TODIM linguistic scale (editable)")
    df_tscale = st.data_editor(
        scale_dict_to_df(st.session_state.get("t_scale", DEFAULT_TODIM_SCALE)),
        hide_index=True,
        use_container_width=True,
        key="t_scale_editor"
    )
    t_scale = df_to_scale_dict(df_tscale)
    st.session_state["t_scale"] = t_scale

    # Experts
    nexp = st.number_input("Number of experts", min_value=1, max_value=30, value=4, step=1)
    st.markdown("**Expert weights (sum=1)**")
    if nexp == 1:
        exp_w = [1.0]
        st.info("Single expert → weight = 1.0")
    else:
        exp_w = []
        cols = st.columns(min(6, nexp))
        for e in range(nexp):
            with cols[e % len(cols)]:
                exp_w.append(st.number_input(f"E{e+1}", min_value=0.0, max_value=1.0, value=round(1/nexp, 4), step=0.01, key=f"t_expw_{e}"))
        if not np.isclose(sum(exp_w), 1.0):
            st.error(f"Expert weights must sum to 1.0 (now {sum(exp_w):.6f}).")
            return

    alpha = st.number_input("Dombi parameter α (aggregation)", min_value=0.1, max_value=20.0, value=2.0, step=0.1, key="t_alpha")
    xi = st.number_input("Attenuation factor ξ", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

    # Decision matrices per expert (linguistic)
    if "t_expert_dfs" not in st.session_state:
        st.session_state["t_expert_dfs"] = {}

    need_reset = (
        len(st.session_state["t_expert_dfs"]) != nexp
        or (nexp > 0 and (
            set(st.session_state["t_expert_dfs"].get(0, pd.DataFrame()).index) != set(alternatives)
            or set(st.session_state["t_expert_dfs"].get(0, pd.DataFrame()).columns) != set(criteria)
        ))
    )
    if need_reset:
        st.session_state["t_expert_dfs"] = {
            e: pd.DataFrame("F", index=alternatives, columns=criteria) for e in range(nexp)
        }

    st.subheader("Expert evaluations (linguistic)")
    tabs = st.tabs([f"Expert {e+1}" for e in range(nexp)])
    for e, tab in enumerate(tabs):
        with tab:
            st.session_state["t_expert_dfs"][e] = st.data_editor(
                st.session_state["t_expert_dfs"][e],
                use_container_width=True,
                column_config={c: st.column_config.SelectboxColumn(c, options=list(t_scale.keys())) for c in criteria},
                key=f"t_editor_{e}"
            )

    if st.button("✅ Run IVFFS–TODIM", type="primary", use_container_width=True):
        # Step 3.2: Aggregate decision matrix S using IVFFDWA (by cell)
        agg = {}
        for a in alternatives:
            for j, c in enumerate(criteria):
                cell_ivffs = []
                for e in range(nexp):
                    term = st.session_state["t_expert_dfs"][e].loc[a, c]
                    cell_ivffs.append(t_scale[term])
                agg[(a, c)] = ivffdwa_aggregate(cell_ivffs, exp_w, alpha=alpha)

        # Step 3.3: Normalize by benefit/cost (swap for cost)
        norm = {}
        for j, c in enumerate(criteria):
            for a in alternatives:
                norm[(a, c)] = normalize_ivffs_benefit_cost(agg[(a, c)], crit_types[j])

        # Step 3.4: Score matrix τ
        tau = pd.DataFrame(index=alternatives, columns=criteria, dtype=float)
        for a in alternatives:
            for c in criteria:
                tau.loc[a, c] = ivffs_score(norm[(a, c)])

        # Step 3.5: Relative weights ω'_j = ω_j / ω_r, ω_r = max ω
        omega = np.array(crit_w, dtype=float)
        omega_r = float(np.max(omega)) if float(np.max(omega)) > EPS else 1.0
        omega_rel = omega / omega_r

        denom_sumw = float(np.sum(omega)) if float(np.sum(omega)) > EPS else 1.0

        # Step 3.6-3.8: dominance ψj, overall superiority Φ, comprehensive Υ, normalized Υ*
        m = len(alternatives)
        ncrit = len(criteria)

        # dominance per pair
        Phi = np.zeros((m, m), dtype=float)

        for i in range(m):
            for q in range(m):
                if i == q:
                    continue
                si = alternatives[i]
                sq = alternatives[q]

                total = 0.0
                for j, c in enumerate(criteria):
                    tij = float(tau.loc[si, c])
                    tqj = float(tau.loc[sq, c])
                    d = todim_distance(tij, tqj)

                    if tij > tqj + 1e-15:
                        part = math.sqrt((omega_rel[j] / denom_sumw) * d)
                    elif abs(tij - tqj) <= 1e-15:
                        part = 0.0
                    else:
                        part = - (1.0 / float(xi)) * math.sqrt((omega_rel[j] / denom_sumw) * d)

                    total += part

                Phi[i, q] = total

        Upsilon = np.sum(Phi, axis=1)  # comprehensive superiority
        umin = float(np.min(Upsilon))
        umax = float(np.max(Upsilon))
        if abs(umax - umin) < EPS:
            U_norm = np.zeros_like(Upsilon)
        else:
            U_norm = (Upsilon - umin) / (umax - umin)

        res = pd.DataFrame({
            "Alternative": alternatives,
            "Υ (comprehensive)": Upsilon,
            "Υ* (normalized)": U_norm
        }).sort_values("Υ* (normalized)", ascending=False).reset_index(drop=True)

        res["Rank"] = np.arange(1, len(res) + 1)

        st.success("TODIM computed.")

        with st.expander("Aggregated IVFFS decision matrix (S)", expanded=False):
            dfS = pd.DataFrame(index=alternatives, columns=criteria, dtype=object)
            for a in alternatives:
                for c in criteria:
                    dfS.loc[a, c] = ivffs_str(agg[(a, c)])
            st.dataframe(dfS, use_container_width=True)

        with st.expander("Normalized IVFFS decision matrix (η)", expanded=False):
            dfN = pd.DataFrame(index=alternatives, columns=criteria, dtype=object)
            for a in alternatives:
                for c in criteria:
                    dfN.loc[a, c] = ivffs_str(norm[(a, c)])
            st.dataframe(dfN, use_container_width=True)

        st.subheader("Score matrix τ")
        st.dataframe(tau.style.format("{:.6f}"), use_container_width=True)

        st.subheader("Dominance matrix Φ(Si,Sq)")
        st.dataframe(pd.DataFrame(Phi, index=alternatives, columns=alternatives).style.format("{:.6f}"), use_container_width=True)

        st.subheader("Final ranking (Υ*)")
        st.dataframe(res.style.format({"Υ (comprehensive)": "{:.6f}", "Υ* (normalized)": "{:.6f}"}), use_container_width=True, hide_index=True)

        # downloads
        csv1 = tau.reset_index().rename(columns={"index": "Alternative"}).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download TODIM τ matrix (CSV)", data=csv1, file_name="ivffs_todim_tau.csv", mime="text/csv")
        csv2 = res.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download TODIM ranking (CSV)", data=csv2, file_name="ivffs_todim_ranking.csv", mime="text/csv")


# =========================================================
# Main App (Two modules)
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

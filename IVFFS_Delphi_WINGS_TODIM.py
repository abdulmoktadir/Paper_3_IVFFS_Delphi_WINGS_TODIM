import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# IVFFS core (Interval-valued Fermatean fuzzy set)
#   beta = ([mu_lb, mu_ub], [nu_lb, nu_ub])
# ============================================================

EPS = 1e-12

@dataclass(frozen=True)
class IVFFS:
    mu_lb: float
    mu_ub: float
    nu_lb: float
    nu_ub: float

    def clipped(self) -> "IVFFS":
        def clip01(x: float) -> float:
            return float(min(1.0 - EPS, max(0.0 + EPS, x)))
        return IVFFS(
            clip01(self.mu_lb), clip01(self.mu_ub),
            clip01(self.nu_lb), clip01(self.nu_ub)
        )

    def fmt(self, nd: int = 4) -> str:
        return f"([{self.mu_lb:.{nd}f},{self.mu_ub:.{nd}f}],[{self.nu_lb:.{nd}f},{self.nu_ub:.{nd}f}])"


# ============================================================
# Your IVFFS linguistic scale for TODIM (as provided)
# ============================================================

IVFFS_TODIM_SCALE: Dict[str, IVFFS] = {
    "VP": IVFFS(0.10, 0.15, 0.90, 0.95),
    "P":  IVFFS(0.20, 0.25, 0.80, 0.85),
    "MP": IVFFS(0.30, 0.35, 0.70, 0.75),
    "F":  IVFFS(0.50, 0.55, 0.40, 0.45),
    "MG": IVFFS(0.70, 0.75, 0.30, 0.35),
    "G":  IVFFS(0.80, 0.85, 0.20, 0.25),
    "VG": IVFFS(0.90, 0.95, 0.10, 0.15),
}

IVFFS_TODIM_FULL = {
    "VP": "Very Poor",
    "P":  "Poor",
    "MP": "Medium Poor",
    "F":  "Fair",
    "MG": "Medium Good",
    "G":  "Good",
    "VG": "Very Good",
}

# ============================================================
# Smart parser for alternatives/criteria
# Supports:
#   "S1, S2, S3"
#   "S1-S10" or "S1..S10"
#   "S1, S2, ..., S10"
#   "S1,S2,...,SN" (if last token is e.g., S10)
# ============================================================

def parse_items(text: str, default_prefix: str = "S") -> List[str]:
    if not text:
        return []

    t = text.strip()

    # Normalize ellipsis
    t = t.replace("….", "...").replace("....", "...")

    # Case 1: Range like S1-S10 or S1..S10
    m = re.match(r"^\s*([A-Za-z_]+)(\d+)\s*(?:-|\.\.)\s*([A-Za-z_]+)?(\d+)\s*$", t)
    if m:
        p1, s1, p2, s2 = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
        p2 = p2 if p2 else p1
        if p1 != p2:
            # If prefixes differ, just fallback to comma parsing
            pass
        else:
            step = 1 if s2 >= s1 else -1
            return [f"{p1}{i}" for i in range(s1, s2 + step, step)]

    # Case 2: Comma list (possibly with "...")
    parts = [p.strip() for p in t.split(",") if p.strip() != ""]
    if not parts:
        return []

    # If there's "..." inside, expand from first item to last item (if both are like PrefixNumber)
    if any(p == "..." or "..." in p for p in parts):
        # get first token that looks like PrefixNumber
        first = None
        last = None
        for p in parts:
            mm = re.match(r"^([A-Za-z_]+)(\d+)$", p)
            if mm:
                first = (mm.group(1), int(mm.group(2)))
                break
        for p in reversed(parts):
            mm = re.match(r"^([A-Za-z_]+)(\d+)$", p)
            if mm:
                last = (mm.group(1), int(mm.group(2)))
                break
        if first and last and first[0] == last[0]:
            pfx, s1 = first
            _, s2 = last
            step = 1 if s2 >= s1 else -1
            return [f"{pfx}{i}" for i in range(s1, s2 + step, step)]

    # Otherwise: just keep clean tokens (remove any "...", "SN" without number won't expand)
    cleaned = []
    for p in parts:
        if p == "..." or p.startswith("..."):
            continue
        cleaned.append(p)

    # If user typed "S1, S2, ..., SN" literally SN, we cannot expand without N number.
    # But we still include it as a label so it shows.
    return cleaned


# ============================================================
# Score function Psi (kept consistent with your earlier usage)
# ============================================================

def psi_score(iv: IVFFS) -> float:
    iv = iv.clipped()
    mu = 0.5 * (iv.mu_lb**3 + iv.mu_ub**3)
    nu = 0.5 * (iv.nu_lb**3 + iv.nu_ub**3)
    return 0.5 * ((mu - nu) + 1.0)


# ============================================================
# IVFFDWA aggregation (Dombi-based), used in BOTH WINGS and TODIM
# ============================================================

def _safe_ratio_mu(x: float) -> float:
    x = float(min(1.0 - EPS, max(EPS, x)))
    x3 = x**3
    den = max(EPS, 1.0 - x3)
    return x3 / den

def _safe_ratio_nu(x: float) -> float:
    x = float(min(1.0 - EPS, max(EPS, x)))
    x3 = max(EPS, x**3)
    return (1.0 - x3) / x3

def _dombi_weighted_mean(terms: List[float], weights: List[float], alpha: float) -> float:
    alpha = float(alpha)
    if alpha <= 0:
        raise ValueError("Dombi parameter alpha must be > 0.")
    s = 0.0
    for t, w in zip(terms, weights):
        s += float(w) * (float(t) ** alpha)
    return float(s) ** (1.0 / alpha)

def _agg_mu(values: List[float], weights: List[float], alpha: float) -> float:
    terms = [_safe_ratio_mu(v) for v in values]
    inner = _dombi_weighted_mean(terms, weights, alpha)
    val = 1.0 - (1.0 / (1.0 + inner))
    return float(val) ** (1.0 / 3.0)

def _agg_nu(values: List[float], weights: List[float], alpha: float) -> float:
    terms = [_safe_ratio_nu(v) for v in values]
    inner = _dombi_weighted_mean(terms, weights, alpha)
    denom = (1.0 + inner) ** (1.0 / 3.0)
    return 1.0 / denom

def ivffdwa_aggregate(iv_list: List[IVFFS], weights: List[float], alpha: float) -> IVFFS:
    if len(iv_list) == 0:
        raise ValueError("Empty iv_list.")
    if len(iv_list) != len(weights):
        raise ValueError("iv_list and weights must have same length.")
    if not np.isclose(sum(weights), 1.0):
        raise ValueError("Expert weights must sum to 1.")

    iv_list = [iv.clipped() for iv in iv_list]
    mu_lb = _agg_mu([iv.mu_lb for iv in iv_list], weights, alpha)
    mu_ub = _agg_mu([iv.mu_ub for iv in iv_list], weights, alpha)
    nu_lb = _agg_nu([iv.nu_lb for iv in iv_list], weights, alpha)
    nu_ub = _agg_nu([iv.nu_ub for iv in iv_list], weights, alpha)

    return IVFFS(mu_lb, mu_ub, nu_lb, nu_ub).clipped()


# ============================================================
# TODIM normalization (Benefit keep; Cost swap μ and ν)
# ============================================================

def normalize_ivffs_todim(matrix: Dict[Tuple[str, str], IVFFS],
                          alternatives: List[str],
                          criteria: List[str],
                          criteria_types: List[str]) -> Dict[Tuple[str, str], IVFFS]:
    out = {}
    for j, c in enumerate(criteria):
        is_benefit = criteria_types[j].strip().lower().startswith("b")
        for a in alternatives:
            iv = matrix[(a, c)].clipped()
            if is_benefit:
                out[(a, c)] = iv
            else:
                out[(a, c)] = IVFFS(iv.nu_lb, iv.nu_ub, iv.mu_lb, iv.mu_ub).clipped()
    return out


# ============================================================
# TODIM ranking (simple, consistent implementation)
# ============================================================

def todim_rank(norm_matrix: Dict[Tuple[str, str], IVFFS],
               alternatives: List[str],
               criteria: List[str],
               weights: List[float],
               xi: float = 1.0) -> pd.DataFrame:
    w = np.array(weights, dtype=float)
    w_r = float(np.max(w)) if float(np.max(w)) > 0 else 1.0
    w_rel = w / w_r

    C = np.zeros((len(alternatives), len(criteria)), dtype=float)
    for i, a in enumerate(alternatives):
        for j, c in enumerate(criteria):
            C[i, j] = psi_score(norm_matrix[(a, c)])

    m = len(alternatives)
    n = len(criteria)
    Phi = np.zeros((m, m), dtype=float)

    for i in range(m):
        for q in range(m):
            if i == q:
                continue
            s = 0.0
            for j in range(n):
                d = abs(C[i, j] - C[q, j])
                if C[i, j] >= C[q, j]:
                    s += math.sqrt(w_rel[j]) * d
                else:
                    s += -(1.0 / max(EPS, xi)) * math.sqrt(w_rel[j]) * d
            Phi[i, q] = s

    U = Phi.sum(axis=1)
    Umin, Umax = float(np.min(U)), float(np.max(U))
    if abs(Umax - Umin) < EPS:
        Un = np.ones_like(U)
    else:
        Un = (U - Umin) / (Umax - Umin)

    df = pd.DataFrame({
        "Alternative": alternatives,
        "TODIM_U": U,
        "TODIM_U_norm": Un
    })
    df["Rank"] = df["TODIM_U_norm"].rank(ascending=False, method="min").astype(int)
    return df.sort_values("Rank").reset_index(drop=True)


# ============================================================
# Display helper
# ============================================================

def ivffs_table(matrix: Dict[Tuple[str, str], IVFFS], alts: List[str], crits: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(index=alts, columns=crits, dtype=object)
    for a in alts:
        for c in crits:
            df.loc[a, c] = matrix[(a, c)].fmt(4)
    return df


# ============================================================
# Module 1: IVFFS-WINGS (simple placeholder to pass weights)
# ============================================================

def ivffs_wings_module():
    st.header("🧠 IVFFS-WINGS")
    st.caption("Paste / store weights for TODIM. Aggregation operator is IVFFDWA (same as TODIM).")

    n = st.number_input("Number of criteria (to send to TODIM)", min_value=1, max_value=200, value=10, step=1)
    weights_str = st.text_area("Paste weights (comma-separated, sum=1)", value=",".join(["0.1"] * int(n)))

    try:
        w = [float(x.strip()) for x in weights_str.split(",") if x.strip() != ""]
        if len(w) != int(n):
            st.warning("Weight count does not match.")
            return
        if not np.isclose(sum(w), 1.0):
            st.warning(f"Sum(weights)={sum(w):.6f} (must be 1).")
            return

        df = pd.DataFrame({"Criterion": [f"ESG{i+1}" for i in range(int(n))], "Weight": w})
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.session_state["WINGS_WEIGHTS_FOR_TODIM"] = w
        st.success("Saved weights for TODIM module (this session).")
    except Exception as e:
        st.error(f"Invalid weights: {e}")


# ============================================================
# Module 2: IVFFS-TODIM (fixed alt/crit parsing + proper reset)
# ============================================================

def ivffs_todim_module():
    st.header("📊 IVFFS-TODIM")
    st.caption("Includes IVFFDWA aggregation + TODIM normalization (Benefit keep; Cost swap μ and ν).")

    with st.expander("IVFFS linguistic scale (VP…VG)"):
        df_scale = pd.DataFrame([{
            "Abbr": k,
            "Meaning": IVFFS_TODIM_FULL[k],
            "IVFFS": IVFFS_TODIM_SCALE[k].fmt(2)
        } for k in IVFFS_TODIM_SCALE])
        st.dataframe(df_scale, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    alts_in = c1.text_input(
        "Alternatives (supports: S1-S10 or S1, S2, ..., S10)",
        "S1, S2, ..., S10",
        key="alts_in"
    )
    crits_in = c2.text_input(
        "Criteria (supports: C1-C5 or ESG1, ESG2, ..., ESG5)",
        "ESG1, ESG2, ESG3",
        key="crits_in"
    )

    alternatives = parse_items(alts_in)
    criteria = parse_items(crits_in)

    if len(alternatives) == 0 or len(criteria) == 0:
        st.warning("Please provide valid alternatives and criteria.")
        return

    st.write(f"✅ Parsed Alternatives: {len(alternatives)} → {alternatives[:6]}{' ...' if len(alternatives) > 6 else ''}")
    st.write(f"✅ Parsed Criteria: {len(criteria)} → {criteria[:6]}{' ...' if len(criteria) > 6 else ''}")

    # --- criteria types & weights
    st.subheader("Criteria types & weights")
    crit_signature = (tuple(criteria),)

    if "todim_crit_signature" not in st.session_state or st.session_state["todim_crit_signature"] != crit_signature:
        w0 = [1.0 / len(criteria)] * len(criteria)
        w0[-1] = 1.0 - sum(w0[:-1])
        st.session_state["todim_crit_df"] = pd.DataFrame({
            "Criterion": criteria,
            "Type": ["Benefit"] * len(criteria),
            "Weight": w0
        })
        st.session_state["todim_crit_signature"] = crit_signature

    default_wings_w = st.session_state.get("WINGS_WEIGHTS_FOR_TODIM")
    if default_wings_w and len(default_wings_w) == len(criteria):
        if st.button("⬇️ Use weights saved from WINGS module"):
            st.session_state["todim_crit_df"]["Weight"] = default_wings_w

    edited = st.data_editor(
        st.session_state["todim_crit_df"],
        hide_index=True,
        use_container_width=True,
        column_config={
            "Type": st.column_config.SelectboxColumn("Type", options=["Benefit", "Cost"]),
            "Weight": st.column_config.NumberColumn("Weight", min_value=0.0, max_value=1.0, format="%.6f"),
        },
        key="todim_crit_editor"
    )

    criteria_types = edited["Type"].tolist()
    weights = edited["Weight"].astype(float).tolist()

    if not np.isclose(sum(weights), 1.0):
        st.error(f"Criteria weights must sum to 1.0 (now {sum(weights):.6f}).")
        return

    # --- experts
    st.subheader("Experts & evaluations")
    l = st.number_input("Number of experts", min_value=1, max_value=30, value=4, step=1, key="nexp")

    if int(l) == 1:
        exp_w = [1.0]
        st.info("Single expert → weight=1.0")
    else:
        cols = st.columns(int(l))
        exp_w = []
        for i in range(int(l)):
            with cols[i]:
                exp_w.append(st.number_input(
                    f"E{i+1} weight",
                    min_value=0.0, max_value=1.0,
                    value=round(1.0 / int(l), 6),
                    step=0.01,
                    format="%.6f",
                    key=f"ew_{i}"
                ))
        if not np.isclose(sum(exp_w), 1.0):
            st.error(f"Expert weights must sum to 1.0 (now {sum(exp_w):.6f}).")
            return

    alpha = st.number_input("Dombi parameter α", min_value=0.01, max_value=10.0, value=0.5, step=0.05, key="alpha")
    xi = st.number_input("TODIM attenuation factor ξ", min_value=0.01, max_value=50.0, value=1.0, step=0.1, key="xi")

    # ✅ Reset expert tables when alternatives/criteria/experts change
    tables_signature = (tuple(alternatives), tuple(criteria), int(l))
    if "todim_tables_signature" not in st.session_state or st.session_state["todim_tables_signature"] != tables_signature:
        st.session_state["todim_expert_tables"] = [
            pd.DataFrame("F", index=alternatives, columns=criteria) for _ in range(int(l))
        ]
        st.session_state["todim_tables_signature"] = tables_signature

    tabs = st.tabs([f"Expert {i+1}" for i in range(int(l))])
    for i in range(int(l)):
        with tabs[i]:
            st.session_state["todim_expert_tables"][i] = st.data_editor(
                st.session_state["todim_expert_tables"][i],
                use_container_width=True,
                column_config={
                    c: st.column_config.SelectboxColumn(c, options=list(IVFFS_TODIM_SCALE.keys()))
                    for c in criteria
                },
                key=f"todim_ed_{i}"
            )

    if st.button("✅ Run IVFFS-TODIM", type="primary", use_container_width=True, key="run_todim"):
        # Aggregate decision matrix using IVFFDWA
        agg: Dict[Tuple[str, str], IVFFS] = {}
        for a in alternatives:
            for c in criteria:
                iv_list = []
                for k in range(int(l)):
                    term = str(st.session_state["todim_expert_tables"][k].loc[a, c])
                    iv_list.append(IVFFS_TODIM_SCALE[term])
                agg[(a, c)] = ivffdwa_aggregate(iv_list, exp_w, alpha)

        st.markdown("#### Aggregated IVFFS decision matrix (S)")
        st.dataframe(ivffs_table(agg, alternatives, criteria), use_container_width=True)

        # Normalize (Benefit keep; Cost swap)
        norm = normalize_ivffs_todim(agg, alternatives, criteria, criteria_types)
        st.markdown("#### Normalized decision matrix (η)")
        st.dataframe(ivffs_table(norm, alternatives, criteria), use_container_width=True)

        # TODIM ranking
        df_rank = todim_rank(norm, alternatives, criteria, weights, xi=xi)
        st.markdown("#### TODIM results")
        st.dataframe(df_rank, use_container_width=True, hide_index=True)


# ============================================================
# Main navigation
# ============================================================

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

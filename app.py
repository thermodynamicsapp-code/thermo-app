import os
import zipfile
import pandas as pd
import streamlit as st

# ============================================================
# Thermodynamic Water Properties (6 Scenarios) — Streamlit App
# Local-only tables: Thermo.zip must be next to app.py
# EPS_SAT fixed at 0.01 °C
# Scenarios:
#   1) (P,T) -> h/u/v/s   (region detection, x optional if saturated)
#   2) (P,v) -> h/u/s     (returns T as well)
#   3) (P,h) -> v/u/s     (returns T as well)
#   4) (P,s) -> v/u/h     (returns T as well)
#   5) (T,x) -> v/u/h/s   (saturated only, returns Psat)
#   6) (P,x) -> v/u/h/s   (saturated only, returns T)
# ============================================================

ZIP_NAME = "Thermo.zip"
EXTRACT_DIR = "thermo_tables"
ZIP_SUBDIR = os.path.join(EXTRACT_DIR, "ZIP")

EPS_SAT = 0.01  # fixed saturation tolerance (°C)

SAT_COLS = {
    "h": ("h_f", "h_g"),
    "v": ("v_f", "v_g"),
    "u": ("u_f", "u_g"),
    "s": ("s_f", "s_g"),
}


# =========================
# Local ZIP extraction
# =========================
def ensure_tables_ready_from_local_zip():
    if not os.path.exists(ZIP_NAME):
        raise FileNotFoundError("Thermo.zip not found. Put Thermo.zip next to app.py.")

    os.makedirs(EXTRACT_DIR, exist_ok=True)

    # already extracted
    if os.path.exists(ZIP_SUBDIR) and os.listdir(ZIP_SUBDIR):
        return

    with zipfile.ZipFile(ZIP_NAME, "r") as zf:
        zf.extractall(EXTRACT_DIR)


@st.cache_data
def load_saturation_table_B12():
    ensure_tables_ready_from_local_zip()
    path = os.path.join(ZIP_SUBDIR, "B.1.2.xlsx")

    raw = pd.read_excel(path, header=None)
    df = raw.iloc[2:].reset_index(drop=True)

    df.columns = [
        "Pressure", "Temperature", "v_f", "v_fg", "v_g",
        "u_f", "u_fg", "u_g", "h_f", "h_fg", "h_g",
        "s_f", "s_fg", "s_g",
    ]
    return df.astype(float)


# =========================
# Math helpers
# =========================
def linear_interp(x, x1, x2, y1, y2):
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)


def inverse_linear_interp(y, y1, y2, x1, x2):
    return x1 + (y - y1) * (x2 - x1) / (y2 - y1)


# =========================
# Saturation (B.1.2)
# =========================
def Tsat_from_P(P, table_sat):
    df = table_sat.sort_values("Pressure").reset_index(drop=True)
    s = df["Pressure"]

    if P < s.min() or P > s.max():
        raise ValueError("Pressure is outside saturation table range.")

    exact = df[df["Pressure"] == P]
    if not exact.empty:
        return float(exact["Temperature"].iloc[0])

    lower = df[df["Pressure"] < P].iloc[-1]
    upper = df[df["Pressure"] > P].iloc[0]

    return float(linear_interp(P, lower["Pressure"], upper["Pressure"], lower["Temperature"], upper["Temperature"]))


def Psat_from_T(T, table_sat):
    df = table_sat.sort_values("Temperature").reset_index(drop=True)
    s = df["Temperature"]

    if T < s.min() or T > s.max():
        raise ValueError("Temperature is outside saturation table range.")

    exact = df[df["Temperature"] == T]
    if not exact.empty:
        return float(exact["Pressure"].iloc[0])

    lower = df[df["Temperature"] < T].iloc[-1]
    upper = df[df["Temperature"] > T].iloc[0]

    return float(linear_interp(T, lower["Temperature"], upper["Temperature"], lower["Pressure"], upper["Pressure"]))


def classify_region_PT(P, T, table_sat, eps=EPS_SAT):
    Tsat = Tsat_from_P(P, table_sat)
    if abs(T - Tsat) <= eps:
        return "saturated", Tsat
    if T > Tsat:
        return "superheated", Tsat
    return "compressed", Tsat


def sat_props_at_P(P, table_sat, prop_key):
    col_f, col_g = SAT_COLS[prop_key]

    df = table_sat.sort_values("Pressure").reset_index(drop=True)
    s = df["Pressure"]

    if P < s.min() or P > s.max():
        raise ValueError("Pressure is outside saturation table range.")

    exact = df[df["Pressure"] == P]
    if not exact.empty:
        row = exact.iloc[0]
        return float(row[col_f]), float(row[col_g])

    lower = df[df["Pressure"] < P].iloc[-1]
    upper = df[df["Pressure"] > P].iloc[0]

    y_f = linear_interp(P, lower["Pressure"], upper["Pressure"], lower[col_f], upper[col_f])
    y_g = linear_interp(P, lower["Pressure"], upper["Pressure"], lower[col_g], upper[col_g])
    return float(y_f), float(y_g)


def sat_props_at_T(T, table_sat, prop_key):
    col_f, col_g = SAT_COLS[prop_key]

    df = table_sat.sort_values("Temperature").reset_index(drop=True)
    s = df["Temperature"]

    if T < s.min() or T > s.max():
        raise ValueError("Temperature is outside saturation table range.")

    exact = df[df["Temperature"] == T]
    if not exact.empty:
        row = exact.iloc[0]
        return float(row[col_f]), float(row[col_g])

    lower = df[df["Temperature"] < T].iloc[-1]
    upper = df[df["Temperature"] > T].iloc[0]

    y_f = linear_interp(T, lower["Temperature"], upper["Temperature"], lower[col_f], upper[col_f])
    y_g = linear_interp(T, lower["Temperature"], upper["Temperature"], lower[col_g], upper[col_g])
    return float(y_f), float(y_g)


# =========================
# Superheated / Compressed tables
# =========================
def load_superheated_table(P):
    ensure_tables_ready_from_local_zip()
    filename = os.path.join(ZIP_SUBDIR, f"B.1.3.{int(P)}.xlsx")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Superheated table not found for P={P} kPa (file {os.path.basename(filename)}).")

    raw = pd.read_excel(filename, header=None)
    df = raw.iloc[2:].reset_index(drop=True)
    df.columns = ["Temperature", "v", "u", "h", "s"]
    return df.astype(float)


def load_compressed_table(P):
    ensure_tables_ready_from_local_zip()
    filename = os.path.join(ZIP_SUBDIR, f"B.1.4.{int(P)}.xlsx")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Compressed table not found for P={P} kPa (file {os.path.basename(filename)}).")

    raw = pd.read_excel(filename, header=None)
    df = raw.iloc[2:].reset_index(drop=True)
    df.columns = ["Temperature", "v", "u", "h", "s"]
    return df.astype(float)


def list_available_pressures(prefix):
    ensure_tables_ready_from_local_zip()
    files = os.listdir(ZIP_SUBDIR)
    ps = []
    for f in files:
        if f.startswith(prefix):
            parts = f.split(".")
            try:
                ps.append(float(parts[-2]))
            except:
                pass
    return sorted(ps)


def find_two_pressures(P, prefix):
    pressures = list_available_pressures(prefix)
    if not pressures:
        raise ValueError(f"No tables found for prefix: {prefix}")

    if P in pressures:
        return P, P

    lower = [p for p in pressures if p < P]
    upper = [p for p in pressures if p > P]
    if not lower or not upper:
        raise ValueError("Pressure is outside available pressure grid for these tables.")
    return max(lower), min(upper)


def interpolate_property_in_T_table(T, table_PT, prop_key):
    df = table_PT.sort_values("Temperature").reset_index(drop=True)
    s = df["Temperature"]

    exact = df[df["Temperature"] == T]
    if not exact.empty:
        return float(exact[prop_key].iloc[0])

    if T < s.min() or T > s.max():
        raise ValueError("Temperature is outside the table range at this pressure.")

    lower = df[df["Temperature"] < T].iloc[-1]
    upper = df[df["Temperature"] > T].iloc[0]
    return float(linear_interp(T, lower["Temperature"], upper["Temperature"], lower[prop_key], upper[prop_key]))


def interpolate_2D_PT(P, T, prop_key, prefix):
    p1, p2 = find_two_pressures(P, prefix)
    loader = load_superheated_table if prefix == "B.1.3" else load_compressed_table

    if p1 == p2:
        return float(interpolate_property_in_T_table(T, loader(p1), prop_key))

    v1 = interpolate_property_in_T_table(T, loader(p1), prop_key)
    v2 = interpolate_property_in_T_table(T, loader(p2), prop_key)
    return float(linear_interp(P, p1, p2, v1, v2))


# =========================
# Inverse (given v/h/s -> T) in a P-fixed table
# =========================
def find_T_from_given_in_table(given_value, table_PT, given_key):
    df = table_PT.sort_values("Temperature").reset_index(drop=True)
    y = df[given_key]

    if given_value < y.min() or given_value > y.max():
        raise ValueError(f"{given_key} is outside the table range at this pressure.")

    for i in range(len(df) - 1):
        y1 = df.loc[i, given_key]
        y2 = df.loc[i + 1, given_key]
        if (y1 <= given_value <= y2) or (y2 <= given_value <= y1):
            T1 = df.loc[i, "Temperature"]
            T2 = df.loc[i + 1, "Temperature"]
            return float(inverse_linear_interp(given_value, y1, y2, T1, T2))

    raise ValueError("Could not determine Temperature from the given value.")


def property_from_given_in_table(given_value, table_PT, given_key, target_prop):
    T = find_T_from_given_in_table(given_value, table_PT, given_key)
    val = float(interpolate_property_in_T_table(T, table_PT, target_prop))
    return float(T), float(val)


def property_from_given_2D(P, given_value, prefix, given_key, target_prop):
    p1, p2 = find_two_pressures(P, prefix)
    loader = load_superheated_table if prefix == "B.1.3" else load_compressed_table

    if p1 == p2:
        return property_from_given_in_table(given_value, loader(p1), given_key, target_prop)

    T1, v1 = property_from_given_in_table(given_value, loader(p1), given_key, target_prop)
    T2, v2 = property_from_given_in_table(given_value, loader(p2), given_key, target_prop)

    T_out = float(linear_interp(P, p1, p2, T1, T2))
    v_out = float(linear_interp(P, p1, p2, v1, v2))
    return float(T_out), float(v_out)


# =========================
# Scenario calculators
# =========================
def calc_PT(table_sat, P, T, prop_key, x_text_optional=""):
    region, Tsat = classify_region_PT(P, T, table_sat, eps=EPS_SAT)
    out = {
        "scenario": "PT",
        "region": region,
        "P_kPa": float(P),
        "T_C": float(T),
        "Tsat_C": float(Tsat),
        "property": prop_key,
    }

    if region == "saturated":
        y_f, y_g = sat_props_at_P(P, table_sat, prop_key)
        out.update({"sat_f": float(y_f), "sat_g": float(y_g)})

        x_text = (x_text_optional or "").strip()
        if x_text != "":
            x = float(x_text)
            if not (0.0 <= x <= 1.0):
                raise ValueError("x must be between 0 and 1.")
            out.update({"x": float(x), "value": float(y_f + x * (y_g - y_f))})
        return out

    if region == "superheated":
        out["value"] = float(interpolate_2D_PT(P, T, prop_key, "B.1.3"))
        return out

    out["value"] = float(interpolate_2D_PT(P, T, prop_key, "B.1.4"))
    return out


def calc_Pv(table_sat, P, v, target_prop):
    v_f, v_g = sat_props_at_P(P, table_sat, "v")
    Tsat = Tsat_from_P(P, table_sat)

    out = {
        "scenario": "Pv",
        "P_kPa": float(P),
        "v_m3kg": float(v),
        "Tsat_C": float(Tsat),
        "property": target_prop,
    }

    if v_f <= v <= v_g:
        x = (v - v_f) / (v_g - v_f)
        y_f, y_g = sat_props_at_P(P, table_sat, target_prop)
        out.update({"region": "saturated", "x": float(x), "T_C": float(Tsat), "value": float(y_f + x * (y_g - y_f))})
        return out

    if v > v_g:
        out["region"] = "superheated"
        T_out, val = property_from_given_2D(P, v, "B.1.3", "v", target_prop)
        out.update({"T_C": float(T_out), "value": float(val)})
        return out

    out["region"] = "compressed"
    T_out, val = property_from_given_2D(P, v, "B.1.4", "v", target_prop)
    out.update({"T_C": float(T_out), "value": float(val)})
    return out


def calc_Ph(table_sat, P, h, target_prop):
    h_f, h_g = sat_props_at_P(P, table_sat, "h")
    Tsat = Tsat_from_P(P, table_sat)

    out = {
        "scenario": "Ph",
        "P_kPa": float(P),
        "h_kJkg": float(h),
        "Tsat_C": float(Tsat),
        "property": target_prop,
    }

    if h_f <= h <= h_g:
        x = (h - h_f) / (h_g - h_f)
        y_f, y_g = sat_props_at_P(P, table_sat, target_prop)
        out.update({"region": "saturated", "x": float(x), "T_C": float(Tsat), "value": float(y_f + x * (y_g - y_f))})
        return out

    if h > h_g:
        out["region"] = "superheated"
        T_out, val = property_from_given_2D(P, h, "B.1.3", "h", target_prop)
        out.update({"T_C": float(T_out), "value": float(val)})
        return out

    out["region"] = "compressed"
    T_out, val = property_from_given_2D(P, h, "B.1.4", "h", target_prop)
    out.update({"T_C": float(T_out), "value": float(val)})
    return out


def calc_Ps(table_sat, P, s, target_prop):
    s_f, s_g = sat_props_at_P(P, table_sat, "s")
    Tsat = Tsat_from_P(P, table_sat)

    out = {
        "scenario": "Ps",
        "P_kPa": float(P),
        "s_kJkgK": float(s),
        "Tsat_C": float(Tsat),
        "property": target_prop,
    }

    if s_f <= s <= s_g:
        x = (s - s_f) / (s_g - s_f)
        y_f, y_g = sat_props_at_P(P, table_sat, target_prop)
        out.update({"region": "saturated", "x": float(x), "T_C": float(Tsat), "value": float(y_f + x * (y_g - y_f))})
        return out

    if s > s_g:
        out["region"] = "superheated"
        T_out, val = property_from_given_2D(P, s, "B.1.3", "s", target_prop)
        out.update({"T_C": float(T_out), "value": float(val)})
        return out

    out["region"] = "compressed"
    T_out, val = property_from_given_2D(P, s, "B.1.4", "s", target_prop)
    out.update({"T_C": float(T_out), "value": float(val)})
    return out


def calc_Tx(table_sat, T, x, prop_key):
    if not (0.0 <= x <= 1.0):
        raise ValueError("x must be between 0 and 1.")

    P_sat = Psat_from_T(T, table_sat)
    y_f, y_g = sat_props_at_T(T, table_sat, prop_key)
    value = y_f + x * (y_g - y_f)

    return {
        "scenario": "Tx",
        "region": "saturated",
        "T_C": float(T),
        "x": float(x),
        "Psat_kPa": float(P_sat),
        "property": prop_key,
        "value": float(value),
        "sat_f": float(y_f),
        "sat_g": float(y_g),
    }


def calc_Px(table_sat, P, x, prop_key):
    if not (0.0 <= x <= 1.0):
        raise ValueError("x must be between 0 and 1.")

    Tsat = Tsat_from_P(P, table_sat)
    y_f, y_g = sat_props_at_P(P, table_sat, prop_key)
    value = y_f + x * (y_g - y_f)

    return {
        "scenario": "Px",
        "region": "saturated",
        "P_kPa": float(P),
        "T_C": float(Tsat),
        "Tsat_C": float(Tsat),
        "x": float(x),
        "property": prop_key,
        "value": float(value),
        "sat_f": float(y_f),
        "sat_g": float(y_g),
    }


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Thermo Tables — 6 Scenarios", layout="centered")
st.title("Thermodynamic Water Properties — 6 Scenarios")
st.caption("Local data only (Thermo.zip). No internet download. EPS_SAT is fixed at 0.01 °C.")

with st.spinner("Loading tables from local Thermo.zip..."):
    df_sat = load_saturation_table_B12()

tabs = st.tabs(["(P,T)", "(P,v)", "(P,h)", "(P,s)", "(T,x)", "(P,x)"])

# ---------------- (P,T) ----------------
with tabs[0]:
    st.subheader("Scenario (P, T)")
    with st.form("form_PT"):
        c1, c2 = st.columns(2)
        with c1:
            P = st.number_input("Pressure P (kPa)", min_value=0.0, value=100.0, step=10.0)
        with c2:
            T = st.number_input("Temperature T (°C)", value=100.0, step=1.0)

        prop = st.selectbox("Property", ["h", "u", "v", "s"], index=0)
        submit = st.form_submit_button("Calculate")

    if "pt_out" not in st.session_state:
        st.session_state.pt_out = None

    if submit:
        try:
            st.session_state.pt_out = calc_PT(df_sat, P, T, prop, x_text_optional="")
        except Exception as e:
            st.session_state.pt_out = None
            st.error(str(e))

    out = st.session_state.pt_out
    if out:
        st.json(out)

        if out["region"] == "saturated":
            x_text = st.text_input("Quality x (0–1), optional (only for saturated):", value="", key="pt_x")
            if x_text.strip() != "":
                try:
                    st.json(calc_PT(df_sat, out["P_kPa"], out["T_C"], out["property"], x_text_optional=x_text))
                except Exception as e:
                    st.error(str(e))

# ---------------- (P,v) ----------------
with tabs[1]:
    st.subheader("Scenario (P, v)")
    with st.form("form_Pv"):
        c1, c2 = st.columns(2)
        with c1:
            P = st.number_input("Pressure P (kPa)", min_value=0.0, value=100.0, step=10.0, key="pv_P")
        with c2:
            v = st.number_input("Specific volume v (m^3/kg)", min_value=0.0, value=1.0, step=0.01, key="pv_v")

        prop = st.selectbox("Property", ["h", "u", "s"], index=0, key="pv_prop")
        submit = st.form_submit_button("Calculate")

    if submit:
        try:
            st.json(calc_Pv(df_sat, P, v, prop))
        except Exception as e:
            st.error(str(e))

# ---------------- (P,h) ----------------
with tabs[2]:
    st.subheader("Scenario (P, h)")
    with st.form("form_Ph"):
        c1, c2 = st.columns(2)
        with c1:
            P = st.number_input("Pressure P (kPa)", min_value=0.0, value=100.0, step=10.0, key="ph_P")
        with c2:
            h = st.number_input("Enthalpy h (kJ/kg)", value=500.0, step=1.0, key="ph_h")

        prop = st.selectbox("Property", ["v", "u", "s"], index=0, key="ph_prop")
        submit = st.form_submit_button("Calculate")

    if submit:
        try:
            st.json(calc_Ph(df_sat, P, h, prop))
        except Exception as e:
            st.error(str(e))

# ---------------- (P,s) ----------------
with tabs[3]:
    st.subheader("Scenario (P, s)")
    with st.form("form_Ps"):
        c1, c2 = st.columns(2)
        with c1:
            P = st.number_input("Pressure P (kPa)", min_value=0.0, value=100.0, step=10.0, key="ps_P")
        with c2:
            s = st.number_input("Entropy s (kJ/kg.K)", value=1.0, step=0.01, key="ps_s")

        prop = st.selectbox("Property", ["v", "u", "h"], index=0, key="ps_prop")
        submit = st.form_submit_button("Calculate")

    if submit:
        try:
            st.json(calc_Ps(df_sat, P, s, prop))
        except Exception as e:
            st.error(str(e))

# ---------------- (T,x) ----------------
with tabs[4]:
    st.subheader("Scenario (T, x) — Saturated only")
    with st.form("form_Tx"):
        c1, c2 = st.columns(2)
        with c1:
            T = st.number_input("Temperature T (°C)", value=100.0, step=1.0, key="tx_T")
        with c2:
            x = st.number_input("Quality x (0..1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="tx_x")

        prop = st.selectbox("Property", ["v", "u", "h", "s"], index=0, key="tx_prop")
        submit = st.form_submit_button("Calculate")

    if submit:
        try:
            st.json(calc_Tx(df_sat, T, x, prop))
        except Exception as e:
            st.error(str(e))

# ---------------- (P,x) ----------------
with tabs[5]:
    st.subheader("Scenario (P, x) — Saturated only")
    with st.form("form_Px"):
        c1, c2 = st.columns(2)
        with c1:
            P = st.number_input("Pressure P (kPa)", min_value=0.0, value=100.0, step=10.0, key="px_P")
        with c2:
            x = st.number_input("Quality x (0..1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="px_x")

        prop = st.selectbox("Property", ["v", "u", "h", "s"], index=0, key="px_prop")
        submit = st.form_submit_button("Calculate")

    if submit:
        try:
            st.json(calc_Px(df_sat, P, x, prop))
        except Exception as e:
            st.error(str(e))

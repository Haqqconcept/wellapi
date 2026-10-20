"""
app.py - WellAPI web frontend (Streamlit).

Talks to the WellAPI server over HTTP, so it exercises the real login, API
keys, credits and billing, exactly as an outside customer would.

Run (two Command Prompt windows, both from the project folder):
    1)  python -m uvicorn main:app --reload --timeout-keep-alive 120
    2)  streamlit run frontend/app.py

Use a different server with:   set WELLAPI_URL=https://your-app.up.railway.app
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from client import ApiError, WellAPIClient

st.set_page_config(page_title="WellAPI", page_icon="🛢️", layout="wide")

DEFAULT_URL = os.environ.get("WELLAPI_URL", "http://127.0.0.1:8000")
COST = {"ml": 1, "physics": 4, "nodal": 8}
Q_VALIDATED_MAX = 1500

# key, label, unit, default, step, trained range (low, high)
WELL_FIELDS = [
    ("QL",       "Liquid rate",           "STB/day",  330.0,  10.0,  (100, 1500)),
    ("WOR",      "Water-oil ratio",       "STB/STB",  1.5,    0.1,   (0, 5)),
    ("GLR",      "Gas-liquid ratio",      "scf/STB",  273.0,  10.0,  (100, 800)),
    ("oil_sg",   "Oil specific gravity",  "",         0.85,   0.01,  (0.80, 0.90)),
    ("water_sg", "Water specific gravity", "",        1.05,   0.01,  (1.03, 1.07)),
    ("gas_sg",   "Gas specific gravity",  "air = 1",  0.75,   0.01,  (0.65, 0.80)),
    ("T",        "Temperature",           "°F",       150.0,  5.0,   (100, 200)),
    ("API",      "Oil API gravity",       "°API",     35.0,   1.0,   (25, 45)),
    ("Bw",       "Water FVF",             "bbl/STB",  1.01,   0.01,  (1.00, 1.02)),
    ("d",        "Tubing inside diameter", "in",      2.441,  0.001, (1.995, 3.476)),
    ("depth",    "Well depth",            "ft",       5000.0, 100.0, (2000, 8000)),
    ("Pwf",      "Bottomhole flowing pressure", "psia", 1000.0, 10.0, (200, 2000)),
]


# ── session helpers ───────────────────────────────────────────────────────────

def init_state():
    for k, v in {"token": None, "api_key": None, "user": None, "credits": None,
                 "last_predict": None, "last_nodal": None}.items():
        st.session_state.setdefault(k, v)


def make_client(base_url: str) -> WellAPIClient:
    return WellAPIClient(base_url, st.session_state.token, st.session_state.api_key)


def refresh_account(client: WellAPIClient):
    me = client.me()
    st.session_state.user = me["user"]
    st.session_state.credits = me["credits"]
    return me


def logout():
    for k in ("token", "api_key", "user", "credits", "last_predict", "last_nodal"):
        st.session_state[k] = None


def attempt(fn, *args, **kwargs):
    """Run an API call; show a friendly error instead of a traceback."""
    try:
        return fn(*args, **kwargs)
    except ApiError as e:
        st.error(str(e))
        if e.status == 401 and st.session_state.token:
            st.caption("Your session may have expired. Log out and log in again.")
        return None


# ── input widgets ─────────────────────────────────────────────────────────────

def well_inputs(prefix: str, include_rate_and_pwf: bool = True) -> dict:
    """Twelve (or ten) well parameters laid out in three columns."""
    fields = [f for f in WELL_FIELDS if include_rate_and_pwf or f[0] not in ("QL", "Pwf")]
    cols = st.columns(3)
    out = {}
    for i, (key, label, unit, default, step, (lo, hi)) in enumerate(fields):
        title = f"{label} ({unit})" if unit else label
        out[key] = cols[i % 3].number_input(
            title, value=float(default), step=float(step), format="%g",
            min_value=0.0, key=f"{prefix}_{key}",
            help=f"Trained range: {lo} to {hi}. Values outside it still run but are flagged.")
    return out


# ── charts ────────────────────────────────────────────────────────────────────

def plot_profile(table: list, thp: float):
    df = pd.DataFrame(table)
    depth = list(df["depth_to_top"]) + [0.0]
    pressure = list(df["P_assumed"]) + [thp]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(pressure, depth, color="#1C7293", linewidth=2.2)
    ax.scatter([thp], [0], color="#E8A33D", zorder=5, s=60, label=f"Wellhead (THP {thp:.1f} psia)")
    ax.invert_yaxis()
    ax.set_xlabel("Pressure (psia)")
    ax.set_ylabel("Depth below wellhead (ft)")
    ax.set_title("Pressure profile along the tubing")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    return fig


def plot_nodal(res: dict):
    ipr = pd.DataFrame(res["ipr_curve"])
    vlp = pd.DataFrame(res["vlp_curve"])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(ipr["Q"], ipr["Pwf"], color="#1C7293", linewidth=2.2, label="IPR (reservoir inflow)")
    ax.plot(vlp["Q"], vlp["Pwf"], color="#B14A3D", linewidth=2.2, label="VLP (tubing outflow)")
    op = res["operating_point"]
    if op["found"]:
        ax.scatter([op["Q"]], [op["Pwf"]], color="#E8A33D", s=110, zorder=5, edgecolor="black",
                   label=f"Operating point ({op['Q']:.0f} STB/day, {op['Pwf']:.0f} psia)")
    ax.axvline(Q_VALIDATED_MAX, color="gray", linestyle=":", linewidth=1)
    ax.text(Q_VALIDATED_MAX, ax.get_ylim()[0], " validated\n rate limit", fontsize=8,
            color="gray", va="bottom")
    ax.set_xlabel("Liquid rate (STB/day)")
    ax.set_ylabel("Bottomhole flowing pressure (psia)")
    ax.set_title("Nodal analysis: where inflow meets outflow")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


# ── pages ─────────────────────────────────────────────────────────────────────

def page_predict(client: WellAPIClient):
    st.header("Predict tubing head pressure")
    st.caption(f"ML surrogate: instant, {COST['ml']} credit. Physics engine: full pressure profile, "
               f"{COST['physics']} credits.")
    well = well_inputs("pred")

    c1, c2, c3 = st.columns(3)
    mode = None
    if c1.button(f"⚡ ML prediction ({COST['ml']} credit)", type="primary", key="btn_ml"):
        mode = "ml"
    if c2.button(f"🧮 Physics engine ({COST['physics']} credits)", key="btn_physics"):
        mode = "physics"
    if c3.button(f"⚖️ Compare both ({COST['ml'] + COST['physics']} credits)", key="btn_both"):
        mode = "both"

    if mode:
        result = {"well": well, "ml": None, "physics": None}
        with st.spinner("Calling the WellAPI server..."):
            if mode in ("ml", "both"):
                result["ml"] = attempt(client.predict_ml, well)
                if result["ml"]:
                    st.session_state.credits = result["ml"]["credits_remaining"]
            if mode in ("physics", "both"):
                result["physics"] = attempt(client.predict_physics, well)
                if result["physics"]:
                    st.session_state.credits = result["physics"]["credits_remaining"]
        st.session_state.last_predict = result

    res = st.session_state.last_predict
    if not res or not (res["ml"] or res["physics"]):
        return
    st.divider()
    ml, ph = res["ml"], res["physics"]

    cols = st.columns(3 if (ml and ph) else 2)
    i = 0
    if ml:
        fell_back = ml.get("method") == "physics_fallback"
        cols[i].metric("THP (physics fallback)" if fell_back else "ML surrogate THP", f"{ml['THP']:.1f} psia")
        cols[i].caption(f"Confidence: **{ml['confidence']}** · {ml['response_ms']:.1f} ms")
        i += 1
    if ph:
        cols[i].metric("Physics engine THP", f"{ph['THP']:.1f} psia")
        cols[i].caption(f"Confidence: **{ph['confidence']}** · {ph['traverse_steps']} steps · "
                        f"{ph['response_ms']:.1f} ms")
        i += 1
    if ml and ph:
        diff = ml["THP"] - ph["THP"]
        cols[i].metric("Difference (ML − physics)", f"{diff:+.1f} psia",
                       f"{abs(diff) / ph['THP'] * 100:.2f}%", delta_color="off")

    if ml and ml.get("note"):
        st.info(ml["note"])

    out_of_range = sorted(set((ml or {}).get("out_of_range", []) + (ph or {}).get("out_of_range", [])))
    if out_of_range:
        st.warning("Outside the range the model was trained on: " + ", ".join(out_of_range) +
                   ". Treat these results with caution.")

    if ph and ph.get("traverse_table"):
        left, right = st.columns([3, 2])
        with left:
            fig = plot_profile(ph["traverse_table"], ph["THP"])
            st.pyplot(fig)
            plt.close(fig)
        with right:
            df = pd.DataFrame(ph["traverse_table"])
            st.markdown("**Traverse table**")
            st.dataframe(df, height=380)
            st.download_button("Download table (CSV)", df.to_csv(index=False),
                               "traverse_table.csv", "text/csv")


def page_nodal(client: WellAPIClient):
    st.header("Nodal analysis")
    st.caption(f"Finds the rate and bottomhole pressure where reservoir inflow (IPR) meets the tubing's "
               f"required outflow (VLP). Costs {COST['nodal']} credits. Results are most reliable up to "
               f"{Q_VALIDATED_MAX} STB/day.")

    st.subheader("Well and fluid")
    well = well_inputs("nod", include_rate_and_pwf=False)

    st.subheader("Reservoir inflow (IPR)")
    a, b, c = st.columns(3)
    model = a.selectbox("IPR model", key="ipr_model", options=["vogel", "darcy", "fetkovich"],
                        format_func=lambda m: {"vogel": "Vogel", "darcy": "Darcy (linear)",
                                               "fetkovich": "Fetkovich"}[m])
    Pr = b.number_input("Reservoir pressure Pr (psia)", value=3000.0, step=50.0, min_value=1.0, format="%g")
    ipr = {"model": model, "Pr": Pr}
    if model == "vogel":
        ipr["Qmax"] = c.number_input("Absolute open flow Qmax (STB/day)", value=1800.0, step=50.0,
                                     min_value=1.0, format="%g")
    elif model == "darcy":
        ipr["PI"] = c.number_input("Productivity index PI (STB/day/psi)", value=0.8, step=0.05,
                                   min_value=0.001, format="%g")
    else:
        ipr["n"] = c.number_input("Exponent n", value=1.0, step=0.05, min_value=0.1, max_value=1.5, format="%g")
        how = st.radio("Fetkovich coefficient", ["Calibrate from a test point", "Enter C directly"],
                       horizontal=True)
        if how == "Enter C directly":
            ipr["C"] = st.number_input("C", value=0.0005, step=0.0001, min_value=1e-9, format="%g")
        else:
            t1, t2 = st.columns(2)
            ipr["Pwf_test"] = t1.number_input("Test flowing pressure (psia)", value=1500.0, step=50.0,
                                              min_value=0.0, format="%g")
            ipr["Q_test"] = t2.number_input("Test rate (STB/day)", value=900.0, step=50.0,
                                            min_value=1.0, format="%g")

    st.subheader("Operating conditions")
    d1, d2, d3, d4 = st.columns(4)
    target = d1.number_input("Target wellhead pressure (psia)", value=200.0, step=10.0, min_value=1.0, format="%g")
    q_min = d2.number_input("Rate sweep from (STB/day)", value=100.0, step=50.0, min_value=1.0, format="%g")
    q_max = d3.number_input("Rate sweep to (STB/day)", value=1500.0, step=50.0, min_value=2.0, format="%g")
    n_pts = d4.slider("Curve points", 4, 25, 10)

    if st.button(f"Run nodal analysis ({COST['nodal']} credits)", type="primary", key="btn_nodal"):
        req = {"well": well, "ipr": ipr, "target_thp": target,
               "q_min": q_min, "q_max": q_max, "n_points": int(n_pts)}
        with st.spinner("Solving inflow and outflow..."):
            res = attempt(client.nodal, req)
        if res:
            st.session_state.credits = res["credits_remaining"]
        st.session_state.last_nodal = res

    res = st.session_state.last_nodal
    if not res:
        return
    st.divider()
    op = res["operating_point"]
    if op["found"]:
        m1, m2 = st.columns(2)
        m1.metric("Operating rate", f"{op['Q']:.0f} STB/day")
        m2.metric("Operating bottomhole pressure", f"{op['Pwf']:.0f} psia")
    else:
        st.info(op.get("note") or "No operating point found in this range.")
    for w in res["warnings"]:
        st.warning(w)
    fig = plot_nodal(res)
    st.pyplot(fig)
    plt.close(fig)
    with st.expander("Curve data"):
        both = pd.DataFrame(res["ipr_curve"]).rename(columns={"Pwf": "IPR Pwf"}).merge(
            pd.DataFrame(res["vlp_curve"]).rename(columns={"Pwf": "VLP Pwf"}), on="Q")
        st.dataframe(both)
        st.download_button("Download curves (CSV)", both.to_csv(index=False),
                           "nodal_curves.csv", "text/csv")


def page_account(client: WellAPIClient):
    st.header("Account")
    me = attempt(refresh_account, client)
    if not me:
        return
    user = me["user"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Credits", me["credits"])
    m2.metric("Total calls", me["total_calls"])
    m3.metric("Signed in as", user["name"])
    st.caption(user["email"] + (f" · {user['organisation']}" if user.get("organisation") else ""))

    st.subheader("API keys")
    keys = attempt(client.list_keys) or []
    for k in keys:
        status = "active" if k["is_active"] else "inactive"
        with st.expander(f"{k['label']}  ·  {status}"):
            st.code(k["key"], language=None)
            st.caption(f"Created {str(k['created_at'])[:19]} · last used "
                       f"{str(k['last_used'])[:19] if k.get('last_used') else 'never'}")
            if k["is_active"] and st.button("Deactivate", key=f"del_{k['id']}"):
                if attempt(client.delete_key, k["id"]) is None:
                    st.session_state.api_key = None
                    st.rerun()
    label = st.text_input("New key label", value="My key")
    if st.button("Create API key"):
        if attempt(client.create_key, label):
            st.rerun()

    st.subheader("Usage history")
    usage = attempt(client.usage, 50) or []
    if usage:
        df = pd.DataFrame(usage)[["created_at", "endpoint", "credits_used", "thp_result",
                                  "confidence", "response_ms"]]
        st.dataframe(df)
    else:
        st.caption("No calls yet.")

    st.subheader("Buy credits")
    bundles = attempt(client.bundles) or []
    if bundles:
        st.dataframe(pd.DataFrame(bundles))
        c1, c2, c3 = st.columns([2, 2, 1])
        bundle = c1.selectbox("Bundle", [b["bundle_id"] for b in bundles])
        gateway = c2.selectbox("Payment method", ["paystack", "stripe"])
        if c3.button("Start payment"):
            pay = attempt(client.topup, bundle, gateway)
            if pay:
                st.success(f"{pay['credits']} credits for ${pay['amount_usd']:.2f}. Complete the payment here:")
                st.markdown(f"[Open payment page]({pay['payment_url']})")


# ── logged-out view ───────────────────────────────────────────────────────────

def view_logged_out(base_url: str):
    st.title("🛢️ WellAPI")
    st.markdown("**Physics-informed AI for well performance.** Predict tubing head pressure "
                "and run nodal analysis through an API, without an enterprise license.")
    tab_login, tab_register = st.tabs(["Log in", "Create account"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            pw = st.text_input("Password", type="password", key="login_pw")
            go = st.form_submit_button("Log in", type="primary")
        if go:
            client = WellAPIClient(base_url)
            try:
                client.login(email.strip(), pw)
                client.ensure_api_key()
            except ApiError as e:
                st.error(str(e))
            else:
                st.session_state.token = client.token
                st.session_state.api_key = client.api_key
                attempt(refresh_account, client)
                st.rerun()

    with tab_register:
        with st.form("register_form"):
            r_name = st.text_input("Full name", key="reg_name")
            r_email = st.text_input("Email ", key="reg_email")
            r_org = st.text_input("Organisation (optional)", key="reg_org")
            r_pw = st.text_input("Password (8+ characters)", type="password", key="reg_pw")
            reg = st.form_submit_button("Create account", type="primary")
        if reg:
            try:
                user = WellAPIClient(base_url).register(r_email.strip(), r_name.strip(), r_pw, r_org)
            except ApiError as e:
                st.error(str(e))
            else:
                st.success(f"Account created with {user['credits']} free credits. You can log in now.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    init_state()
    st.sidebar.markdown("### 🛢️ WellAPI")
    base_url = st.sidebar.text_input("Server address", value=DEFAULT_URL)

    probe = WellAPIClient(base_url, timeout=4)
    try:
        probe.health()
        st.sidebar.success("Server online")
    except ApiError as e:
        st.sidebar.error("Server offline")
        st.error(str(e))
        st.stop()

    if not st.session_state.token:
        view_logged_out(base_url)
        return

    client = make_client(base_url)
    user = st.session_state.user or {}
    st.sidebar.write(f"Signed in as **{user.get('name', '')}**")
    credits_slot = st.sidebar.empty()      # filled after the page runs, so it is never a step behind
    page = st.sidebar.radio("Go to", ["Predict THP", "Nodal analysis", "Account"])
    if st.sidebar.button("Log out"):
        logout()
        st.rerun()

    {"Predict THP": page_predict, "Nodal analysis": page_nodal, "Account": page_account}[page](client)
    credits_slot.metric("Credits", st.session_state.credits if st.session_state.credits is not None else "-")


main()

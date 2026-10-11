import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Predictive Maintenance", layout="wide")

SENSORS = ["vibration", "temperature", "rotation", "voltage", "pressure"]


@st.cache_data
def load(name, columns=None):
    return pd.read_parquet(f"data/processed/{name}.parquet", columns=columns)


machines = load("machines")
failures = load("failures")
maint = load("maint")
errors = load("errors")
risk = load("risk")
importance = load("feature_importance")
telemetry = load("telemetry", ["datetime", "machineID"] + SENSORS + ["fail_within_24h"])


@st.cache_data
def sensor_shift(ids):
    t = telemetry[telemetry["machineID"].isin(ids)]
    m = t.groupby("fail_within_24h")[SENSORS].mean()
    if 1 not in m.index:
        return pd.DataFrame({"sensor": [], "change_pct": []})
    pct = (m.loc[1] - m.loc[0]) / m.loc[0] * 100
    return pd.DataFrame({"sensor": pct.index, "change_pct": pct.values.round(1)})


@st.cache_data
def days_since_replacement():
    mc = maint.rename(columns={"comp": "failure"})[["machineID", "failure", "datetime"]].copy()
    mc["maint_date"] = mc["datetime"]
    mc = mc.sort_values("datetime")
    g = pd.merge_asof(failures.sort_values("datetime"), mc, on="datetime",
                      by=["machineID", "failure"], direction="backward",
                      allow_exact_matches=False)
    g["days_since_same_comp"] = (g["datetime"] - g["maint_date"]).dt.days
    return g


st.title("Predictive Maintenance Dashboard")
st.caption("Machine failures, sensor behaviour and failure risk for 100 factory machines (2015). "
           "Temperature and operating hours are AI-generated columns.")

# ---------- Sidebar filter ----------
st.sidebar.header("Filters")
all_models = sorted(machines["model"].unique())
models = st.sidebar.multiselect("Machine model", all_models, default=all_models)
if not models:
    st.warning("Select at least one machine model in the sidebar.")
    st.stop()

ids = machines[machines["model"].isin(models)]["machineID"]
f = failures[failures["machineID"].isin(ids)]
r = risk[risk["machineID"].isin(ids)]

# ---------- Summary numbers ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Machines", len(ids))
c2.metric("Total failures", len(f))
c3.metric("Failures per machine", round(len(f) / len(ids), 1))
c4.metric("High-risk machines (risk > 0.4)", int((r["risk"] > 0.4).sum()))

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Failures", "Sensor trends", "Warning signs", "Age & errors", "Maintenance", "Risk scores"])

# ---------- Tab 1: Failures ----------
with tab1:
    left, right = st.columns(2)
    by_part = f["failure"].value_counts().reset_index()
    left.plotly_chart(px.bar(by_part, x="failure", y="count", title="Failures by component"))

    fm = f.merge(machines, on="machineID")
    per_model = (fm.groupby("model").size() / machines[machines["model"].isin(models)]
                 .groupby("model").size()).reset_index()
    per_model.columns = ["model", "failures_per_machine"]
    right.plotly_chart(px.bar(per_model, x="model", y="failures_per_machine",
                              title="Failures per machine, by model"))

# ---------- Tab 2: Sensor trends ----------
with tab2:
    a, b = st.columns(2)
    m = a.selectbox("Machine", sorted(ids))
    sensor = b.selectbox("Sensor", SENSORS)

    t = (telemetry[telemetry["machineID"] == m].set_index("datetime")[sensor]
         .resample("D").mean().reset_index())
    fig = px.line(t, x="datetime", y=sensor, title=f"Machine {m}: daily average {sensor}")

    fail_days = failures[failures["machineID"] == m]["datetime"].dt.floor("D").unique()
    pts = t[t["datetime"].isin(fail_days)]
    fig.add_scatter(x=pts["datetime"], y=pts[sensor], mode="markers",
                    marker=dict(size=11, color="red"), name="Failure")
    st.plotly_chart(fig)
    st.caption("Red dots mark days when this machine failed. Temperature is AI-generated.")

# ---------- Tab 3: Warning signs ----------
with tab3:
    shift = sensor_shift(tuple(ids))
    fig = px.bar(shift, x="sensor", y="change_pct",
                 title="Change in sensor average during the 24 hours before a failure (%)")
    st.plotly_chart(fig)
    st.markdown("Rotation falls while vibration, pressure and voltage rise before a failure. "
                "No single sensor is a reliable alarm on its own, so the model combines them.")
    st.caption("Temperature is AI-generated to rise before failures, so its change reflects "
               "the synthetic design. The other four sensors come from the original Azure data.")

# ---------- Tab 4: Age and errors ----------
with tab4:
    fc = f.groupby("machineID").size().reset_index(name="failure_count")
    ec = errors.groupby("machineID").size().reset_index(name="error_count")
    pm = machines[machines["machineID"].isin(ids)]
    pm = pm.merge(fc, on="machineID", how="left").merge(ec, on="machineID", how="left")
    pm = pm.fillna({"failure_count": 0, "error_count": 0})

    left, right = st.columns(2)
    left.plotly_chart(px.scatter(pm, x="age", y="failure_count", color="model",
                                 title="Machine age vs number of failures"))
    pm["age_group"] = pm["age"].ge(14).map({True: "14+ years", False: "Under 14 years"})
    grp = pm.groupby("age_group")["failure_count"].mean().round(1).reset_index()
    right.plotly_chart(px.bar(grp, x="age_group", y="failure_count",
                              title="Average failures per machine, by age group"))

    st.plotly_chart(px.scatter(pm, x="error_count", y="failure_count", color="model",
                               trendline="ols", trendline_scope="overall",
                               title="Errors vs failures per machine"))
    st.caption(f"Correlation between error count and failure count: "
               f"{pm['error_count'].corr(pm['failure_count']):.2f}")

# ---------- Tab 5: Maintenance ----------
with tab5:
    md = maint.drop_duplicates(["machineID", "datetime"]).sort_values(["machineID", "datetime"])
    cycle = md.groupby("machineID")["datetime"].diff().dt.days.dropna().astype(int)
    cyc = cycle.value_counts().head(8).sort_index().reset_index()
    cyc.columns = ["days_between_maintenance", "count"]
    left, right = st.columns(2)
    fig1 = px.bar(cyc, x="days_between_maintenance", y="count",
                  title="Gap between consecutive maintenance dates (days)")
    fig1.update_xaxes(type="category")
    left.plotly_chart(fig1)

    g = days_since_replacement()
    g = g[g["machineID"].isin(ids)]
    right.plotly_chart(px.histogram(g, x="days_since_same_comp", nbins=40,
                                    title="Days since the same component was last replaced, at failure"))
    st.markdown("Maintenance follows a fixed 15-day cycle. Failures occur a median of "
                f"{g['days_since_same_comp'].median():.0f} days after the failing component was last replaced.")

# ---------- Tab 6: Risk scores ----------
with tab6:
    top = r.sort_values("risk", ascending=False).head(10).copy()
    top["machine"] = "Machine " + top["machineID"].astype(str)
    st.plotly_chart(px.bar(top, x="machine", y="risk", color="model",
                           title="Top 10 riskiest machines"))
    st.plotly_chart(px.bar(importance.sort_values("importance"), x="importance", y="feature",
                           orientation="h", title="Which inputs matter most for the model"))
    st.dataframe(r.sort_values("risk", ascending=False).round(3), hide_index=True)
"""U.S. Recession Probability Dashboard (Streamlit).

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from recession.config import CATEGORY_LABELS
from recession.history import load_history
from recession.labels import HORIZONS
from recession.paths import MODELS_DIR
from recession.predict import load_bundle, run_prediction

# ----------------------------------------------------------------- palette --
BLUE = "#2a78d6"      # primary series
RED = "#e34948"       # risk-up / diverging warm pole
ORANGE = "#eb6834"    # categorical slot 2
AQUA = "#1baf7a"      # categorical slot 3
YELLOW = "#eda100"    # categorical slot 4
MUTED = "#898781"
GRID = "#e1e0d9"
INK = "#0b0b0b"
INK2 = "#52514e"
SURFACE = "#fcfcfb"
SHADE = "rgba(137,135,129,0.18)"   # NBER recession shading
BAND_FILL = "rgba(42,120,214,0.15)"

MODEL_COLORS = {  # fixed categorical order, never cycled
    "A_yield_curve": ORANGE, "B_elastic_net": BLUE,
    "C_grad_boost": AQUA, "D_ensemble": YELLOW,
}

st.set_page_config(
    page_title="U.S. Recession Probability", page_icon="📉", layout="wide"
)


def style_fig(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=SURFACE,
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif',
                  color=INK2, size=13),
        margin=dict(l=10, r=10, t=70, b=10),
        hovermode="x unified",
        title=dict(yanchor="top", y=0.99, x=0, font=dict(size=15, color=INK)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0,
                    font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor="#c3c2b7", zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor="#c3c2b7", zeroline=False)
    return fig


def pct(x: float | None, digits: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{digits}f}%"


def pp(x: float | None, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{100 * x:+.{digits}f} pp"


# ------------------------------------------------------------------ loaders --
@st.cache_resource(show_spinner=False)
def _bundle():
    return load_bundle()


@st.cache_data(show_spinner="Refreshing data from FRED/ALFRED and running the model…", ttl=6 * 3600)
def _prediction(offline: bool):
    out = run_prediction(ttl_hours=6.0, offline=offline, save=True)
    return out


@st.cache_data(show_spinner=False)
def _backtest_artifacts():
    results = json.loads((MODELS_DIR / "backtest_results.json").read_text())
    oos = pd.read_parquet(MODELS_DIR / "oos_predictions.parquet")
    oos.index = pd.to_datetime(oos.index)
    return results, oos


def recession_spans(flag: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    f = flag.astype(bool)
    spans, start = [], None
    for t, v in f.items():
        if v and start is None:
            start = t
        elif not v and start is not None:
            spans.append((start, t)); start = None
    if start is not None:
        spans.append((start, f.index.max()))
    return spans


# ---------------------------------------------------------------- sidebar ---
st.sidebar.title("Controls")
offline = st.sidebar.toggle(
    "Offline mode (cache only)", value=False,
    help="Skip API calls and use the local cache; staleness is flagged in Data Health.",
)
if st.sidebar.button("🔄 Refresh now", use_container_width=True):
    _prediction.clear()
st.sidebar.caption(
    "Data: FRED / ALFRED (St. Louis Fed). Training uses first-release vintage "
    "data where available."
)

try:
    bundle = _bundle()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

try:
    out = _prediction(offline)
except Exception as exc:  # graceful degradation: cache-only retry
    st.warning(f"Live refresh failed ({exc}); falling back to cached data.")
    out = _prediction(True)

pred = out["prediction"]
curve = out["curve"]
probs = pred["probabilities"]
band = {int(k): v for k, v in pred.get("band", {}).items()}

# ----------------------------------------------------------------- header ---
left, right = st.columns([3, 2])
with left:
    st.title("U.S. Recession Probability")
    st.caption(
        "Probability that the U.S. economy enters an NBER-style recession within "
        "each horizon, from a calibrated hazard model on point-in-time data. "
        "**An analytical estimate with real uncertainty — not an oracle.**"
    )
with right:
    st.markdown(
        f"**Overall risk: {pred['overall_classification']}** &nbsp;·&nbsp; "
        f"Model confidence: **{pred['confidence']}**  \n"
        f"Data through **{pred['data_date']}** · refreshed "
        f"{pd.to_datetime(pred['generated_at']).strftime('%Y-%m-%d %H:%M UTC')}  \n"
        f"Model `{pred['model_name']}` v{pred['model_version']} · "
        f"calibration `{pred['calibration']}` · trained through {pred['train_end']}"
    )
if pred.get("currently_in_recession_labeled"):
    st.warning(
        "The latest published NBER label already marks the economy as in recession; "
        "onset probabilities below are conditional on not being in one."
    )

# ------------------------------------------------------ Section 1: headline --
st.subheader("Recession probability by horizon")
cols = st.columns(5)
for c, h in zip(cols, HORIZONS):
    with c:
        st.metric(
            label=f"{h} days",
            value=pct(probs[h]),
            delta=pp(pred["changes"]["1d"].get(h)) + " vs yesterday",
            delta_color="inverse",
        )
        lohi = band.get(h)
        st.caption(
            f"{pred['classification'][h]}"
            + (f" · range {pct(lohi[0])}–{pct(lohi[1])}" if lohi else "")
            + f"  \n7d {pp(pred['changes']['7d'].get(h))} · 30d {pp(pred['changes']['30d'].get(h))}"
        )

term = go.Figure()
xs = list(HORIZONS)
if band:
    term.add_trace(go.Scatter(
        x=xs + xs[::-1],
        y=[band[h][1] * 100 for h in xs] + [band[h][0] * 100 for h in xs][::-1],
        fill="toself", fillcolor=BAND_FILL, line=dict(width=0),
        name="10–90% bootstrap band", hoverinfo="skip",
    ))
term.add_trace(go.Scatter(
    x=xs, y=[probs[h] * 100 for h in xs], mode="lines+markers+text",
    text=[pct(probs[h]) for h in xs], textposition="top center",
    textfont=dict(color=INK2),
    line=dict(color=BLUE, width=2), marker=dict(size=9, color=BLUE),
    name="Cumulative onset probability",
))
term.update_layout(
    title="Term structure — cumulative probability of recession onset",
    showlegend=False,
)
term.update_xaxes(title="Horizon (days)", tickvals=xs)
term.update_yaxes(title="Probability (%)", rangemode="tozero")
st.plotly_chart(style_fig(term, 300), use_container_width=True)
st.caption(
    "Shaded band = 10–90% bootstrap range. Monotonicity (P15 ≤ P30 ≤ … ≤ P90) is "
    "structural: horizons share one calibrated 90-day hazard mapped through "
    "fitted exponents θ (see Methodology)."
)

# ---------------------------------------------------- Section 2: why moved ---
st.subheader("Why risk is where it is — and why it moved")
st.caption(
    "All horizons share one calibrated 90-day hazard, so the same factors drive "
    "every card above — shorter horizons scale them through the fitted term "
    "structure. Attributions below are computed from the production model itself "
    "(feature substitution), not narrated."
)
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Current drivers vs neutral (median) conditions**")
    up = pred["factors"]["increasing"][:5]
    dn = pred["factors"]["decreasing"][:5]
    st.markdown("*Raising recession risk*")
    if up:
        for f in up:
            st.markdown(f"- 🔺 {f['label']} — **{f['delta_pp']:+.1f} pp** ({f['category']})")
    else:
        st.markdown("- none material")
    st.markdown("*Lowering recession risk*")
    if dn:
        for f in dn:
            st.markdown(f"- 🔻 {f['label']} — **{f['delta_pp']:+.1f} pp** ({f['category']})")
    else:
        st.markdown("- none material")
with c2:
    st.markdown(
        f"**Change over the past week** &nbsp;·&nbsp; 90-day probability "
        f"{pp(pred['changes']['7d'].get(90))}"
    )
    fc = pred.get("factors_change_7d") or {}
    ups, dns = fc.get("increasing", []), fc.get("decreasing", [])
    contrib = ups + dns
    if contrib:
        contrib = sorted(contrib, key=lambda f: f["delta_pp"])
        fig = go.Figure(go.Bar(
            x=[f["delta_pp"] for f in contrib],
            y=[f["label"] for f in contrib],
            orientation="h",
            marker_color=[RED if f["delta_pp"] > 0 else BLUE for f in contrib],
            marker_line=dict(color=SURFACE, width=2),
        ))
        fig.update_layout(title="Contribution to 7-day change (model score, pp)", showlegend=False)
        fig.update_xaxes(title="pp of pre-calibration model score")
        st.plotly_chart(style_fig(fig, 320), use_container_width=True)
        st.caption("Red bars pushed risk up this week; blue bars pulled it down. "
                   "Feature-substitution attribution on the model's smooth score "
                   "(calibration is monotone, so directions and rankings carry to "
                   "the headline probability).")
    else:
        st.info("No material feature-driven change over the past week.")

# ------------------------------------------------- Section 3: category risk --
st.subheader("Category stress scores (diagnostic, 0–100)")
cat = out["category_summary"]
if not cat.empty:
    cat_sorted = cat.sort_values("score", ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cat_sorted["score"], y=cat_sorted["label"], orientation="h",
        marker_color=BLUE, marker_line=dict(color=SURFACE, width=2),
        name="Current", text=[f"{v:.0f}" for v in cat_sorted["score"]],
        textposition="outside", textfont=dict(color=INK2), cliponaxis=False,
    ))
    fig.add_trace(go.Scatter(
        x=cat_sorted["prev_month"], y=cat_sorted["label"], mode="markers",
        marker=dict(symbol="line-ns-open", size=14, color=MUTED, line_width=2),
        name="1 month ago",
    ))
    fig.update_xaxes(range=[0, 105], title="Stress score (expanding historical percentile, risk-oriented)")
    st.plotly_chart(style_fig(fig, 380), use_container_width=True)
    st.dataframe(
        cat[["label", "score", "prev_week", "prev_month", "hist_percentile", "trend"]]
        .rename(columns={
            "label": "Category", "score": "Now", "prev_week": "1w ago",
            "prev_month": "1m ago", "hist_percentile": "Historical pctile", "trend": "Trend",
        }),
        use_container_width=True, hide_index=True,
    )
    st.caption("Diagnostic composites of oriented indicator percentiles — context for the "
               "probability model, not a substitute for it.")

# --------------------------------------------- Section 4: indicator explorer --
st.subheader("Leading indicators")
snap = out["snapshot"]
if not snap.empty:
    cats = sorted(snap["Category"].unique())
    chosen = st.multiselect("Filter by category", cats, default=cats, key="cat_filter")
    view = snap[snap["Category"].isin(chosen)]
    st.dataframe(
        view, use_container_width=True, hide_index=True, height=420,
        column_config={
            "Latest": st.column_config.NumberColumn(format="%.2f"),
            "Change": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.caption("Signal = risk-oriented percentile of the indicator's model features "
               "(≥70 ⇒ risk ↑, ≤30 ⇒ risk ↓). Sort any column by clicking its header.")

# ------------------------------------------- Section 5: historical probability --
st.subheader("Historical out-of-sample probability")
try:
    results, oos = _backtest_artifacts()
except Exception as exc:
    results, oos = None, None
    st.info(f"Backtest artifacts unavailable: {exc}")

if oos is not None:
    hsel = st.multiselect(
        "Horizons", [f"{h}D" for h in HORIZONS], default=["90D", "30D"], key="hsel"
    )
    fig = go.Figure()
    for a, b in recession_spans(oos["in_recession"]):
        fig.add_vrect(x0=a, x1=b, fillcolor=SHADE, line_width=0)
    horder = [90, 30, 60, 45, 15]  # legend/assignment follows fixed palette slots
    colors = {90: BLUE, 30: ORANGE, 60: AQUA, 45: YELLOW, 15: "#e87ba4"}
    for h in horder:
        if f"{h}D" not in hsel:
            continue
        fig.add_trace(go.Scatter(
            x=oos.index, y=oos[f"p{h}"] * 100, mode="lines",
            line=dict(color=colors[h], width=2), name=f"{h}-day",
        ))
    fig.add_trace(go.Scatter(
        x=[pd.Timestamp(pred["data_date"])], y=[probs[90] * 100], mode="markers",
        marker=dict(color=INK, size=9, symbol="diamond"), name="Today (production model)",
    ))
    fig.add_hline(y=30, line_dash="dot", line_color=MUTED,
                  annotation_text="30% alert level", annotation_font_color=MUTED)
    fig.update_yaxes(title="Probability (%)", range=[0, 100])
    fig.update_layout(title="Walk-forward out-of-sample probabilities · gray bands = NBER recessions")
    st.plotly_chart(style_fig(fig, 420), use_container_width=True)
    st.caption(
        "Every point is predicted by a model trained only on data available before "
        "that fold, with first-release (vintage) inputs — this is what the model "
        "would genuinely have said in real time. Fold 1 (pre-1984) shows raw, "
        "uncalibrated output."
    )

# ------------------------------------------------------- Section 6: backtest --
st.subheader("Backtest & calibration diagnostics")
if results is not None:
    mnames = [m for m in ("A_yield_curve", "B_elastic_net", "C_grad_boost", "D_ensemble",
                          "bl_constant", "bl_sahm", "bl_nfci") if m in results["metrics"]]
    tbl = pd.DataFrame([
        {
            "Model": m,
            "Brier ↓": results["metrics"][m]["pooled_cal"]["brier"],
            "Log loss ↓": results["metrics"][m]["pooled_cal"]["log_loss"],
            "ROC AUC ↑": results["metrics"][m]["pooled_cal"].get("roc_auc"),
            "PR AUC ↑": results["metrics"][m]["pooled_cal"].get("pr_auc"),
            "Calib. error ↓": results["metrics"][m]["pooled_cal"]["ece"],
            "Detect rate ↑": results["metrics"][m].get("event", {}).get("detect_rate"),
            "False alarms ↓": results["metrics"][m].get("event", {}).get("n_false_alarms"),
        } for m in mnames
    ])
    prod = results["selection"]["production_model"]
    st.markdown(
        f"Production model: **`{prod}`** — {results['selection']['criterion']}. "
        f"Brier improvement vs constant baseline: "
        f"**{results['selection']['brier_vs_constant']:.4f}**; vs yield-curve probit: "
        f"**{results['selection']['brier_vs_yield_curve']:.4f}**."
    )
    st.dataframe(
        tbl.style.format({c: "{:.4f}" for c in tbl.columns if c != "Model"}, na_rep="—"),
        use_container_width=True, hide_index=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines",
                                 line=dict(color=MUTED, dash="dash"), name="Perfect calibration"))
        shown = 0
        for m in ("D_ensemble", "B_elastic_net", "C_grad_boost", "A_yield_curve"):
            if m not in results["reliability"] or shown >= 3:
                continue
            r = pd.DataFrame(results["reliability"][m])
            fig.add_trace(go.Scatter(
                x=r["pred"] * 100, y=r["obs"] * 100, mode="lines+markers",
                line=dict(color=MODEL_COLORS[m], width=2), marker=dict(size=8),
                name=m,
            ))
            shown += 1
        fig.update_xaxes(title="Predicted probability (%)")
        fig.update_yaxes(title="Observed onset frequency (%)")
        fig.update_layout(title="Reliability (out-of-sample, 90-day)")
        st.plotly_chart(style_fig(fig, 360), use_container_width=True)
    with c2:
        st.markdown("**Recession-by-recession (production model, out-of-sample)**")
        prod_events = results["events"].get(prod, [])
        ev = pd.DataFrame(prod_events)
        if not ev.empty:
            ev = ev.rename(columns={
                "onset": "Onset", "max_p_90d_before": "Max P90 (90d prior)",
                "max_p_365d_before": "Max P90 (1y prior)",
                "first_cross": "First ≥20%", "lead_days": "Lead (days)",
                "detected": "Detected",
            })
            st.dataframe(
                ev.style.format({"Max P90 (90d prior)": "{:.2f}", "Max P90 (1y prior)": "{:.2f}"}),
                use_container_width=True, hide_index=True,
            )
            leads = [e for e in prod_events if e["lead_days"] is not None]
            if leads:
                st.caption(
                    f"Average lead time at the 20% detection level: "
                    f"**{np.mean([e['lead_days'] for e in leads]):.0f} days** "
                    f"({len(leads)}/{len(prod_events)} recessions flagged in the prior year)."
                )
        fa = results["false_alarms"].get(prod, [])
        st.markdown(
            f"**False alarms** (P90 ≥ 30% for ≥28 days, no onset within 270 days): "
            f"{len(fa)}"
        )
        for a in fa:
            st.caption(f"· {a['start']} → {a['end']} (max {a['max_p']:.0%})")

    with st.expander("Per-fold results, horizons and hazard exponents"):
        pf = pd.DataFrame(results["metrics"][prod]["per_fold"])
        if not pf.empty:
            cols_show = [c for c in ("fold", "n", "n_pos", "brier", "log_loss", "roc_auc", "ece") if c in pf]
            st.dataframe(
                pf[cols_show].style.format({c: "{:.4f}" for c in ("brier", "log_loss", "roc_auc", "ece")}, na_rep="—"),
                use_container_width=True, hide_index=True,
            )
        hm = pd.DataFrame(results["horizon_metrics"]).T
        hm.index.name = "Horizon (days)"
        st.markdown("**Term-structure check** — each horizon's probability vs its own label:")
        st.dataframe(
            hm[["n_pos", "brier", "log_loss", "ece"]].style.format(
                {"brier": "{:.4f}", "log_loss": "{:.4f}", "ece": "{:.4f}"}),
            use_container_width=True,
        )
        st.markdown(
            "Hazard exponents θ (fitted on out-of-sample predictions, monotone by "
            "construction): "
            + ", ".join(f"{h}d: {t:.3f}" for h, t in results["theta"].items())
        )

# ----------------------------------------------------- Section 7: data health --
st.subheader("Data health")
health = out["health"]
if not health.empty:
    r = out["refresh"]
    ok = health["status"].eq("fresh").sum()
    refresh_note = "✅ clean" if r["n_fetch_errors"] == 0 else f"⚠️ {r['n_fetch_errors']} fetch errors"
    st.markdown(
        f"Refresh {refresh_note} · "
        f"{ok}/{len(health)} sources fresh · "
        f"{r['n_vintage_series']} series with first-release vintages · "
        f"dataset version `{r['dataset_version']}`"
    )
    hv = health.copy()
    hv["category"] = hv["category"].map(lambda c: CATEGORY_LABELS.get(c, c))
    hv["last_period"] = pd.to_datetime(hv["last_period"]).dt.strftime("%Y-%m-%d")
    hv["fetched_at"] = pd.to_datetime(hv["fetched_at"], format="ISO8601", utc=True).dt.strftime("%Y-%m-%d %H:%M")
    hv = hv[["name", "category", "tier", "source", "vintage_mode", "last_period",
             "days_stale", "fetched_at", "status", "error"]].rename(columns={
        "name": "Series", "category": "Category", "tier": "Tier", "source": "Source",
        "vintage_mode": "Vintage data", "last_period": "Latest observation",
        "days_stale": "Days since obs.", "fetched_at": "Retrieved",
        "status": "Status", "error": "Error",
    })
    def _status_style(v):
        color = {"fresh": "#0ca30c", "stale": "#ec835a", "cache_fallback": "#ec835a",
                 "error": "#d03b3b"}.get(v, INK2)
        return f"color: {color}; font-weight: 600"
    st.dataframe(
        hv.style.map(_status_style, subset=["Status"]),
        use_container_width=True, hide_index=True, height=420,
    )
    st.caption(
        "Monthly macro series are inherently 2–8 weeks old — 'Days since obs.' reflects "
        "publication lag, not an error. ⚠ cache_fallback = the API failed and the most "
        "recent cached data is in use."
    )

# -------------------------------------------------------- probability history --
with st.expander("Saved prediction history (this machine)"):
    hist = load_history()
    if hist.empty:
        st.caption("No saved predictions yet — each successful refresh appends one row.")
    else:
        show = hist[["ts_utc", "data_date", "p15", "p30", "p45", "p60", "p90",
                     "model_name", "model_version", "dataset_version"]].tail(60)
        st.dataframe(
            show.style.format({c: "{:.2%}" for c in ("p15", "p30", "p45", "p60", "p90")}),
            use_container_width=True, hide_index=True,
        )

# ------------------------------------------------------------- methodology ----
st.subheader("Methodology & model card")
with st.expander("Read the full methodology", expanded=False):
    card = MODELS_DIR / "model_card.md"
    if card.exists():
        st.markdown(card.read_text())
    st.markdown("""
---
**Key conventions and caveats**

- **NBER timing convention.** NBER dates recessions to months. We define onset as
  the first day of the first recession month. The 15/30/45/60/90-day horizons are
  cumulative windows on that convention; sub-monthly precision comes from the
  hazard term structure, not from the (monthly) historical labels.
- **Point-in-time discipline.** Training features use ALFRED *first-release* values
  and actual release dates where vintages exist; otherwise configured publication
  lags. Residual revision bias (pre-vintage history, series without vintages, or
  keyless mode) is disclosed in Data Health rather than hidden.
- **Today's prediction** uses latest-vintage data — correct for today's forecaster,
  who knows current revisions.
- **The NBER label itself lags**: recent months can be re-labeled after the fact,
  so the most recent ~2 quarters of "no recession" ground truth are provisional.
- **Uncertainty.** The band is a block-bootstrap 10–90% range on the interpretable
  model, centered on the production probability — a lower bound on true
  uncertainty. With ~8 recessions of usable history, differences like 17.4% vs
  15–22% are not statistically meaningful; read ranges, not decimals.
""")

st.caption(
    "Built for analysis, not as investment advice. Sources: FRED®/ALFRED® (Federal "
    "Reserve Bank of St. Louis) and derived public-domain series."
)

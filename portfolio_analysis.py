"""
Multi-Asset Portfolio Analysis & Risk Dashboard
------------------------------------------------
Run:  pip install yfinance pandas numpy matplotlib seaborn
      python portfolio_analysis.py          # real data from Yahoo Finance
      python portfolio_analysis.py --demo   # synthetic data (offline test only)

Outputs (./output): dashboard.png, metrics.csv, correlation.csv
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- CONFIG ----------------
START = "2016-01-01"
RF = 0.065            # annual risk-free rate (approx. India 10Y G-Sec yield)
TRADING_DAYS = 252
INR_TERMS = True      # convert USD assets (S&P 500, US bonds) into INR

TICKERS = {
    "NIFTY 50": "^NSEI",
    "S&P 500": "^GSPC",
    "Gold": "GC=F",          # COMEX gold futures (USD), converted to INR below
    "Bonds": "IEF",          # US 7-10Y Treasury ETF (swap for an Indian gilt ETF if you like)
    "USD/INR": "USDINR=X",
}
USD_ASSETS = ["S&P 500", "Bonds", "Gold"]

PORTFOLIOS = {  # weights must sum to 1
    "Conservative": {"NIFTY 50": 0.15, "S&P 500": 0.10, "Gold": 0.20, "Bonds": 0.50, "USD/INR": 0.05},
    "Balanced":     {"NIFTY 50": 0.30, "S&P 500": 0.25, "Gold": 0.15, "Bonds": 0.25, "USD/INR": 0.05},
    "Aggressive":   {"NIFTY 50": 0.45, "S&P 500": 0.40, "Gold": 0.10, "Bonds": 0.00, "USD/INR": 0.05},
    "Equal Weight": {k: 0.2 for k in TICKERS},
}
BENCHMARK = {"NIFTY 50": 1.0}
# ----------------------------------------


def load_prices(demo=False):
    if demo:  # synthetic data so the pipeline can be tested offline
        rng = np.random.default_rng(1)
        idx = pd.bdate_range(START, periods=2500)
        mu = np.array([.12, .11, .09, .04, .03]) / TRADING_DAYS
        sd = np.array([.16, .17, .14, .06, .05]) / np.sqrt(TRADING_DAYS)
        rets = rng.normal(mu, sd, (len(idx), 5))
        return pd.DataFrame(100 * np.exp(rets.cumsum(0)), index=idx, columns=list(TICKERS))
    import yfinance as yf
    raw = yf.download(list(TICKERS.values()), start=START, auto_adjust=True, progress=False)["Close"]
    px = raw.rename(columns={v: k for k, v in TICKERS.items()})[list(TICKERS)]
    px = px.ffill().dropna()
    if INR_TERMS:
        for a in USD_ASSETS:
            px[a] = px[a] * px["USD/INR"]
    return px


def max_drawdown(r):
    wealth = (1 + r).cumprod()
    return (wealth / wealth.cummax() - 1).min()


def metrics(r):
    years = len(r) / TRADING_DAYS
    ann_ret = (1 + r).prod() ** (1 / years) - 1
    vol = r.std() * np.sqrt(TRADING_DAYS)
    downside = r[r < 0].std() * np.sqrt(TRADING_DAYS)
    mdd = max_drawdown(r)
    return {
        "Ann. Return": ann_ret, "Volatility": vol,
        "Sharpe": (ann_ret - RF) / vol,
        "Sortino": (ann_ret - RF) / downside,
        "Max Drawdown": mdd,
        "Calmar": ann_ret / abs(mdd),
        "Total Return": (1 + r).prod() - 1,
    }


def port_returns(rets, w):
    """Constant-weight (daily rebalanced) portfolio return series."""
    return rets[list(w)].mul(pd.Series(w)).sum(axis=1)


def main():
    demo = "--demo" in sys.argv
    px = load_prices(demo)
    rets = px.pct_change().dropna()
    bad = rets[(rets.abs() > 0.20).any(axis=1)]
    if len(bad):
        print("WARNING: daily moves >20% found (possible bad data):\n", bad.round(3))

    # ---- metrics: individual assets + portfolios + benchmark ----
    series = {a: rets[a] for a in rets}
    series.update({f"[P] {n}": port_returns(rets, w) for n, w in PORTFOLIOS.items()})
    series["[B] NIFTY 50 Benchmark"] = port_returns(rets, BENCHMARK)
    table = pd.DataFrame({k: metrics(v) for k, v in series.items()}).T
    corr = rets.corr()

    import os
    os.makedirs("output", exist_ok=True)
    table.to_csv("output/metrics.csv")
    corr.to_csv("output/correlation.csv")
    print(table.round(3).to_string())

    # ---- random portfolios (Monte Carlo) for risk-return comparison ----
    rng = np.random.default_rng(42)
    W = rng.dirichlet(np.ones(len(rets.columns)), 3000)
    mu = rets.mean().values * TRADING_DAYS
    cov = rets.cov().values * TRADING_DAYS
    pr = W @ mu
    pv = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
    ps = (pr - RF) / pv

    # ---- dashboard ----
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle("Multi-Asset Portfolio Analysis & Risk Dashboard", fontsize=18, fontweight="bold")

    # 1 allocation
    bw = PORTFOLIOS["Balanced"]
    ax[0, 0].pie([v for v in bw.values() if v > 0], labels=[k for k, v in bw.items() if v > 0],
                 autopct="%1.0f%%", startangle=90)
    ax[0, 0].set_title("Asset Allocation (Balanced)")

    # 2 growth of 100
    for n in list(PORTFOLIOS) + ["Benchmark"]:
        r = port_returns(rets, PORTFOLIOS[n]) if n in PORTFOLIOS else port_returns(rets, BENCHMARK)
        ((1 + r).cumprod() * 100).plot(ax=ax[0, 1], label=n, ls="--" if n == "Benchmark" else "-", lw=1.6)
    ax[0, 1].set_title("Growth of 100: Portfolios vs NIFTY 50 Benchmark")
    ax[0, 1].legend()

    # 3 drawdowns
    for n in ["Balanced", "Aggressive"]:
        w = (1 + port_returns(rets, PORTFOLIOS[n])).cumprod()
        (w / w.cummax() - 1).plot(ax=ax[0, 2], label=n)
    b = (1 + port_returns(rets, BENCHMARK)).cumprod()
    (b / b.cummax() - 1).plot(ax=ax[0, 2], label="Benchmark", ls="--")
    ax[0, 2].set_title("Drawdowns")
    ax[0, 2].legend()

    # 4 correlation heatmap
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax[1, 0])
    ax[1, 0].set_title("Correlation of Daily Returns")

    # 5 risk-return
    sc = ax[1, 1].scatter(pv, pr, c=ps, cmap="viridis", s=8, alpha=.6)
    plt.colorbar(sc, ax=ax[1, 1], label="Sharpe")
    for n in table.index:
        ax[1, 1].scatter(table.loc[n, "Volatility"], table.loc[n, "Ann. Return"],
                         marker="*" if n.startswith("[") else "o", s=150, edgecolor="k")
        ax[1, 1].annotate(n.replace("[P] ", "").replace("[B] ", ""),
                          (table.loc[n, "Volatility"], table.loc[n, "Ann. Return"]), fontsize=8)
    ax[1, 1].set_xlabel("Volatility"); ax[1, 1].set_ylabel("Annualized Return")
    ax[1, 1].set_title("Risk-Return (random portfolios + named)")

    # 6 sharpe comparison
    table["Sharpe"].plot.barh(ax=ax[1, 2])
    ax[1, 2].set_title("Sharpe Ratio")

    plt.tight_layout()
    plt.savefig("output/dashboard.png", dpi=150)
    print("\nSaved output/dashboard.png, metrics.csv, correlation.csv")


if __name__ == "__main__":
    main()

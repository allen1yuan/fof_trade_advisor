"""
risk.py
=======
风险特征因子（8 个）。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from base.factor_base import (
    FactorBase, FactorContext, TRADING_DAYS_PER_YEAR,
    annualize_vol, max_drawdown, downside_vol,
)


class AnnualizedVol(FactorBase):
    """年化波动率。direction=-1，越小越好。"""
    name = "VOL_ANN"
    category = "RISK"
    direction = -1

    def _compute_one(self, fund_ret: pd.Series, ctx: FactorContext) -> float:
        return annualize_vol(fund_ret)


class MaxDrawdown(FactorBase):
    """最大回撤（负值）。direction=1，越大（越接近 0）越好。"""
    name = "MDD"
    category = "RISK"
    direction = 1

    def _compute_one(self, fund_ret: pd.Series, ctx: FactorContext) -> float:
        mdd, _ = max_drawdown(fund_ret)
        return mdd


class MaxDrawdownDays(FactorBase):
    """回撤未恢复的最长持续天数。direction=-1。"""
    name = "MDD_DAYS"
    category = "RISK"
    direction = -1

    def _compute_one(self, fund_ret: pd.Series, ctx: FactorContext) -> float:
        _, days = max_drawdown(fund_ret)
        return float(days)


class DownsideVol(FactorBase):
    """下行波动率（年化），direction=-1。"""
    name = "DOWNSIDE_VOL"
    category = "RISK"
    direction = -1

    def __init__(self, mar: float = 0.0):
        self.mar = mar

    def _compute_one(self, fund_ret: pd.Series, ctx: FactorContext) -> float:
        return downside_vol(fund_ret, mar=self.mar)


class HistoricalVaR(FactorBase):
    """
    历史 VaR（日频，正数表示亏损幅度）。
        VaR_alpha = -Quantile_alpha(r_t)
    direction=-1（VaR 越大代表越亏，越小越好）。
    """
    name = "VAR_95"
    category = "RISK"
    direction = -1

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def _compute_one(self, fund_ret: pd.Series, ctx: FactorContext) -> float:
        return float(-np.quantile(fund_ret, self.alpha))


class ConditionalVaR(FactorBase):
    """
    条件 VaR（CVaR / Expected Shortfall）。
        CVaR = -E[r_t | r_t < -VaR]
    """
    name = "CVAR_95"
    category = "RISK"
    direction = -1

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def _compute_one(self, fund_ret: pd.Series, ctx: FactorContext) -> float:
        var_thresh = np.quantile(fund_ret, self.alpha)
        tail = fund_ret[fund_ret <= var_thresh]
        if len(tail) == 0:
            return np.nan
        return float(-tail.mean())


class TrackingError(FactorBase):
    """
    跟踪误差（年化）。
        TE = sqrt(252) * std(r_p - r_b)
    """
    name = "TRACK_ERROR"
    category = "RISK"
    direction = -1

    def __init__(self, bench_id: str = "中证偏股"):
        self.bench_id = bench_id

    def _compute_one(self, fund_ret: pd.Series, ctx: FactorContext) -> float:
        if ctx.bench_df is None:
            return np.nan
        b = ctx.bench_df
        if not isinstance(b.index, pd.MultiIndex):
            b = b.set_index(['date', 'bench_id'])
        try:
            br = b.xs(self.bench_id, level='bench_id')['ret']
        except KeyError:
            return np.nan
        df = pd.concat([fund_ret, br], axis=1, join='inner').dropna()
        df.columns = ['p', 'b']
        if len(df) < 60:
            return np.nan
        return annualize_vol(df['p'] - df['b'])


class BetaToBench(FactorBase):
    """
    相对基准 Beta。
        beta = Cov(r_p, r_b) / Var(r_b)
    direction=0（不直接打分，作为风险敞口指标）。
    """
    name = "BETA_BENCH"
    category = "RISK"
    direction = 0

    def __init__(self, bench_id: str = "中证偏股"):
        self.bench_id = bench_id

    def _compute_one(self, fund_ret: pd.Series, ctx: FactorContext) -> float:
        if ctx.bench_df is None:
            return np.nan
        b = ctx.bench_df
        if not isinstance(b.index, pd.MultiIndex):
            b = b.set_index(['date', 'bench_id'])
        try:
            br = b.xs(self.bench_id, level='bench_id')['ret']
        except KeyError:
            return np.nan
        df = pd.concat([fund_ret, br], axis=1, join='inner').dropna()
        df.columns = ['p', 'b']
        if len(df) < 60 or df['b'].var() == 0:
            return np.nan
        return float(df['p'].cov(df['b']) / df['b'].var())

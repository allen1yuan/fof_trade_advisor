"""
performance.py
==============
业绩表现因子 + 风险调整收益因子（11 个）。

包含：
- 累计收益（多窗口）
- 年化收益
- 相对基准/同类超额
- 月度胜率
- 夏普 / 索提诺 / 卡玛 / 信息比率 / 特雷诺
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from factor_base import (
    FactorBase, FactorContext, TRADING_DAYS_PER_YEAR,
    annualize_return, annualize_vol, max_drawdown, downside_vol,
)


# ------------------------- 累计收益（参数化窗口） -------------------------
class CumulativeReturn(FactorBase):
    """
    N 个交易日的累计收益。
        R_T = prod(1 + r_t) - 1
    用法：CumulativeReturn(window=63)  ->  3 个月
    """
    category = "PERFORMANCE"
    direction = 1

    def __init__(self, window: int):
        self.window = int(window)
        self.name = f"RET_{self._tag()}"
        self.min_obs = max(20, int(window * 0.7))

    def _tag(self) -> str:
        m = self.window
        if m <= 21:   return f"{m}D"
        if m <= 63:   return "3M"
        if m <= 126:  return "6M"
        if m <= 252:  return "12M"
        if m <= 504:  return "24M"
        return "36M"

    def _compute_one(self, fund_ret: pd.Series, ctx: FactorContext) -> float:
        r = fund_ret.tail(self.window)
        if len(r) < self.min_obs:
            return np.nan
        return float((1 + r).prod() - 1)


class AnnualizedReturn(FactorBase):
    """
    年化几何收益（窗口默认 252 日）。
        R_ann = (1 + R_T)^(252/N) - 1
    """
    name = "ANN_RET"
    category = "PERFORMANCE"
    direction = 1

    def __init__(self, window: int = TRADING_DAYS_PER_YEAR):
        self.window = window
        self.min_obs = max(60, int(window * 0.6))

    def _compute_one(self, fund_ret: pd.Series, ctx: FactorContext) -> float:
        r = fund_ret.tail(self.window)
        return annualize_return(r)


# ------------------------- 超额收益 -------------------------
class ExcessReturnVsBench(FactorBase):
    """
    相对基准的超额收益（年化）。
        alpha = R_p_ann - R_b_ann
    """
    name = "EXC_RET_BENCH"
    category = "PERFORMANCE"
    direction = 1

    def __init__(self, bench_id: str = "中证偏股", window: int = TRADING_DAYS_PER_YEAR):
        self.bench_id = bench_id
        self.window = window

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

        # 时间对齐，取交集
        aligned = pd.concat([fund_ret, br], axis=1, join='inner').dropna()
        if len(aligned) < 60:
            return np.nan
        aligned = aligned.tail(self.window)
        return annualize_return(aligned.iloc[:, 0]) - annualize_return(aligned.iloc[:, 1])


class WinRateMonthly(FactorBase):
    """
    月度胜率：相对基准每月跑赢的比例。
        WR = (1/N) * sum( I{r_p_m > r_b_m} )
    """
    name = "WIN_RATE_M"
    category = "PERFORMANCE"
    direction = 1

    def __init__(self, bench_id: str = "中证偏股", window: int = TRADING_DAYS_PER_YEAR * 2):
        self.bench_id = bench_id
        self.window = window

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
        df = df.tail(self.window)
        # 月度收益：用日收益 resample 'ME'（月末）
        m = df.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        if len(m) < 6:
            return np.nan
        return float((m['p'] > m['b']).mean())


# ------------------------- 风险调整收益 -------------------------
class SharpeRatio(FactorBase):
    """
    夏普比率。
        Sharpe = (R_ann - r_f) / sigma_ann
    """
    name = "SHARPE"
    category = "RISK_ADJ"
    direction = 1

    def _compute_one(self, fund_ret: pd.Series, ctx: FactorContext) -> float:
        r_ann = annualize_return(fund_ret)
        s = annualize_vol(fund_ret)
        if s == 0 or np.isnan(s):
            return np.nan
        return float((r_ann - ctx.rf) / s)


class SortinoRatio(FactorBase):
    """
    索提诺比率。
        Sortino = (R_ann - MAR) / sigma_d
    """
    name = "SORTINO"
    category = "RISK_ADJ"
    direction = 1

    def __init__(self, mar: float = 0.0):
        self.mar = mar

    def _compute_one(self, fund_ret: pd.Series, ctx: FactorContext) -> float:
        r_ann = annualize_return(fund_ret)
        sd = downside_vol(fund_ret, mar=self.mar)
        if sd == 0 or np.isnan(sd):
            return np.nan
        return float((r_ann - self.mar) / sd)


class CalmarRatio(FactorBase):
    """
    卡玛比率。
        Calmar = R_ann / |MDD|
    """
    name = "CALMAR"
    category = "RISK_ADJ"
    direction = 1

    def _compute_one(self, fund_ret: pd.Series, ctx: FactorContext) -> float:
        r_ann = annualize_return(fund_ret)
        mdd, _ = max_drawdown(fund_ret)
        if mdd == 0:
            return np.nan
        return float(r_ann / abs(mdd))


class InformationRatio(FactorBase):
    """
    信息比率。
        IR = alpha_ann / TE
    其中 TE = sqrt(252) * std(r_p - r_b)
    """
    name = "INFO_RATIO"
    category = "RISK_ADJ"
    direction = 1

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
        diff = df['p'] - df['b']
        te = annualize_vol(diff)
        if te == 0:
            return np.nan
        alpha = annualize_return(df['p']) - annualize_return(df['b'])
        return float(alpha / te)


class TreynorRatio(FactorBase):
    """
    特雷诺比率。
        Treynor = (R_ann - r_f) / beta
    """
    name = "TREYNOR"
    category = "RISK_ADJ"
    direction = 1

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
        cov = df['p'].cov(df['b'])
        var_b = df['b'].var()
        if var_b == 0:
            return np.nan
        beta = cov / var_b
        if beta == 0:
            return np.nan
        return float((annualize_return(df['p']) - ctx.rf) / beta)

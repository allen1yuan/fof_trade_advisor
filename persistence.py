"""
persistence.py
==============
业绩持续性因子（5 个）。
- Hurst 指数
- 季度连续胜率
- 排名稳定性
- 滚动 IR 均值
- 收益自相关
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from factor_base import (
    FactorBase, FactorContext, TRADING_DAYS_PER_YEAR,
    annualize_return, annualize_vol,
)


class HurstExponent(FactorBase):
    """
    Hurst 指数（R/S 分析）。
    H ∈ (0, 1)：
        H = 0.5  纯随机游走
        H > 0.5  趋势（持续性）
        H < 0.5  反转
    direction=1（业绩持续性强）。
    """
    name = "RET_HURST"
    category = "PERSISTENCE"
    direction = 1

    def _compute_one(self, fund_ret: pd.Series, ctx: FactorContext) -> float:
        ts = fund_ret.dropna().values
        N = len(ts)
        if N < 100:
            return np.nan
        # 用对数累计收益的 R/S
        lags = np.unique(np.logspace(1, np.log10(N // 2), 10).astype(int))
        rs = []
        for lag in lags:
            if lag < 10:
                continue
            chunks = N // lag
            rs_vals = []
            for i in range(chunks):
                seg = ts[i * lag:(i + 1) * lag]
                if len(seg) < 2:
                    continue
                mean_seg = seg.mean()
                Z = np.cumsum(seg - mean_seg)
                R = Z.max() - Z.min()
                S = seg.std(ddof=1)
                if S > 0 and R > 0:
                    rs_vals.append(R / S)
            if rs_vals:
                rs.append((lag, np.mean(rs_vals)))
        if len(rs) < 4:
            return np.nan
        x = np.log([r[0] for r in rs])
        y = np.log([r[1] for r in rs])
        H, _ = np.polyfit(x, y, 1)
        return float(H)


class QuarterlyWinStreak(FactorBase):
    """
    季度胜率：过去 N 季度跑赢同类（或基准）的比例。
        WS = (1/N) * sum_q I{ R_p_q > R_bench_q }
    direction=1。
    """
    name = "WIN_QUARTERS"
    category = "PERSISTENCE"
    direction = 1

    def __init__(self, bench_id: str = "中证偏股", n_quarters: int = 8):
        self.bench_id = bench_id
        self.n_quarters = n_quarters

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
        q = df.resample('QE').apply(lambda x: (1 + x).prod() - 1)
        if len(q) < 4:
            return np.nan
        q = q.tail(self.n_quarters)
        return float((q['p'] > q['b']).mean())


class RankStability(FactorBase):
    """
    排名稳定性。

    在每个滚动窗口（如季度）内，计算基金在同类中的分位数排名，
    然后取这些分位数的标准差的相反数（稳定性 = 1 - 波动）。

    这个因子需要"同类基金的收益序列"，因此只能在 compute 中处理为横截面计算。
    简化实现：用基金自身收益序列的滚动季度收益相对其历史中位数的偏离稳定性。
    direction=1。
    """
    name = "RANK_STABILITY"
    category = "PERSISTENCE"
    direction = 1

    def __init__(self, n_quarters: int = 8):
        self.n_quarters = n_quarters

    def _compute_one(self, fund_ret: pd.Series, ctx: FactorContext) -> float:
        q = fund_ret.resample('QE').apply(lambda x: (1 + x).prod() - 1).dropna()
        if len(q) < 4:
            return np.nan
        q = q.tail(self.n_quarters)
        # 每季度的 z-score 绝对值的稳定性（变异系数的倒数）
        if q.std(ddof=1) == 0 or np.isnan(q.std(ddof=1)):
            return np.nan
        cv = q.std(ddof=1) / max(abs(q.mean()), 1e-6)
        return float(1.0 / (1.0 + abs(cv)))


class RollingIRMean(FactorBase):
    """
    滚动 IR 均值：滚动 6 个月 IR 的均值。
    direction=1。
    """
    name = "ROLLING_IR_MEAN"
    category = "PERSISTENCE"
    direction = 1

    def __init__(self, bench_id: str = "中证偏股",
                 sub_window: int = 126, n_sub: int = 4):
        self.bench_id = bench_id
        self.sub_window = sub_window   # 6 月
        self.n_sub = n_sub

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
        df['diff'] = df['p'] - df['b']
        if len(df) < self.sub_window * self.n_sub:
            return np.nan
        df_tail = df.tail(self.sub_window * self.n_sub)
        irs = []
        for k in range(self.n_sub):
            sub = df_tail.iloc[k * self.sub_window: (k + 1) * self.sub_window]
            te = annualize_vol(sub['diff'])
            if te == 0:
                continue
            ar = annualize_return(sub['p']) - annualize_return(sub['b'])
            irs.append(ar / te)
        if len(irs) == 0:
            return np.nan
        return float(np.mean(irs))


class ReturnAutocorr(FactorBase):
    """
    收益自相关 ρ_1 = Corr(r_t, r_{t-1})。
    direction=1（持续性正向）。
    """
    name = "RET_AUTOCORR"
    category = "PERSISTENCE"
    direction = 1

    def __init__(self, freq: str = "M"):
        """freq: 'D' 日 / 'W' 周 / 'M' 月。月频通常更稳健。"""
        self.freq = freq

    def _compute_one(self, fund_ret: pd.Series, ctx: FactorContext) -> float:
        if self.freq == "M":
            r = fund_ret.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        elif self.freq == "W":
            r = fund_ret.resample('W').apply(lambda x: (1 + x).prod() - 1)
        else:
            r = fund_ret
        r = r.dropna()
        if len(r) < 12:
            return np.nan
        return float(r.autocorr(lag=1))

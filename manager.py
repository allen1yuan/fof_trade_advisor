"""
manager.py
==========
基金经理能力因子（6 个）。
- T-M 模型：选股 (alpha_TM) + 择时 (gamma_TM)
- H-M 模型：选股 (alpha_HM) + 择时 (beta2_HM)
- Brinson 归因：选股贡献
- 基金经理任职年限
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from factor_base import FactorBase, FactorContext, TRADING_DAYS_PER_YEAR


def _get_bench_ret(ctx: FactorContext, bench_id: str) -> pd.Series:
    if ctx.bench_df is None:
        return None
    b = ctx.bench_df
    if not isinstance(b.index, pd.MultiIndex):
        b = b.set_index(['date', 'bench_id'])
    try:
        return b.xs(bench_id, level='bench_id')['ret']
    except KeyError:
        return None


def _ols_with_t(y: np.ndarray, X: np.ndarray):
    """
    OLS 回归并返回 t 统计量。
    返回 (beta, t_stats, r2)
    """
    n, k = X.shape
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.full(k, np.nan), np.full(k, np.nan), np.nan
    resid = y - X @ beta
    if n - k <= 0:
        return beta, np.full(k, np.nan), np.nan
    sse = (resid ** 2).sum()
    sigma2 = sse / (n - k)
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return beta, np.full(k, np.nan), np.nan
    se = np.sqrt(np.diag(cov))
    t = beta / se
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - sse / ss_tot if ss_tot > 0 else np.nan
    return beta, t, float(r2)


# ------------------------- Treynor-Mazuy -------------------------
class TM_Alpha(FactorBase):
    """
    T-M 选股能力（alpha）。
        r_p_t - r_f_t = alpha + beta * (r_m_t - r_f_t) + gamma * (r_m_t - r_f_t)^2 + eps
    输出年化 alpha。direction=1。
    """
    name = "ALPHA_TM"
    category = "MANAGER"
    direction = 1

    def __init__(self, bench_id: str = "中证偏股"):
        self.bench_id = bench_id

    def _compute_one(self, fund_ret: pd.Series, ctx: FactorContext) -> float:
        br = _get_bench_ret(ctx, self.bench_id)
        if br is None:
            return np.nan
        df = pd.concat([fund_ret.rename('p'), br.rename('m')], axis=1, join='inner').dropna()
        if len(df) < 60:
            return np.nan
        rf_d = ctx.rf / TRADING_DAYS_PER_YEAR
        y = (df['p'] - rf_d).values
        excess = (df['m'] - rf_d).values
        X = np.column_stack([np.ones(len(df)), excess, excess ** 2])
        beta, _, _ = _ols_with_t(y, X)
        # 日 alpha 转年化
        return float(beta[0] * TRADING_DAYS_PER_YEAR)


class TM_Gamma(FactorBase):
    """
    T-M 择时能力（gamma）。gamma > 0 表示牛市加杠杆有效。
    direction=1。
    """
    name = "GAMMA_TM"
    category = "MANAGER"
    direction = 1

    def __init__(self, bench_id: str = "中证偏股"):
        self.bench_id = bench_id

    def _compute_one(self, fund_ret: pd.Series, ctx: FactorContext) -> float:
        br = _get_bench_ret(ctx, self.bench_id)
        if br is None:
            return np.nan
        df = pd.concat([fund_ret.rename('p'), br.rename('m')], axis=1, join='inner').dropna()
        if len(df) < 60:
            return np.nan
        rf_d = ctx.rf / TRADING_DAYS_PER_YEAR
        y = (df['p'] - rf_d).values
        excess = (df['m'] - rf_d).values
        X = np.column_stack([np.ones(len(df)), excess, excess ** 2])
        beta, _, _ = _ols_with_t(y, X)
        return float(beta[2])


# ------------------------- Henriksson-Merton -------------------------
class HM_Alpha(FactorBase):
    """
    H-M 选股能力。
        r_p_t - r_f_t = alpha + beta1 * (r_m - r_f) + beta2 * D_t * (r_m - r_f) + eps
    其中 D_t = 1 if r_m > r_f else 0
    输出年化 alpha。
    """
    name = "ALPHA_HM"
    category = "MANAGER"
    direction = 1

    def __init__(self, bench_id: str = "中证偏股"):
        self.bench_id = bench_id

    def _compute_one(self, fund_ret: pd.Series, ctx: FactorContext) -> float:
        br = _get_bench_ret(ctx, self.bench_id)
        if br is None:
            return np.nan
        df = pd.concat([fund_ret.rename('p'), br.rename('m')], axis=1, join='inner').dropna()
        if len(df) < 60:
            return np.nan
        rf_d = ctx.rf / TRADING_DAYS_PER_YEAR
        y = (df['p'] - rf_d).values
        excess = (df['m'] - rf_d).values
        D = (excess > 0).astype(float)
        X = np.column_stack([np.ones(len(df)), excess, D * excess])
        beta, _, _ = _ols_with_t(y, X)
        return float(beta[0] * TRADING_DAYS_PER_YEAR)


class HM_Beta2(FactorBase):
    """
    H-M 择时能力 beta2，>0 表示牛市增 Beta（择时正向）。
    """
    name = "BETA2_HM"
    category = "MANAGER"
    direction = 1

    def __init__(self, bench_id: str = "中证偏股"):
        self.bench_id = bench_id

    def _compute_one(self, fund_ret: pd.Series, ctx: FactorContext) -> float:
        br = _get_bench_ret(ctx, self.bench_id)
        if br is None:
            return np.nan
        df = pd.concat([fund_ret.rename('p'), br.rename('m')], axis=1, join='inner').dropna()
        if len(df) < 60:
            return np.nan
        rf_d = ctx.rf / TRADING_DAYS_PER_YEAR
        y = (df['p'] - rf_d).values
        excess = (df['m'] - rf_d).values
        D = (excess > 0).astype(float)
        X = np.column_stack([np.ones(len(df)), excess, D * excess])
        beta, _, _ = _ols_with_t(y, X)
        return float(beta[2])


# ------------------------- Brinson 归因 -------------------------
class BrinsonSelection(FactorBase):
    """
    Brinson 选股贡献（年化）。
        Selection_j = w_b_j * (r_p_j - r_b_j)
        Total_Selection = sum_j Selection_j
    需要持仓与个股收益、基准行业权重。

    实现简化：以基金披露持仓作为 w_p，以基准（如沪深300）的行业权重作为 w_b，
    每个行业用持仓股票收益加权得到 r_p_j、用基准行业指数收益作为 r_b_j。

    本实现假设 ctx.holding_df 提供了 industry & 个股收益可通过 stock_ret_df 取到。
    粒度：按季度计算，再年化。
    """
    name = "BRINSON_SEL"
    category = "MANAGER"
    direction = 1
    min_obs = 1

    def __init__(self, bench_industry_weight: dict[str, float],
                 bench_industry_ret: dict[str, float]):
        """
        bench_industry_weight : 基准行业权重 {ind: w}（最近一期）
        bench_industry_ret    : 基准行业上一持仓期至 eval_date 的收益 {ind: r}
        """
        self.bench_w = bench_industry_weight
        self.bench_r = bench_industry_ret

    def _compute_one(self, fund_ret, ctx):
        return np.nan

    def compute(self, ctx: FactorContext, fund_ids=None):
        if ctx.holding_df is None or ctx.stock_ret_df is None:
            return pd.Series(dtype=float, name=self.name)

        h = ctx.holding_df
        if not isinstance(h.index, pd.MultiIndex):
            h = h.set_index(['date', 'fund_id'])
        if fund_ids is None:
            fund_ids = h.index.get_level_values('fund_id').unique()

        s = ctx.stock_ret_df
        if not isinstance(s.index, pd.MultiIndex):
            s = s.set_index(['date', 'stock_id'])

        out = {}
        for fid in fund_ids:
            try:
                df_fund = h.xs(fid, level='fund_id')
            except KeyError:
                out[fid] = np.nan
                continue
            dates = df_fund.index.get_level_values('date').unique().sort_values()
            valid = dates[dates <= ctx.eval_date]
            if len(valid) < 1:
                out[fid] = np.nan
                continue
            # 上一披露日 -> eval_date 期间，每只持仓股的累计收益
            d_prev = valid[-1]
            snap = df_fund.xs(d_prev, level='date')
            stk_ids = snap['stock_id'].unique() if 'stock_id' in snap.columns else snap.index.tolist()
            try:
                stk_panel = s.loc[(slice(d_prev, ctx.eval_date), list(stk_ids)), 'ret']
            except KeyError:
                out[fid] = np.nan
                continue
            cum_stk = stk_panel.unstack('stock_id').add(1).cumprod().iloc[-1] - 1
            df_use = snap.copy()
            if 'stock_id' in df_use.columns:
                df_use = df_use.set_index('stock_id')
            df_use['stk_ret'] = cum_stk.reindex(df_use.index)
            df_use = df_use.dropna(subset=['stk_ret', 'industry', 'weight'])

            sel_total = 0.0
            for ind in df_use['industry'].unique():
                sub = df_use[df_use['industry'] == ind]
                if sub['weight'].sum() == 0:
                    continue
                # 行业内股票收益按权重加权
                rp_j = (sub['stk_ret'] * sub['weight']).sum() / sub['weight'].sum()
                rb_j = self.bench_r.get(ind, 0.0)
                wb_j = self.bench_w.get(ind, 0.0)
                sel_total += wb_j * (rp_j - rb_j)
            out[fid] = float(sel_total)
        return pd.Series(out, name=self.name)


# ------------------------- 经理任职年限 -------------------------
class ManagerTenure(FactorBase):
    """
    现任基金经理任职年限（年）。
    direction=1（经验通常正面，但与基金类型相关）。
    """
    name = "MANAGER_TENURE"
    category = "MANAGER"
    direction = 1
    min_obs = 1

    def _compute_one(self, fund_ret, ctx):
        return np.nan

    def compute(self, ctx: FactorContext, fund_ids=None):
        if ctx.fund_info_df is None:
            return pd.Series(dtype=float, name=self.name)
        info = ctx.fund_info_df
        if 'fund_id' in info.columns:
            info = info.set_index('fund_id')
        if 'manager_start_date' not in info.columns:
            return pd.Series(dtype=float, name=self.name)

        if fund_ids is None:
            fund_ids = info.index.tolist()
        out = {}
        for fid in fund_ids:
            try:
                d = pd.Timestamp(info.loc[fid, 'manager_start_date'])
                out[fid] = float((ctx.eval_date - d).days / 365.25)
            except (KeyError, ValueError, TypeError):
                out[fid] = np.nan
        return pd.Series(out, name=self.name)

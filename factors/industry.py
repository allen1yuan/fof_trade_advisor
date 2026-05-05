"""
industry.py
===========
行业配置因子（5 个）。基于持仓表 holding_df 计算。

注意：A 股公募季报披露前十大重仓股（占比约 30%-50%），半年报/年报披露全部持仓。
本模块假设 holding_df 已包含可用的 weight 与 industry 字段，
若只有重仓数据，权重和不会等于 1，需在使用方提前 normalize。
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

from base.factor_base import FactorBase, FactorContext


def _get_holding_at(ctx: FactorContext, fund_id: str,
                    asof: pd.Timestamp) -> Optional[pd.DataFrame]:
    """取 fund_id 在 ``asof`` 日期之前最近一次披露的持仓。返回不含 MultiIndex 的扁平 DataFrame。"""
    if ctx.holding_df is None:
        return None
    h = ctx.holding_df
    if not isinstance(h.index, pd.MultiIndex):
        h = h.set_index(['date', 'fund_id'])
    fund_level = h.index.get_level_values('fund_id')
    date_level = h.index.get_level_values('date')
    mask = (fund_level == fund_id) & (date_level <= asof)
    df_fund = h.loc[mask]
    if df_fund.empty:
        return None
    last_date = df_fund.index.get_level_values('date').max()
    mask2 = df_fund.index.get_level_values('date') == last_date
    return df_fund.loc[mask2].reset_index()


class HoldingFactorBase(FactorBase):
    """所有持仓型因子的共同基类，重写 compute 以使用持仓快照而非净值。"""
    min_obs = 1   # 持仓型不依赖净值长度

    def _compute_one(self, fund_ret: pd.Series, ctx: FactorContext) -> float:
        # 委托到 _compute_from_holding
        if ctx.eval_date is None:
            return np.nan
        h = _get_holding_at(ctx, getattr(self, '_current_fid', None), ctx.eval_date)
        if h is None or h.empty:
            return np.nan
        return self._compute_from_holding(h, ctx)

    def _compute_from_holding(self, h: pd.DataFrame, ctx: FactorContext) -> float:
        raise NotImplementedError

    def compute(self, ctx: FactorContext, fund_ids=None):
        """重写：按 fund_id 取持仓快照，不依赖净值序列。"""
        if ctx.eval_date is None:
            raise ValueError("FactorContext.eval_date 必填")
        if fund_ids is None:
            if ctx.holding_df is None:
                return pd.Series(dtype=float, name=self.name)
            fund_ids = ctx.holding_df['fund_id'].unique() if 'fund_id' in ctx.holding_df.columns \
                else ctx.holding_df.index.get_level_values('fund_id').unique()

        out = {}
        for fid in fund_ids:
            self._current_fid = fid
            h = _get_holding_at(ctx, fid, ctx.eval_date)
            if h is None or h.empty:
                out[fid] = np.nan
                continue
            try:
                out[fid] = self._compute_from_holding(h, ctx)
            except Exception:
                out[fid] = np.nan
        return pd.Series(out, name=self.name)


class IndustryHHI(HoldingFactorBase):
    """
    行业 Herfindahl 集中度。
        HHI = sum_j w_j^2
    direction=-1，越分散越好。
    """
    name = "IND_HHI"
    category = "INDUSTRY"
    direction = -1

    def _compute_from_holding(self, h: pd.DataFrame, ctx: FactorContext) -> float:
        if 'industry' not in h.columns:
            return np.nan
        ind_w = h.groupby('industry')['weight'].sum()
        # 归一化到 1
        s = ind_w.sum()
        if s <= 0:
            return np.nan
        ind_w = ind_w / s
        return float((ind_w ** 2).sum())


class EffectiveIndustryNumber(HoldingFactorBase):
    """
    有效行业数 = 1 / HHI。
    direction=1，分散度越高越好。
    """
    name = "IND_NUM"
    category = "INDUSTRY"
    direction = 1

    def _compute_from_holding(self, h: pd.DataFrame, ctx: FactorContext) -> float:
        if 'industry' not in h.columns:
            return np.nan
        ind_w = h.groupby('industry')['weight'].sum()
        s = ind_w.sum()
        if s <= 0:
            return np.nan
        ind_w = ind_w / s
        hhi = (ind_w ** 2).sum()
        return float(1 / hhi) if hhi > 0 else np.nan


class IndustryDeviation(HoldingFactorBase):
    """
    相对基准的行业偏离度。
        D = sum_j |w_p_j - w_b_j|
    需要传入基准的行业权重字典 ``bench_industry_weight``。
    direction=0（中性指标）。
    """
    name = "IND_DEVIATION"
    category = "INDUSTRY"
    direction = 0

    def __init__(self, bench_industry_weight: Optional[dict[str, float]] = None):
        # 例如：{'电子': 0.12, '食品饮料': 0.10, ...}
        self.bench_w = bench_industry_weight or {}

    def _compute_from_holding(self, h: pd.DataFrame, ctx: FactorContext) -> float:
        if not self.bench_w or 'industry' not in h.columns:
            return np.nan
        ind_w = h.groupby('industry')['weight'].sum()
        s = ind_w.sum()
        if s <= 0:
            return np.nan
        ind_w = ind_w / s
        all_inds = set(ind_w.index) | set(self.bench_w.keys())
        diff = sum(abs(ind_w.get(j, 0.0) - self.bench_w.get(j, 0.0)) for j in all_inds)
        return float(diff)


class Top3IndustryWeight(HoldingFactorBase):
    """前三大行业权重之和。direction=-1（高度集中视为风险）。"""
    name = "TOP3_IND"
    category = "INDUSTRY"
    direction = -1

    def _compute_from_holding(self, h: pd.DataFrame, ctx: FactorContext) -> float:
        if 'industry' not in h.columns:
            return np.nan
        ind_w = h.groupby('industry')['weight'].sum().sort_values(ascending=False)
        s = ind_w.sum()
        if s <= 0:
            return np.nan
        ind_w = ind_w / s
        return float(ind_w.head(3).sum())


class IndustryRotation(HoldingFactorBase):
    """
    行业轮动速度（季度环比）。
        rotation = 0.5 * sum_j |w_j_t - w_j_{t-1}|
    direction=0（不必然好坏，作为风格刻画）。
    """
    name = "IND_ROTATION"
    category = "INDUSTRY"
    direction = 0

    def _compute_from_holding(self, h_curr: pd.DataFrame, ctx: FactorContext) -> float:
        if 'industry' not in h_curr.columns:
            return np.nan
        if ctx.holding_df is None:
            return np.nan
        h = ctx.holding_df
        if not isinstance(h.index, pd.MultiIndex):
            h = h.set_index(['date', 'fund_id'])
        fid = self._current_fid
        fund_mask = h.index.get_level_values('fund_id') == fid
        df_fund = h.loc[fund_mask]
        if df_fund.empty:
            return np.nan
        dates = df_fund.index.get_level_values('date').unique().sort_values()
        valid_dates = dates[dates <= ctx.eval_date]
        if len(valid_dates) < 2:
            return np.nan
        d_curr, d_prev = valid_dates[-1], valid_dates[-2]
        h_curr_df = df_fund.loc[df_fund.index.get_level_values('date') == d_curr].reset_index()
        h_prev_df = df_fund.loc[df_fund.index.get_level_values('date') == d_prev].reset_index()

        def _ind_w(df):
            iw = df.groupby('industry')['weight'].sum()
            s = iw.sum()
            return iw / s if s > 0 else iw

        c, p = _ind_w(h_curr_df), _ind_w(h_prev_df)
        all_inds = set(c.index) | set(p.index)
        diff = sum(abs(c.get(j, 0.0) - p.get(j, 0.0)) for j in all_inds)
        return float(0.5 * diff)

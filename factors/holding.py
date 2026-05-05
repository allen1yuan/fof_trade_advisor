"""
holding.py
==========
持仓特征因子（6 个）。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from base.factor_base import FactorBase, FactorContext
from factors.industry import HoldingFactorBase, _get_holding_at


class Top10Concentration(HoldingFactorBase):
    """
    前十大重仓股权重之和。
        TOP10 = sum_{i=1}^{10} w_i
    direction=0（既是集中度也是信心，不直接打分）。
    """
    name = "TOP10_CONC"
    category = "HOLDING"
    direction = 0

    def _compute_from_holding(self, h: pd.DataFrame, ctx: FactorContext) -> float:
        if 'stock_id' in h.columns:
            ws = h.groupby('stock_id')['weight'].sum().sort_values(ascending=False).head(10)
        else:
            ws = h['weight'].sort_values(ascending=False).head(10)
        return float(ws.sum())


class StockHHI(HoldingFactorBase):
    """
    个股 HHI。
        HHI = sum_i w_i^2
    direction=-1。
    """
    name = "STOCK_HHI"
    category = "HOLDING"
    direction = -1

    def _compute_from_holding(self, h: pd.DataFrame, ctx: FactorContext) -> float:
        w = h['weight'].values
        s = w.sum()
        if s <= 0:
            return np.nan
        w = w / s
        return float(np.sum(w ** 2))


class StockNumber(HoldingFactorBase):
    """
    持股数量（已披露口径）。
    direction=1（多元化越好；但需注意只看重仓时口径不准）。
    """
    name = "STOCK_NUM"
    category = "HOLDING"
    direction = 1

    def _compute_from_holding(self, h: pd.DataFrame, ctx: FactorContext) -> float:
        return float(h['stock_id'].nunique() if 'stock_id' in h.columns else len(h))


class TurnoverRate(FactorBase):
    """
    换手率（半年频）。常用近似公式：
        TO = min(Buy, Sell) / mean(AUM)
    若没有交易明细，可用：相邻披露期持仓权重变化的绝对值之和 * 0.5 近似。
        TO ≈ 0.5 * sum_i |w_i_t - w_i_{t-1}|
    （注意此处的 TO 是基于披露持仓的近似换手率，不是真实成交换手率）
    direction=0。
    """
    name = "TURNOVER"
    category = "HOLDING"
    direction = 0
    min_obs = 1

    def _compute_one(self, fund_ret: pd.Series, ctx: FactorContext) -> float:
        return np.nan

    def compute(self, ctx: FactorContext, fund_ids=None):
        if ctx.holding_df is None:
            return pd.Series(dtype=float, name=self.name)
        h = ctx.holding_df
        if not isinstance(h.index, pd.MultiIndex):
            h = h.set_index(['date', 'fund_id'])
        if fund_ids is None:
            fund_ids = h.index.get_level_values('fund_id').unique()
        out = {}
        for fid in fund_ids:
            fund_mask = h.index.get_level_values('fund_id') == fid
            df_fund = h.loc[fund_mask]
            if df_fund.empty:
                out[fid] = np.nan
                continue
            dates = df_fund.index.get_level_values('date').unique().sort_values()
            valid = dates[dates <= ctx.eval_date]
            if len(valid) < 2:
                out[fid] = np.nan
                continue
            d_curr, d_prev = valid[-1], valid[-2]
            curr = df_fund.loc[df_fund.index.get_level_values('date') == d_curr] \
                .reset_index().groupby('stock_id')['weight'].sum()
            prev = df_fund.loc[df_fund.index.get_level_values('date') == d_prev] \
                .reset_index().groupby('stock_id')['weight'].sum()
            curr = curr / curr.sum() if curr.sum() > 0 else curr
            prev = prev / prev.sum() if prev.sum() > 0 else prev
            all_stk = curr.index.union(prev.index)
            c = curr.reindex(all_stk).fillna(0)
            p = prev.reindex(all_stk).fillna(0)
            out[fid] = float(0.5 * (c - p).abs().sum())
        return pd.Series(out, name=self.name)


class PeerOverlap(FactorBase):
    """
    同类持仓重合度：与同类基金平均持仓的余弦相似度。
        Overlap_i = (w_i · w_peer) / (||w_i|| * ||w_peer||)
    direction=-1（与同类越相似 alpha 越难，越不相似越有差异化收益潜力）。

    实现：在 eval_date 上，按 fund_info_df 的 type 分组，
    对每只基金计算它与同类其他基金平均持仓的余弦相似度。
    """
    name = "OVERLAP_PEER"
    category = "HOLDING"
    direction = -1
    min_obs = 1

    def _compute_one(self, fund_ret, ctx):
        return np.nan

    def compute(self, ctx: FactorContext, fund_ids=None):
        if ctx.holding_df is None or ctx.fund_info_df is None:
            return pd.Series(dtype=float, name=self.name)
        info = ctx.fund_info_df.set_index('fund_id') if 'fund_id' in ctx.fund_info_df.columns \
            else ctx.fund_info_df

        h = ctx.holding_df
        if not isinstance(h.index, pd.MultiIndex):
            h = h.set_index(['date', 'fund_id'])

        # 取每只基金 eval_date 之前最近一次披露
        if fund_ids is None:
            fund_ids = h.index.get_level_values('fund_id').unique()

        latest_holdings: dict[str, pd.Series] = {}
        for fid in fund_ids:
            snap = _get_holding_at(ctx, fid, ctx.eval_date)
            if snap is None or snap.empty:
                continue
            w = snap.set_index('stock_id')['weight']
            s = w.sum()
            if s > 0:
                latest_holdings[fid] = w / s

        out = {}
        for fid, w_self in latest_holdings.items():
            try:
                fund_type = info.loc[fid, 'type']
            except KeyError:
                out[fid] = np.nan
                continue
            peers = [f for f, w in latest_holdings.items()
                     if f != fid and info.loc[f, 'type'] == fund_type] \
                if 'type' in info.columns else []
            if not peers:
                out[fid] = np.nan
                continue
            # 同类平均
            df_peer = pd.concat([latest_holdings[f] for f in peers], axis=1).fillna(0).mean(axis=1)
            stocks = w_self.index.union(df_peer.index)
            v1 = w_self.reindex(stocks).fillna(0).values
            v2 = df_peer.reindex(stocks).fillna(0).values
            denom = np.linalg.norm(v1) * np.linalg.norm(v2)
            out[fid] = float(np.dot(v1, v2) / denom) if denom > 0 else np.nan
        return pd.Series(out, name=self.name)


class CrowdingExposure(HoldingFactorBase):
    """
    抱团股暴露：重仓持有的"机构抱团股"权重之和。
    需要外部传入 crowding_stock_set（一个 set/list，包含被定义为抱团股的 stock_id）。
    direction=-1（抱团股拥挤交易，未来超额预期较差）。

    抱团股识别建议：将所有主动权益基金按持仓权重加总，
    取前 100 名（或 top 5%）作为抱团股池，每季度更新。
    """
    name = "CROWDING"
    category = "HOLDING"
    direction = -1

    def __init__(self, crowding_stock_set):
        self.crowding = set(crowding_stock_set)

    def _compute_from_holding(self, h: pd.DataFrame, ctx: FactorContext) -> float:
        if 'stock_id' not in h.columns:
            return np.nan
        s_total = h['weight'].sum()
        if s_total <= 0:
            return np.nan
        sub = h[h['stock_id'].isin(self.crowding)]
        return float(sub['weight'].sum() / s_total)

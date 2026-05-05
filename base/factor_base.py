"""
factor_base.py
==============
所有因子的基类。约定：
- 输入：长表 DataFrame（MultiIndex 或 columns 化的 date/fund_id）
- 输出：以 (date, fund_id) 为索引的 Series，列名为因子代码
- 支持横截面与时序两种调用
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Iterable
import numpy as np
import pandas as pd


# ----------------------------- 常量 -----------------------------
TRADING_DAYS_PER_YEAR = 252
DEFAULT_RF = 0.02   # 默认无风险利率（年化）
DEFAULT_MAR = 0.0   # Sortino 的最小可接受收益（年化）


# ------------------------- 数据契约结构 -------------------------
@dataclass
class FactorContext:
    """
    传给每个因子的"数据上下文"，避免每个因子重复传 5、6 个 DataFrame。

    Attributes
    ----------
    nav_df : pd.DataFrame  ['date','fund_id','adj_nav','ret']
    holding_df : pd.DataFrame  ['date','fund_id','stock_id','weight','industry']
    stock_ret_df : pd.DataFrame  ['date','stock_id','ret']
    bench_df : pd.DataFrame  ['date','bench_id','ret']  # 如沪深300、中证800、偏股基金指数
    factor_ret_df : pd.DataFrame  ['date','factor_name','ret'] # Barra 风格因子收益
    fund_info_df : pd.DataFrame  ['fund_id','type','setup_date','manager_id','company']
    rf : float
        无风险利率（年化）
    eval_date : pd.Timestamp
        因子计算的截面日期（横截面调用时使用）
    lookback : int
        计算窗口（交易日），如 252 表示一年
    """
    nav_df: pd.DataFrame
    holding_df: Optional[pd.DataFrame] = None
    stock_ret_df: Optional[pd.DataFrame] = None
    bench_df: Optional[pd.DataFrame] = None
    factor_ret_df: Optional[pd.DataFrame] = None
    fund_info_df: Optional[pd.DataFrame] = None
    rf: float = DEFAULT_RF
    eval_date: Optional[pd.Timestamp] = None
    lookback: int = TRADING_DAYS_PER_YEAR


# ------------------------- 因子基类 -------------------------
class FactorBase(ABC):
    """
    因子基类。

    子类必须实现 ``_compute_one(fund_ret: pd.Series, ctx: FactorContext) -> float``。
    基类负责遍历基金、对齐时间窗口、收集结果。
    """
    name: str = "BASE"
    category: str = "BASE"
    direction: int = 1   # 1: 因子越大越好 / -1: 越小越好
    min_obs: int = 60    # 最少观测天数，否则返回 NaN

    @abstractmethod
    def _compute_one(self, fund_ret: pd.Series, ctx: FactorContext) -> float:
        """计算单只基金、单个截面日的因子值。"""
        ...

    def compute(self, ctx: FactorContext, fund_ids: Optional[Iterable[str]] = None) -> pd.Series:
        """
        横截面计算：在 ``ctx.eval_date`` 上为所有基金计算因子值。

        Returns
        -------
        pd.Series, name=self.name, index=fund_id
        """
        if ctx.eval_date is None:
            raise ValueError("FactorContext.eval_date 必填（横截面调用）")

        nav = ctx.nav_df
        if fund_ids is None:
            fund_ids = nav['fund_id'].unique() if 'fund_id' in nav.columns else nav.index.get_level_values('fund_id').unique()

        end = ctx.eval_date
        start = end - pd.Timedelta(days=int(ctx.lookback * 1.5))   # 自然日，留出冗余覆盖交易日

        # 标准化为 (date, fund_id) 索引
        nav_use = self._slice_nav(nav, start, end)

        results = {}
        for fid in fund_ids:
            try:
                ser = nav_use.xs(fid, level='fund_id')['ret']
            except KeyError:
                results[fid] = np.nan
                continue
            ser = ser.dropna().tail(ctx.lookback)
            if len(ser) < self.min_obs:
                results[fid] = np.nan
                continue
            try:
                results[fid] = self._compute_one(ser, ctx)
            except Exception:
                results[fid] = np.nan
        return pd.Series(results, name=self.name)

    # ---- 工具方法 ----
    @staticmethod
    def _slice_nav(nav: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """切出时间窗口并保证 (date, fund_id) 索引。"""
        df = nav.copy()
        if not isinstance(df.index, pd.MultiIndex):
            df = df.set_index(['date', 'fund_id'])
        idx_date = df.index.get_level_values('date')
        mask = (idx_date >= start) & (idx_date <= end)
        return df.loc[mask].sort_index()


# ------------------------- 计算辅助函数 -------------------------
def annualize_return(ret: pd.Series, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    几何年化收益。
    R_ann = prod(1 + r_t) ^ (periods / N) - 1
    """
    if len(ret) == 0:
        return np.nan
    cum = (1 + ret).prod()
    return cum ** (periods / len(ret)) - 1


def annualize_vol(ret: pd.Series, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """年化波动率：sqrt(periods) * std(r_t)."""
    return float(ret.std(ddof=1) * np.sqrt(periods))


def max_drawdown(ret: pd.Series) -> tuple[float, int]:
    """
    最大回撤与回撤天数。
    MDD_t = min_t (NAV_t - max_{s<=t} NAV_s) / max_{s<=t} NAV_s
    返回 (MDD, recovery_days)
    """
    cum = (1 + ret).cumprod()
    running_max = cum.cummax()
    dd = cum / running_max - 1
    mdd = float(dd.min())
    # 计算回撤持续天数：从最大回撤前峰值到收复峰值的最长间隔
    if mdd == 0:
        return 0.0, 0
    end_idx = int(dd.values.argmin())
    # 用峰值索引到 MDD 谷底的天数作为"未恢复"长度的近似
    try:
        peak_pos = int(cum.iloc[:end_idx + 1].values.argmax())
        recovery = end_idx - peak_pos
    except Exception:
        recovery = 0
    return mdd, int(recovery)


def downside_vol(ret: pd.Series, mar: float = 0.0,
                 periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    下行波动率。MAR 为日频可接受收益（默认 0）。
    sigma_d = sqrt(periods/N * sum(min(r_t - MAR, 0)^2))
    """
    diff = ret - mar / periods
    neg = np.minimum(diff, 0)
    return float(np.sqrt(periods * np.mean(neg ** 2)))

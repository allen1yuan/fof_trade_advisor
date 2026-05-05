"""
style.py
========
风格特征因子（7 个）— Barra 风格暴露。

核心方法：用基金日收益对 K 个风格因子日收益做时序回归，
            r_p_t - r_f_t = alpha + sum_i (beta_i * F_i_t) + eps_t
取回归系数 beta_i 作为该基金对风格 i 的暴露。

注意：风格因子收益（factor_ret_df）需要由用户上游提供，常见做法：
- Barra CNE5/CNE6 因子：购买商业风险模型
- 自构造：按市值/估值/成长/动量/质量等指标分组，做 long-short 组合日收益
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from base.factor_base import FactorBase, FactorContext, TRADING_DAYS_PER_YEAR


# 默认 7 个 Barra 风格因子代码
DEFAULT_STYLES = ['SIZE', 'VALUE', 'GROWTH', 'MOMENTUM',
                  'QUALITY', 'VOLATILITY', 'LIQUIDITY']


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    """普通最小二乘，返回 (beta, R2)。X 第一列已含常数项。"""
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.full(X.shape[1], np.nan), np.nan
    y_hat = X @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta, float(r2)


def _prepare_style_panel(ctx: FactorContext) -> pd.DataFrame:
    """把 factor_ret_df 转成宽表 [date x style]。"""
    if ctx.factor_ret_df is None:
        return None
    f = ctx.factor_ret_df
    if isinstance(f.index, pd.MultiIndex):
        wide = f['ret'].unstack('factor_name')
    else:
        wide = f.pivot_table(index='date', columns='factor_name', values='ret')
    return wide.sort_index()


class StyleExposure(FactorBase):
    """
    某一个风格因子的暴露度。
        r_p_t - r_f_t/252 = alpha + sum_k beta_k F_k_t + eps_t
    取出 beta_{style}。

    用法：StyleExposure('SIZE'), StyleExposure('VALUE') 等
    """
    category = "STYLE"
    direction = 0

    def __init__(self, style: str, all_styles=None, window: int = TRADING_DAYS_PER_YEAR):
        self.style = style
        self.all_styles = list(all_styles) if all_styles else DEFAULT_STYLES
        self.window = window
        self.name = f"BETA_{style}"
        if style not in self.all_styles:
            raise ValueError(f"{style} not in styles {self.all_styles}")

    def _compute_one(self, fund_ret: pd.Series, ctx: FactorContext) -> float:
        wide = _prepare_style_panel(ctx)
        if wide is None:
            return np.nan
        df = pd.concat([fund_ret.rename('p'), wide], axis=1, join='inner').dropna()
        if len(df) < 60:
            return np.nan
        df = df.tail(self.window)
        y = (df['p'] - ctx.rf / TRADING_DAYS_PER_YEAR).values
        cols = [c for c in self.all_styles if c in df.columns]
        if self.style not in cols:
            return np.nan
        X = np.column_stack([np.ones(len(df)), df[cols].values])
        beta, _ = _ols(y, X)
        idx = cols.index(self.style) + 1   # +1 跳过常数项
        return float(beta[idx])


class StyleR2(FactorBase):
    """
    风格回归 R²：解释度越高，说明基金风格越"系统化"。
    """
    name = "STYLE_R2"
    category = "STYLE"
    direction = 0

    def __init__(self, all_styles=None, window: int = TRADING_DAYS_PER_YEAR):
        self.all_styles = list(all_styles) if all_styles else DEFAULT_STYLES
        self.window = window

    def _compute_one(self, fund_ret: pd.Series, ctx: FactorContext) -> float:
        wide = _prepare_style_panel(ctx)
        if wide is None:
            return np.nan
        df = pd.concat([fund_ret.rename('p'), wide], axis=1, join='inner').dropna()
        if len(df) < 60:
            return np.nan
        df = df.tail(self.window)
        y = (df['p'] - ctx.rf / TRADING_DAYS_PER_YEAR).values
        cols = [c for c in self.all_styles if c in df.columns]
        X = np.column_stack([np.ones(len(df)), df[cols].values])
        _, r2 = _ols(y, X)
        return r2


class StyleDrift(FactorBase):
    """
    风格漂移度：滚动窗口的风格暴露的标准差均值。
        drift = mean_i [ std_t( beta_{i,t} ) ]
    direction=-1，漂移小说明风格稳定。
    """
    name = "STYLE_DRIFT"
    category = "STYLE"
    direction = -1

    def __init__(self, all_styles=None, sub_window: int = 63, n_sub: int = 4):
        """
        sub_window : 每个滚动子窗口的天数（默认季度 63）
        n_sub      : 计算 n_sub 个子窗口的暴露，再求标准差
        """
        self.all_styles = list(all_styles) if all_styles else DEFAULT_STYLES
        self.sub_window = sub_window
        self.n_sub = n_sub

    def _compute_one(self, fund_ret: pd.Series, ctx: FactorContext) -> float:
        wide = _prepare_style_panel(ctx)
        if wide is None:
            return np.nan
        df = pd.concat([fund_ret.rename('p'), wide], axis=1, join='inner').dropna()
        if len(df) < self.sub_window * self.n_sub:
            return np.nan
        cols = [c for c in self.all_styles if c in df.columns]

        # 滑动窗口估暴露
        beta_panels = []
        df_tail = df.tail(self.sub_window * self.n_sub)
        for k in range(self.n_sub):
            sub = df_tail.iloc[k * self.sub_window: (k + 1) * self.sub_window]
            y = (sub['p'] - ctx.rf / TRADING_DAYS_PER_YEAR).values
            X = np.column_stack([np.ones(len(sub)), sub[cols].values])
            beta, _ = _ols(y, X)
            beta_panels.append(beta[1:])   # 去掉常数项

        beta_mat = np.vstack(beta_panels)        # shape (n_sub, K)
        return float(np.nanmean(np.nanstd(beta_mat, axis=0, ddof=1)))

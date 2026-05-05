"""
processor.py
============
因子预处理与合成。

流程：
    raw_factor -> 缺失处理 -> MAD 缩尾 -> 标准化 -> 中性化 -> （单因子有效性检验） -> 合成
"""
from __future__ import annotations
from typing import Optional, Iterable
import numpy as np
import pandas as pd


# ----------------- 1. 异常值处理 -----------------
def winsorize_mad(s: pd.Series, n: float = 3.0) -> pd.Series:
    """
    MAD 法缩尾。
        MAD = 1.4826 * median(|x - median(x)|)
        x_clipped = clip(x, median - n*MAD, median + n*MAD)
    """
    x = s.dropna().astype(float)
    if x.empty:
        return s
    med = x.median()
    mad = 1.4826 * np.median(np.abs(x - med))
    if mad == 0:
        return s
    upper = med + n * mad
    lower = med - n * mad
    return s.clip(lower=lower, upper=upper)


# ----------------- 2. 标准化 -----------------
def zscore(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return s * np.nan
    return (s - mu) / sd


def rank_normalize(s: pd.Series) -> pd.Series:
    """
    分位数标准化：先求秩 -> 转 0-1 分位 -> 反正态化（可选）。
    这里直接返回 0-1 之间的均匀分位，简单且稳健。
    """
    return s.rank(pct=True, method='average')


# ----------------- 3. 中性化 -----------------
def neutralize(s: pd.Series, controls: pd.DataFrame) -> pd.Series:
    """
    对控制变量做 OLS 回归，取残差。
        s_neut = s - X * (X'X)^-1 X' s
    controls : DataFrame, index 必须与 s 对齐。可包含 dummies（如基金类型）。
    """
    df = pd.concat([s.rename('y'), controls], axis=1).dropna()
    if df.empty:
        return s * np.nan
    y = df['y'].values
    X = df.drop(columns='y').values
    X = np.column_stack([np.ones(len(X)), X])
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return s * np.nan
    resid = y - X @ beta
    return pd.Series(resid, index=df.index, name=s.name).reindex(s.index)


# ----------------- 4. 完整预处理流程 -----------------
def preprocess_factor(s: pd.Series,
                      controls: Optional[pd.DataFrame] = None,
                      mad_n: float = 3.0,
                      method: str = 'zscore',
                      direction: int = 1) -> pd.Series:
    """
    一键预处理：缩尾 -> 标准化 -> 中性化 -> 方向调整。
    direction 用于把 "越小越好" 的因子翻转为 "越大越好"。
    """
    s = winsorize_mad(s, n=mad_n)
    if method == 'zscore':
        s = zscore(s)
    elif method == 'rank':
        s = rank_normalize(s) - 0.5   # 中心化到 0 附近
    if controls is not None:
        s = neutralize(s, controls)
    if direction == -1:
        s = -s
    return s


# ----------------- 5. 因子合成 -----------------
def equal_weight_combine(factor_panel: pd.DataFrame) -> pd.Series:
    """
    等权合成。
    factor_panel : DataFrame, index=fund_id, columns=factor_name
    """
    return factor_panel.mean(axis=1, skipna=True)


def ic_weighted_combine(factor_panel: pd.DataFrame,
                        ic_history: pd.DataFrame) -> pd.Series:
    """
    IC 加权合成。
        w_k = mean(IC_k)
        score = sum_k w_k * f_k
    ic_history : DataFrame, index=date, columns=factor_name, 每期截面 IC 值。
    """
    w = ic_history.mean(axis=0, skipna=True)
    w = w / w.abs().sum()
    common = factor_panel.columns.intersection(w.index)
    return (factor_panel[common] * w[common]).sum(axis=1, skipna=True)


def icir_weighted_combine(factor_panel: pd.DataFrame,
                          ic_history: pd.DataFrame) -> pd.Series:
    """
    ICIR 加权（更稳健）：
        w_k = mean(IC_k) / std(IC_k)
    """
    mu = ic_history.mean(axis=0, skipna=True)
    sd = ic_history.std(axis=0, ddof=1, skipna=True).replace(0, np.nan)
    w = (mu / sd).fillna(0)
    if w.abs().sum() == 0:
        return pd.Series(np.nan, index=factor_panel.index)
    w = w / w.abs().sum()
    common = factor_panel.columns.intersection(w.index)
    return (factor_panel[common] * w[common]).sum(axis=1, skipna=True)


def max_icir_combine(factor_panel: pd.DataFrame,
                     ic_history: pd.DataFrame) -> pd.Series:
    """
    最大化复合 ICIR（类似均值-方差形式）：
        w* = Sigma^{-1} * mean(IC)
    其中 Sigma 是 IC 的协方差矩阵。
    """
    mu = ic_history.mean(axis=0, skipna=True)
    cov = ic_history.cov()
    try:
        w = np.linalg.solve(cov.values + 1e-6 * np.eye(len(cov)), mu.values)
    except np.linalg.LinAlgError:
        return equal_weight_combine(factor_panel)
    w = pd.Series(w, index=mu.index)
    if w.abs().sum() == 0:
        return equal_weight_combine(factor_panel)
    w = w / w.abs().sum()
    common = factor_panel.columns.intersection(w.index)
    return (factor_panel[common] * w[common]).sum(axis=1, skipna=True)


# ----------------- 6. 单因子有效性检验 -----------------
def compute_ic(factor_panel_history: dict[pd.Timestamp, pd.Series],
               forward_return: pd.DataFrame,
               method: str = 'rank') -> pd.Series:
    """
    在多个截面日上计算因子 IC。
    factor_panel_history : {eval_date: Series(index=fund_id)}
    forward_return       : DataFrame, index=eval_date, columns=fund_id, 下一调仓期收益
    method               : 'rank' (Spearman) or 'pearson'
    """
    ics = {}
    for d, fac in factor_panel_history.items():
        if d not in forward_return.index:
            continue
        ret = forward_return.loc[d]
        df = pd.concat([fac, ret], axis=1, join='inner').dropna()
        if len(df) < 10:
            continue
        if method == 'rank':
            ic = df.iloc[:, 0].corr(df.iloc[:, 1], method='spearman')
        else:
            ic = df.iloc[:, 0].corr(df.iloc[:, 1])
        ics[d] = ic
    return pd.Series(ics).sort_index()


def ic_summary(ic_series: pd.Series) -> dict:
    """IC 摘要：均值、IR、胜率、t 统计。"""
    n = ic_series.dropna().count()
    if n < 6:
        return {'IC_mean': np.nan, 'IC_IR': np.nan, 'IC_winrate': np.nan, 'n': n}
    mu = ic_series.mean()
    sd = ic_series.std(ddof=1)
    ir = mu / sd if sd > 0 else np.nan
    wr = (ic_series > 0).mean()
    t_stat = mu / (sd / np.sqrt(n)) if sd > 0 else np.nan
    return {
        'IC_mean': float(mu),
        'IC_IR': float(ir) if not pd.isna(ir) else np.nan,
        'IC_winrate': float(wr),
        't_stat': float(t_stat) if not pd.isna(t_stat) else np.nan,
        'n': int(n),
    }

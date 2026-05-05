"""
demo.py
=======
端到端示例：
1. 构造模拟数据（净值、基准、风格因子、持仓）
2. 实例化各类因子
3. 计算横截面因子矩阵
4. 预处理 + 合成综合得分
5. 输出 TOP-N 基金

运行：
    cd fund_factors
    python -m examples.demo
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from base.factor_base import FactorContext

# 因子导入
from factors.performance import (
    CumulativeReturn, AnnualizedReturn, ExcessReturnVsBench, WinRateMonthly,
    SharpeRatio, SortinoRatio, CalmarRatio, InformationRatio,
)
from factors.risk import (
    AnnualizedVol, MaxDrawdown, DownsideVol,
    HistoricalVaR, ConditionalVaR, TrackingError, BetaToBench,
)
from factors.style import StyleExposure, StyleR2, StyleDrift, DEFAULT_STYLES
from factors.industry import IndustryHHI, EffectiveIndustryNumber, Top3IndustryWeight
from factors.holding import Top10Concentration, StockHHI, StockNumber, TurnoverRate
from factors.manager import TM_Alpha, TM_Gamma, HM_Alpha, HM_Beta2
from factors.persistence import (
    HurstExponent, QuarterlyWinStreak, RankStability, ReturnAutocorr,
)

from preprocess.processor import preprocess_factor, equal_weight_combine


# ============================================================
# 1. 模拟数据
# ============================================================
def make_mock_data(n_funds: int = 30, n_days: int = 1000,
                   seed: int = 42) -> dict:
    """生成模拟数据集，仅用于演示模块联通。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    fund_ids = [f"F{str(i).zfill(4)}" for i in range(n_funds)]

    # 风格因子日收益
    styles = DEFAULT_STYLES
    f_ret = rng.normal(0, 0.008, size=(n_days, len(styles)))
    factor_ret_df = (
        pd.DataFrame(f_ret, index=dates, columns=styles)
        .stack().rename('ret').reset_index()
        .rename(columns={'level_0': 'date', 'level_1': 'factor_name'})
    )

    # 基金日收益：每只基金随机加载到几个风格因子，再加 alpha
    nav_records = []
    for fid in fund_ids:
        loadings = rng.normal(0, 0.5, size=len(styles))
        idio = rng.normal(0.0002, 0.012, size=n_days)   # 个体噪声
        ret = (f_ret @ loadings) + idio
        nav_records.append(pd.DataFrame({
            'date': dates, 'fund_id': fid,
            'ret': ret,
            'adj_nav': np.exp(np.cumsum(ret)),
        }))
    nav_df = pd.concat(nav_records, ignore_index=True)

    # 基准
    bench_ret = rng.normal(0.0003, 0.011, size=n_days)
    bench_df = pd.DataFrame({
        'date': dates, 'bench_id': '中证偏股', 'ret': bench_ret
    })

    # 持仓（季度披露，每只基金 30 只重仓股）
    industries = ['电子', '食品饮料', '医药生物', '电力设备', '银行',
                  '非银金融', '机械设备', '计算机', '汽车', '化工']
    quarter_ends = pd.date_range(end=dates[-1], freq='QE', periods=8)
    quarter_ends = [d for d in quarter_ends if d <= dates[-1]]

    holdings = []
    for fid in fund_ids:
        for d in quarter_ends:
            n_stocks = 30
            stocks = [f"S{str(rng.integers(0, 500)).zfill(4)}" for _ in range(n_stocks)]
            inds = rng.choice(industries, size=n_stocks)
            ws = rng.dirichlet(np.ones(n_stocks)) * 0.6   # 重仓覆盖 60%
            for st, ind, w in zip(stocks, inds, ws):
                holdings.append({'date': d, 'fund_id': fid,
                                 'stock_id': st, 'industry': ind, 'weight': w})
    holding_df = pd.DataFrame(holdings)

    # 基金信息
    fund_info_df = pd.DataFrame({
        'fund_id': fund_ids,
        'type': '偏股混合型',
        'setup_date': dates[0] - pd.Timedelta(days=int(rng.integers(500, 2000))),
        'manager_start_date': dates[0] - pd.Timedelta(days=int(rng.integers(200, 1500))),
    })

    return {
        'nav_df': nav_df,
        'bench_df': bench_df,
        'factor_ret_df': factor_ret_df,
        'holding_df': holding_df,
        'fund_info_df': fund_info_df,
        'eval_date': dates[-1],
    }


# ============================================================
# 2. 主流程
# ============================================================
def main():
    print("=" * 60)
    print("基金画像因子体系 - 端到端 Demo")
    print("=" * 60)

    data = make_mock_data(n_funds=30, n_days=1000)
    ctx = FactorContext(
        nav_df=data['nav_df'],
        bench_df=data['bench_df'],
        factor_ret_df=data['factor_ret_df'],
        holding_df=data['holding_df'],
        fund_info_df=data['fund_info_df'],
        rf=0.02,
        eval_date=data['eval_date'],
        lookback=252,
    )
    print(f"截面日: {ctx.eval_date.date()}, 基金数: {data['nav_df']['fund_id'].nunique()}")

    # 实例化要计算的因子
    factors = [
        # 业绩
        CumulativeReturn(window=63), CumulativeReturn(window=252),
        AnnualizedReturn(window=252),
        ExcessReturnVsBench(),
        WinRateMonthly(),
        # 风险调整
        SharpeRatio(), SortinoRatio(), CalmarRatio(), InformationRatio(),
        # 风险
        AnnualizedVol(), MaxDrawdown(), DownsideVol(),
        HistoricalVaR(), ConditionalVaR(), TrackingError(), BetaToBench(),
        # 风格
        StyleExposure('SIZE'), StyleExposure('VALUE'), StyleExposure('MOMENTUM'),
        StyleR2(), StyleDrift(),
        # 行业
        IndustryHHI(), EffectiveIndustryNumber(), Top3IndustryWeight(),
        # 持仓
        Top10Concentration(), StockHHI(), StockNumber(), TurnoverRate(),
        # 经理
        TM_Alpha(), TM_Gamma(), HM_Alpha(), HM_Beta2(),
        # 持续性
        HurstExponent(), QuarterlyWinStreak(),
        RankStability(), ReturnAutocorr(),
    ]

    # 横截面计算
    panel = {}
    for f in factors:
        ser = f.compute(ctx)
        panel[f.name] = ser
        valid = ser.dropna().shape[0]
        print(f"  {f.name:<18} 有效样本: {valid}/{len(ser)}")
    panel = pd.DataFrame(panel)

    print("\n--- 原始因子矩阵 (前 5 只基金) ---")
    print(panel.head().round(3))

    # 预处理：每个因子按 direction 调整方向 + zscore
    print("\n--- 预处理后 ---")
    processed = {}
    for f in factors:
        if f.name in panel.columns:
            processed[f.name] = preprocess_factor(
                panel[f.name],
                method='zscore',
                direction=f.direction if f.direction != 0 else 1,
            )
    processed = pd.DataFrame(processed)
    print(processed.head().round(3))

    # 等权合成（实战中应改为 ICIR 加权）
    score = equal_weight_combine(processed)
    score = score.sort_values(ascending=False)

    print("\n--- TOP-10 基金（等权合成得分） ---")
    print(score.head(10).round(3))

    return panel, processed, score


if __name__ == "__main__":
    main()

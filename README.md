# 基金画像因子体系（Fund Profile Factor System）

> 用于 FOF 量化选基的多维度因子库，覆盖业绩、风险、风格、持仓、经理能力、持续性六大维度，约 50 个底层因子。

## 数据契约

所有模块以下列三张长表为标准输入：

| 表名 | 索引 | 主要字段 |
|---|---|---|
| `nav_df` | `(date, fund_id)` | `nav`, `adj_nav`, `ret` |
| `holding_df` | `(date, fund_id, stock_id)` | `weight`, `mkt_value`, `industry` |
| `stock_ret_df` | `(date, stock_id)` | `ret`, `mkt_cap`, `industry`, `pb`, `pe` |
| `bench_df` | `(date, bench_id)` | `ret` |
| `factor_ret_df` | `(date, factor_name)` | `ret` (Barra 风格因子日收益) |

约定：
- 收益率均为对数收益 `ret = ln(adj_nav_t / adj_nav_{t-1})`，年化系数 `252`
- 基金类型代码：`1=普通股票型, 2=偏股混合型, 3=灵活配置, 4=债基, 5=指数, 6=QDII`
- 行业分类：申万一级（31 个）

---

## 一、业绩表现因子（Performance）

| 因子代码 | 含义 | 公式 |
|---|---|---|
| `RET_3M / 6M / 12M / 24M / 36M` | N 月累计收益 | $R_T = \prod_{t=1}^{T}(1+r_t) - 1$ |
| `ANN_RET` | 年化收益 | $R_{ann} = (1+R_T)^{252/T} - 1$ |
| `EXC_RET_BENCH` | 相对基准超额 | $\alpha_{bench} = R_{fund} - R_{bench}$ |
| `EXC_RET_PEER` | 相对同类超额 | $\alpha_{peer} = R_{fund} - \bar{R}_{peer}$ |
| `WIN_RATE_M` | 月度胜率 | $\frac{1}{N}\sum \mathbb{1}\{r_m > r_{m,bench}\}$ |
| `BEST_DAY / WORST_DAY` | 极值日 | 单日最大/最小收益 |

## 二、风险特征因子（Risk）

| 因子代码 | 含义 | 公式 |
|---|---|---|
| `VOL_ANN` | 年化波动率 | $\sigma_{ann} = \sqrt{252}\cdot \text{std}(r_t)$ |
| `MDD` | 最大回撤 | $MDD = \min_t \frac{NAV_t - \max_{s\le t} NAV_s}{\max_{s\le t} NAV_s}$ |
| `MDD_DAYS` | 回撤恢复天数 | 从峰值到收复峰值的最大持续天数 |
| `DOWNSIDE_VOL` | 下行波动率 | $\sigma_d = \sqrt{\frac{252}{N}\sum_t \min(r_t - MAR, 0)^2}$ |
| `VAR_95` | 95% 历史 VaR | $VaR = -\text{Quantile}_{5\%}(r_t)$ |
| `CVAR_95` | 95% 条件 VaR | $CVaR = -E[r_t \mid r_t < -VaR]$ |
| `TRACK_ERROR` | 跟踪误差 | $TE = \sqrt{252}\cdot \text{std}(r_t - r_{bench,t})$ |
| `BETA_BENCH` | 基准 Beta | $\beta = \text{Cov}(r_p, r_b)/\text{Var}(r_b)$ |

## 三、风险调整后收益因子（Risk-Adjusted）

| 因子代码 | 含义 | 公式 |
|---|---|---|
| `SHARPE` | 夏普比率 | $\text{Sharpe} = (R_{ann} - r_f)/\sigma_{ann}$ |
| `SORTINO` | 索提诺比率 | $\text{Sortino} = (R_{ann} - MAR)/\sigma_d$ |
| `CALMAR` | 卡玛比率 | $\text{Calmar} = R_{ann}/|MDD|$ |
| `INFO_RATIO` | 信息比率 | $\text{IR} = \alpha_{bench,ann}/TE$ |
| `TREYNOR` | 特雷诺比率 | $\text{Treynor} = (R_{ann} - r_f)/\beta$ |

## 四、风格特征因子（Style — Barra-like）

通过净值对风格因子收益做时序回归：
$$r_{p,t} - r_{f,t} = \alpha + \sum_{i=1}^{K} \beta_i F_{i,t} + \varepsilon_t$$

其中 $F_i$ 包括 7 个 Barra 风格因子：`SIZE`、`VALUE`、`GROWTH`、`MOMENTUM`、`QUALITY`、`VOLATILITY`、`LIQUIDITY`。

| 因子代码 | 含义 |
|---|---|
| `BETA_SIZE` | 大小盘暴露（正值偏大盘） |
| `BETA_VALUE` | 价值/成长暴露（正值偏价值） |
| `BETA_MOM` | 动量暴露 |
| `BETA_QUALITY` | 质量暴露（高 ROE） |
| `BETA_VOL` | 高波动暴露 |
| `STYLE_DRIFT` | 风格漂移度（滚动暴露的标准差均值） |
| `STYLE_R2` | 风格解释度 |

## 五、行业配置因子（Industry）

设基金 $p$ 在行业 $j$ 的权重为 $w_{p,j}$，基准为 $w_{b,j}$：

| 因子代码 | 含义 | 公式 |
|---|---|---|
| `IND_HHI` | 行业集中度 | $HHI = \sum_j w_{p,j}^2$ |
| `IND_DEVIATION` | 行业偏离度 | $D = \sum_j |w_{p,j} - w_{b,j}|$ |
| `IND_NUM` | 有效持仓行业数 | $N_{eff} = 1/\sum_j w_{p,j}^2$ |
| `IND_ROTATION` | 行业轮动速度 | $\frac{1}{2}\sum_j |w_{p,j,t} - w_{p,j,t-1}|$（季度环比） |
| `TOP3_IND` | 前三大行业占比 | $\sum_{top3} w_{p,j}$ |

## 六、持仓特征因子（Holding）

| 因子代码 | 含义 | 公式 |
|---|---|---|
| `TOP10_CONC` | 前十大集中度 | $\sum_{i=1}^{10} w_{p,i}$ |
| `STOCK_HHI` | 个股 HHI | $\sum_i w_{p,i}^2$ |
| `STOCK_NUM` | 持股数量 | 持仓股票数（全部披露口径） |
| `TURNOVER` | 双边换手率 | $\text{TO} = \frac{\min(\text{Buy}, \text{Sell})}{\overline{AUM}}$（半年） |
| `OVERLAP_PEER` | 同类持仓重合度 | 与同类基金平均持仓的余弦相似度 |
| `CROWDING` | 抱团股暴露 | 重仓股中"机构抱团股"权重之和 |

## 七、基金经理能力因子（Manager Skill）

### Treynor-Mazuy (T-M) 二次项模型
$$r_{p,t} - r_{f,t} = \alpha_{TM} + \beta(r_{m,t}-r_{f,t}) + \gamma(r_{m,t}-r_{f,t})^2 + \varepsilon_t$$

- $\alpha_{TM}$：选股能力
- $\gamma$：择时能力（$\gamma > 0$ 表示有正向择时）

### Henriksson-Merton (H-M) 双 Beta 模型
$$r_{p,t} - r_{f,t} = \alpha_{HM} + \beta_1(r_{m,t}-r_{f,t}) + \beta_2 D_t(r_{m,t}-r_{f,t}) + \varepsilon_t$$

其中 $D_t = \mathbb{1}\{r_{m,t} > r_{f,t}\}$，$\beta_2 > 0$ 表示牛市增加 Beta（正向择时）。

### Brinson 归因
$$\text{Allocation}_j = (w_{p,j} - w_{b,j}) \cdot r_{b,j}$$
$$\text{Selection}_j = w_{b,j} \cdot (r_{p,j} - r_{b,j})$$
$$\text{Interaction}_j = (w_{p,j} - w_{b,j}) \cdot (r_{p,j} - r_{b,j})$$

| 因子代码 | 含义 |
|---|---|
| `ALPHA_TM` | T-M 选股能力 |
| `GAMMA_TM` | T-M 择时能力 |
| `ALPHA_HM` | H-M 选股能力 |
| `BETA2_HM` | H-M 择时能力 |
| `BRINSON_SEL` | Brinson 选股贡献 |
| `MANAGER_TENURE` | 现任基金经理任职年限 |

## 八、业绩持续性因子（Persistence）

| 因子代码 | 含义 | 公式 |
|---|---|---|
| `RET_HURST` | Hurst 指数 | R/S 分析得到的 H，>0.5 表示趋势性 |
| `WIN_QUARTERS` | 季度连续胜率 | 过去 N 季度跑赢同类的比例 |
| `RANK_STABILITY` | 排名稳定性 | $1 - \text{std}(\text{rank}_t)/\text{mean}(\text{rank}_t)$ |
| `ROLLING_IR_MEAN` | 滚动 IR 均值 | 滚动 6 月 IR 的平均 |
| `RET_AUTOCORR` | 收益自相关 | $\rho_1 = \text{Corr}(r_t, r_{t-1})$ |

---

## 因子预处理流程

1. **缺失处理**：MICE 多重插补 / 中位数填充 / 直接剔除（视缺失率）
2. **异常值处理**：MAD 方法
   $$\tilde{x}_i = \begin{cases} x_{med} + 3 \cdot MAD & x_i > x_{med} + 3 \cdot MAD \\ x_{med} - 3 \cdot MAD & x_i < x_{med} - 3 \cdot MAD \\ x_i & \text{otherwise} \end{cases}$$
   其中 $MAD = 1.4826 \cdot \text{median}(|x_i - x_{med}|)$
3. **标准化**：z-score 或 rank 标准化
4. **中性化**：对基金类型、规模、成立时间做线性回归取残差
   $$x_i^{neut} = x_i - \hat{\beta}^T Z_i$$

## 因子合成

- **等权法**：$\text{Score} = \frac{1}{K}\sum_k \tilde{f}_k$
- **IC 加权**：$w_k = \overline{IC_k}$，$\text{Score} = \sum_k w_k \tilde{f}_k$
- **ICIR 加权**：$w_k = \overline{IC_k}/\sigma(IC_k)$
- **最大化复合 ICIR**：$w^* = \Sigma^{-1} \mathbb{IC}$，类似均值-方差形式

# FOF 量化选基策略 — 基金画像因子体系完整文档

> 基于多维度基金画像因子的 FOF（Fund of Funds）量化选基框架，覆盖策略规划、因子体系、预处理与合成、回测评估全流程。

---

## 目录

1. [项目背景与目标](#一项目背景与目标)
2. [整体架构](#二整体架构)
3. [基金池定义与数据准备](#三基金池定义与数据准备)
4. [数据契约](#四数据契约)
5. [因子体系](#五因子体系)
   - 5.1 业绩表现因子
   - 5.2 风险特征因子
   - 5.3 风险调整收益因子
   - 5.4 风格特征因子（Barra-like）
   - 5.5 行业配置因子
   - 5.6 持仓特征因子
   - 5.7 基金经理能力因子
   - 5.8 业绩持续性因子
6. [因子预处理](#六因子预处理)
7. [因子有效性检验](#七因子有效性检验)
8. [因子合成与打分模型](#八因子合成与打分模型)
9. [组合构建](#九组合构建)
10. [回测与评估](#十回测与评估)
11. [风险管理与监控](#十一风险管理与监控)
12. [技术实施方案](#十二技术实施方案)
13. [项目时间线](#十三项目时间线)
14. [使用示例](#十四使用示例)
15. [附录：因子速查表](#附录因子速查表)

---

## 一、项目背景与目标

随着公募基金数量突破万只，单纯依赖历史收益率排名的选基方式存在明显缺陷：业绩均值回归、风格漂移难以识别、基金经理变更等因素都会侵蚀选基效果。基金画像因子方法通过多维度刻画基金的"内在特征"，能够更前瞻、更稳定地识别优质基金。

**核心目标：**

- 构建一套覆盖主动权益型基金的量化选基框架，第一阶段聚焦主动管理型偏股基金
- 策略层面：年化超额收益（相对中证偏股基金指数）≥ 4%，信息比率 ≥ 0.5，最大回撤可控
- 输出可解释、可复现、可迭代的因子库与组合管理体系
- 可拓展至债券型 FOF、"固收＋"FOF 等其他子策略

**传统方法的局限：**

传统选基多依赖近期业绩排名，存在三个根本性问题：① 排名均值回归：短期业绩最好的基金往往在下一期回落；② 风格敞口混淆：牛市中所有高 Beta 基金业绩都好，但这与管理人能力无关；③ 幸存者偏差：若只看存续基金，历史统计会系统高估整体收益。因子画像方法通过拆解业绩来源、分离 beta 与 alpha、引入经理能力与持续性维度，试图在噪声中找到真实信号。

---

## 二、整体架构

```
fund_factors/
├── README.md                    ← 本文件（完整项目文档）
├── base/
│   ├── factor_base.py           ← 因子基类、数据契约（FactorBase / FactorContext）
│   └── data_loader.py           ← 数据加载与对齐（待实现，对接 Wind/聚源/Choice）
├── factors/
│   ├── performance.py           ← 业绩 + 风险调整收益（9 个因子）
│   ├── risk.py                  ← 风险特征（8 个因子）
│   ├── style.py                 ← Barra 风格暴露（7+ 个因子）
│   ├── industry.py              ← 行业配置（5 个因子）
│   ├── holding.py               ← 持仓特征（6 个因子）
│   ├── manager.py               ← 经理能力 TM/HM/Brinson（6 个因子）
│   └── persistence.py           ← 业绩持续性（5 个因子）
├── preprocess/
│   └── processor.py             ← 缩尾 / 标准化 / 中性化 / IC 检验 / 合成
└── examples/
    └── demo.py                  ← 端到端示例（模拟数据 → 因子 → 预处理 → 打分）
```

**核心设计原则：**

- 所有因子继承统一的 `FactorBase`，对外只暴露 `compute(ctx) -> Series` 一个接口，便于批量调度
- `FactorContext` 统一封装五类数据（净值、持仓、个股收益、基准、风格因子收益、基金信息），避免参数到处传递
- 持仓型因子（行业、持仓模块）与净值型因子走两条计算路径，互不干扰
- 每个因子都有 `direction` 标记（1=越大越好 / -1=越小越好 / 0=中性），预处理时自动统一方向
- 因子计算与因子合成完全解耦：因子层只产出原始值，预处理层负责清洗和合成

---

## 三、基金池定义与数据准备

### 3.1 投资范围

第一阶段聚焦主动管理型偏股基金（普通股票型、偏股混合型、灵活配置型中权益仓位 ≥ 60% 的部分），后续可扩展至债基、QDII 等。

### 3.2 基金池筛选条件

| 筛选维度 | 标准 | 说明 |
|---|---|---|
| 成立时间 | ≥ 2 年 | 保证业绩样本充足 |
| 规模 | 2 亿—100 亿 | 避免迷你基金清盘风险与大基金灵活度不足 |
| 基金经理 | 过去 1 年未变更 | 历史数据需与现任经理挂钩 |
| 产品形态 | 排除定开/持有期限制类 | 保证申赎流动性 |
| 数据完整性 | 净值缺失率 < 5% | 基础数据质量要求 |

### 3.3 数据来源与频率

- **主数据源**：Wind（推荐）/ 聚源 / Choice，互为校验
- **净值**：日频复权净值，用于计算所有净值类因子
- **持仓**：季报前十大重仓股（季频），半年报/年报全部持仓（半年频）
- **基金经理**：历任经理信息、任职区间
- **基准行情**：沪深 300、中证 500、中证 800、中证偏股基金指数 日频收益
- **风格因子**：Barra CNE5/CNE6 日频因子收益（商业数据源），或自构造 long-short 组合

### 3.4 数据质量处理

- **幸存者偏差**：保留已清盘/合并基金的历史数据，不能只用存续基金
- **净值跳变检测**：单日收益绝对值 > 20% 视为异常，触发人工核查
- **持仓不全处理**：季报仅披露前十大（权重和约 30%-60%），需在因子计算中显式标注口径
- **分红复权**：统一使用累计复权净值，排除分红扰动
- **数据时滞**：季报在次季度第 15 个工作日前披露，使用时需做 T+15 延迟处理

---

## 四、数据契约

所有模块以下列长表为标准输入，索引约定为 `MultiIndex(date, fund_id/stock_id/bench_id)`：

| 表名 | 标准索引 | 主要字段 |
|---|---|---|
| `nav_df` | `(date, fund_id)` | `adj_nav`（复权净值）、`ret`（日简单收益） |
| `holding_df` | `(date, fund_id, stock_id)` | `weight`（仓位权重）、`mkt_value`、`industry`（申万一级） |
| `stock_ret_df` | `(date, stock_id)` | `ret`、`mkt_cap`、`industry`、`pb`、`pe` |
| `bench_df` | `(date, bench_id)` | `ret` |
| `factor_ret_df` | `(date, factor_name)` | `ret`（Barra 风格因子日收益） |
| `fund_info_df` | `fund_id` | `type`、`setup_date`、`manager_id`、`manager_start_date`、`company` |

**约定：**
- 收益率统一使用简单收益 `ret = adj_nav_t / adj_nav_{t-1} - 1`，年化系数 252
- 行业分类：申万一级（2021 版，31 个）
- 基金类型代码：`1=普通股票型` / `2=偏股混合型` / `3=灵活配置` / `4=债基` / `5=指数` / `6=QDII`
- 所有 DataFrame 须按索引排序，避免时序错位

---

## 五、因子体系

因子体系分为 8 大类、约 46 个细化因子，每个因子均有：计算公式、所需数据、经济学逻辑、预期方向。

---

### 5.1 业绩表现因子（Performance）

**经济学逻辑：** 历史收益是最直接的选基依据，多窗口设计捕捉短期动量与长期稳健性，超额收益剥离市场 beta，胜率反映一致性。

| 因子代码 | 含义 | 公式 | 方向 |
|---|---|---|---|
| `RET_3M` | 近 3 月累计收益 | $R_T = \prod_{t=1}^{T}(1+r_t) - 1$，$T=63$ | ↑ |
| `RET_6M` | 近 6 月累计收益 | 同上，$T=126$ | ↑ |
| `RET_12M` | 近 12 月累计收益 | 同上，$T=252$ | ↑ |
| `RET_24M` | 近 24 月累计收益 | 同上，$T=504$ | ↑ |
| `RET_36M` | 近 36 月累计收益 | 同上，$T=756$ | ↑ |
| `ANN_RET` | 年化几何收益 | $R_{ann} = (1+R_T)^{252/T} - 1$ | ↑ |
| `EXC_RET_BENCH` | 相对基准超额（年化） | $\alpha_{bench} = R_{p,ann} - R_{b,ann}$ | ↑ |
| `WIN_RATE_M` | 月度胜率 | $WR = \frac{1}{N}\sum_{m=1}^{N} \mathbf{1}\{R_{p,m} > R_{b,m}\}$ | ↑ |

**参数说明：**
- 基准 `bench_id` 默认使用"中证偏股基金指数"，也可配置为沪深 300 或同类均值
- 多窗口超额因子建议以 IC_IR 加权合成，而非全部独立使用（相关性较高）

---

### 5.2 风险特征因子（Risk）

**经济学逻辑：** 下行风险比总波动率对投资者效用的损害更大；MDD 反映极端情景下的资本保全能力；跟踪误差刻画相对基准的主动程度。

| 因子代码 | 含义 | 公式 | 方向 |
|---|---|---|---|
| `VOL_ANN` | 年化波动率 | $\sigma_{ann} = \sqrt{252} \cdot \text{std}(r_t)$ | ↓ |
| `MDD` | 最大回撤（负值） | $MDD = \min_t \frac{NAV_t - \max_{s \le t} NAV_s}{\max_{s \le t} NAV_s}$ | ↑（越接近 0 越好） |
| `MDD_DAYS` | 回撤未恢复最长天数 | 从峰值到谷底的交易日跨度 | ↓ |
| `DOWNSIDE_VOL` | 下行波动率（年化） | $\sigma_d = \sqrt{\frac{252}{N}\sum_{t} \min(r_t - MAR, 0)^2}$ | ↓ |
| `VAR_95` | 95% 历史 VaR（正值） | $VaR = -\text{Quantile}_{5\%}(r_t)$ | ↓ |
| `CVAR_95` | 95% 条件 VaR（ES） | $CVaR = -\mathbb{E}[r_t \mid r_t \le -VaR]$ | ↓ |
| `TRACK_ERROR` | 跟踪误差（年化） | $TE = \sqrt{252} \cdot \text{std}(r_{p,t} - r_{b,t})$ | ↓ |
| `BETA_BENCH` | 相对基准 Beta | $\beta = \text{Cov}(r_p, r_b) / \text{Var}(r_b)$ | 中性 |

**参数说明：**
- `MAR`（最小可接受收益）默认 0，即以日收益为 0 作为下行参照，可配置为无风险利率日折算值
- `BETA_BENCH` 不直接用于打分，主要作为组合层面的风险约束指标

---

### 5.3 风险调整收益因子（Risk-Adjusted）

**经济学逻辑：** 单纯高收益可能由高风险换来；风险调整后收益才是筛选真正优质基金的核心依据。卡玛比率对 FOF 选基尤为重要，因为 FOF 客户对极端回撤容忍度极低。

| 因子代码 | 含义 | 公式 | 方向 |
|---|---|---|---|
| `SHARPE` | 夏普比率 | $\text{Sharpe} = (R_{ann} - r_f) / \sigma_{ann}$ | ↑ |
| `SORTINO` | 索提诺比率 | $\text{Sortino} = (R_{ann} - MAR) / \sigma_d$ | ↑ |
| `CALMAR` | 卡玛比率 | $\text{Calmar} = R_{ann} / \vert MDD \vert$ | ↑ |
| `INFO_RATIO` | 信息比率 | $IR = \alpha_{bench,ann} / TE$ | ↑ |
| `TREYNOR` | 特雷诺比率 | $\text{Treynor} = (R_{ann} - r_f) / \beta$ | ↑ |

**注意：** Sharpe/Sortino/Calmar 三者相关性较高，合成时建议降权或仅保留 IC_IR 最高的两个。信息比率在中国市场选基意义显著，建议单独保留为核心因子。

---

### 5.4 风格特征因子（Style — Barra-like）

**经济学逻辑：** 风格暴露决定了基金的系统性收益来源，而非经理的主动贡献。通过回归拆解，可以判断基金是否"名实相符"（如宣称价值风格但实际更接近成长），以及风格是否稳定。

**模型：** 用基金日超额收益对 K 个风格因子日收益做时序 OLS 回归

$$r_{p,t} - r_{f,t} = \alpha + \sum_{i=1}^{K} \beta_i F_{i,t} + \varepsilon_t$$

$F_i$ 为 Barra 风格因子日收益（来自商业数据源或自构造 long-short 组合）。

| 因子代码 | 含义 | 预期方向 |
|---|---|---|
| `BETA_SIZE` | 大盘股暴露（正=偏大盘） | 中性 |
| `BETA_VALUE` | 价值因子暴露（正=偏价值） | 中性 |
| `BETA_GROWTH` | 成长因子暴露 | 中性 |
| `BETA_MOMENTUM` | 动量因子暴露 | 中性 |
| `BETA_QUALITY` | 质量因子暴露（ROE、盈利稳定） | ↑ |
| `BETA_VOLATILITY` | 高波动暴露 | ↓ |
| `STYLE_DRIFT` | 风格漂移度（滚动暴露标准差均值） | ↓（稳定性越好越好） |
| `STYLE_R2` | 风格解释度（回归 R²） | 中性 |

**风格漂移度公式：**

$$\text{StyleDrift} = \frac{1}{K} \sum_{i=1}^{K} \text{std}_{t}(\hat{\beta}_{i,t})$$

其中 $\hat{\beta}_{i,t}$ 为第 $t$ 个滚动子窗口（如季度）回归得到的暴露。漂移度小表示风格稳定，更易预测未来暴露。

**风格因子自建构造参考：**

| 风格 | 构造方法 |
|---|---|
| SIZE | 按流通市值分十组，最大组 - 最小组日收益之差 |
| VALUE | 按 PE 倒数（E/P）分十组，高 E/P - 低 E/P |
| MOMENTUM | 过去 12 月（排除最近 1 月）收益率排名，前后组差 |
| QUALITY | 综合 ROE、毛利率、资产负债率，前后组差 |

---

### 5.5 行业配置因子（Industry）

**经济学逻辑：** 行业集中度高的基金收益方差大，稳定性差；行业偏离基准过大则面临行业轮动风险；行业轮动速度反映基金经理的行业配置主动程度（换手率代理指标）。

设基金 $p$ 在行业 $j$ 的权重为 $w_{p,j}$，基准行业权重为 $w_{b,j}$，共 $J$ 个行业：

| 因子代码 | 含义 | 公式 | 方向 |
|---|---|---|---|
| `IND_HHI` | 行业 Herfindahl 集中度 | $HHI = \sum_{j=1}^{J} w_{p,j}^2$ | ↓ |
| `IND_NUM` | 有效行业数 | $N_{eff} = 1 / HHI$ | ↑ |
| `IND_DEVIATION` | 行业偏离度 | $D = \sum_j \vert w_{p,j} - w_{b,j} \vert$ | 中性 |
| `TOP3_IND` | 前三大行业权重和 | $\sum_{\text{top3}} w_{p,j}$ | ↓ |
| `IND_ROTATION` | 行业轮动速度（季度） | $0.5 \times \sum_j \vert w_{p,j,t} - w_{p,j,t-1} \vert$ | 中性 |

**数据口径说明：** 基于季报持仓计算，所有权重相对持仓总权重（重仓股之和）归一化，而非相对基金净值。

---

### 5.6 持仓特征因子（Holding）

**经济学逻辑：** 持仓集中度反映组合构建风格（集中持股需要更强的选股能力支撑）；换手率是交易成本代理指标；与同类高度重合意味着 alpha 来源高度同质化，难以产生差异化超额。

| 因子代码 | 含义 | 公式 | 方向 |
|---|---|---|---|
| `TOP10_CONC` | 前十大重仓股权重和 | $\sum_{i=1}^{10} w_{p,i}$ | 中性 |
| `STOCK_HHI` | 个股 HHI | $\sum_i w_{p,i}^2$ | ↓ |
| `STOCK_NUM` | 持股数量（披露口径） | 持仓股票数 | ↑ |
| `TURNOVER` | 近似双边换手率 | $0.5 \times \sum_i \vert w_{i,t} - w_{i,t-1} \vert$（相邻披露期） | ↓ |
| `OVERLAP_PEER` | 与同类持仓余弦相似度 | $\cos(w_p, \bar{w}_{peer}) = \frac{w_p \cdot \bar{w}_{peer}}{\Vert w_p \Vert \cdot \Vert \bar{w}_{peer} \Vert}$ | ↓ |
| `CROWDING` | 抱团股暴露 | 持仓中机构抱团股权重之和 | ↓ |

**抱团股识别方法：** 以所有主动权益基金持仓按权重加总，季度更新，取全市场持仓权重前 100 只（或前 5%）定义为抱团股池。

---

### 5.7 基金经理能力因子（Manager Skill）

#### Treynor-Mazuy（T-M）二次项模型

通过引入市场超额收益的平方项，将总超额收益分解为选股能力与择时能力：

$$r_{p,t} - r_{f,t} = \alpha_{TM} + \beta(r_{m,t} - r_{f,t}) + \gamma(r_{m,t} - r_{f,t})^2 + \varepsilon_t$$

- $\alpha_{TM} > 0$：选股能力显著
- $\gamma > 0$：在市场上涨时增加 Beta（牛市多拿、熊市少拿），具有有效择时能力

#### Henriksson-Merton（H-M）双 Beta 模型

用虚拟变量 $D_t = \mathbf{1}\{r_{m,t} > r_{f,t}\}$ 将牛市与熊市的 Beta 分开：

$$r_{p,t} - r_{f,t} = \alpha_{HM} + \beta_1(r_{m,t} - r_{f,t}) + \beta_2 D_t(r_{m,t} - r_{f,t}) + \varepsilon_t$$

- $\alpha_{HM}$：选股能力（截距）
- $\beta_2 > 0$：牛市时 Beta 更高，具有正向择时能力

#### Brinson 归因（行业层面）

$$\text{Allocation}_j = (w_{p,j} - w_{b,j}) \cdot r_{b,j}$$

$$\text{Selection}_j = w_{b,j} \cdot (r_{p,j} - r_{b,j})$$

$$\text{Interaction}_j = (w_{p,j} - w_{b,j}) \cdot (r_{p,j} - r_{b,j})$$

$$\text{Total\_Selection} = \sum_j \text{Selection}_j$$

| 因子代码 | 含义 | 方向 |
|---|---|---|
| `ALPHA_TM` | T-M 年化选股能力 | ↑ |
| `GAMMA_TM` | T-M 择时能力 | ↑ |
| `ALPHA_HM` | H-M 年化选股能力 | ↑ |
| `BETA2_HM` | H-M 择时能力 | ↑ |
| `BRINSON_SEL` | Brinson 选股贡献（上一持仓期） | ↑ |
| `MANAGER_TENURE` | 现任基金经理任职年限（年） | ↑ |

**注意：** T-M 与 H-M 两套模型对择时的测量有差异，建议并列计算后取 IC_IR 更高的一组，不要重复叠加。样本窗口建议不少于 3 年（约 756 个交易日），以保证统计显著性。

---

### 5.8 业绩持续性因子（Persistence）

**经济学逻辑：** 好基金不只是"曾经好过"，而是"持续地好"。持续性因子用于筛选那些跨市场环境都能保持相对优势的基金，剔除靠运气或特定牛市风口短期超额的基金。

| 因子代码 | 含义 | 公式 | 方向 |
|---|---|---|---|
| `RET_HURST` | Hurst 指数 | R/S 分析：$H = \text{slope}(\log(R/S)$ vs $\log n)$ | ↑（$H > 0.5$ 表趋势） |
| `WIN_QUARTERS` | 近 8 季度跑赢基准比例 | $\frac{1}{8}\sum_{q=1}^{8}\mathbf{1}\{R_{p,q} > R_{b,q}\}$ | ↑ |
| `RANK_STABILITY` | 排名稳定性（变异系数倒数） | $1 / (1 + \vert CV_q \vert)$，$CV = \sigma_q / \mu_q$ | ↑ |
| `ROLLING_IR_MEAN` | 滚动 6 月 IR 均值 | $\bar{IR} = \frac{1}{N}\sum_{k=1}^{N} IR_{6M,k}$ | ↑ |
| `RET_AUTOCORR` | 月度收益自相关 $\rho_1$ | $\rho_1 = \text{Corr}(R_{m}, R_{m-1})$ | ↑ |

**Hurst 指数计算步骤（R/S 法）：**

1. 取不同长度 $n$ 的子序列
2. 计算每段 $R_n/S_n$（极差与标准差之比）
3. 以 $\log(n)$ 对 $\log(R_n/S_n)$ 做线性回归，斜率即为 $H$
4. $H > 0.5$ 表示业绩具有正向持续性，$H < 0.5$ 表示均值回归倾向

---

## 六、因子预处理

所有因子在横截面上依次经历以下处理，保证量纲统一、分布合理。

### 6.1 缺失值处理

按缺失率分档处理：
- 缺失率 < 5%：中位数填充
- 缺失率 5%~20%：MICE（多重插补，迭代随机森林）
- 缺失率 > 20%：该截面直接剔除此因子

### 6.2 异常值处理（MAD 法）

$$MAD = 1.4826 \times \text{median}(\vert x_i - \text{median}(x) \vert)$$

$$\tilde{x}_i = \begin{cases} x_{med} + n \cdot MAD & x_i > x_{med} + n \cdot MAD \\ x_{med} - n \cdot MAD & x_i < x_{med} - n \cdot MAD \\ x_i & \text{otherwise} \end{cases}$$

其中 $n=3$（默认），1.4826 为使 MAD 在正态分布下与 $\sigma$ 一致的校正系数。相比分位数缩尾，MAD 对极端值更稳健。

### 6.3 标准化

两种方式可选，对应不同因子分布特征：

**Z-score（适合分布较规则的因子）：**
$$z_i = \frac{x_i - \bar{x}}{\sigma_x}$$

**Rank 标准化（适合偏态分布，如规模、换手率）：**
$$r_i = \frac{\text{rank}(x_i)}{N} - 0.5 \quad \in (-0.5, 0.5)$$

### 6.4 因子中性化

对控制变量（基金类型、规模、成立时间）做横截面回归，取残差：

$$\tilde{f}_i = f_i - Z_i \hat{\gamma}$$

其中 $Z_i$ 为控制变量矩阵（可包含类型 dummy），$\hat{\gamma}$ 由横截面 OLS 估计。中性化剥除"外生特征"对因子值的影响，让因子更纯粹地反映主动能力。

### 6.5 方向统一

根据 `factor.direction` 字段，对 `direction = -1` 的因子取负值，统一为"因子值越大，该基金越优"的约定，便于后续合成。

---

## 七、因子有效性检验

每个因子上线前须通过以下检验，不达标则剔除或降权。

### 7.1 IC / Rank IC 检验

在每个截面日 $t$ 计算：

$$IC_t = \text{Corr}(f_{i,t},\ r_{i,t+\Delta})$$

其中 $r_{i,t+\Delta}$ 为下一调仓期（如 1 个月）的超额收益。

**评估指标与门槛：**

| 指标 | 计算 | 建议门槛 |
|---|---|---|
| IC 均值 | $\bar{IC} = \frac{1}{T}\sum_t IC_t$ | $\vert\bar{IC}\vert \ge 0.03$ |
| IC_IR | $\bar{IC} / \text{std}(IC)$ | $\ge 0.30$ |
| IC 胜率 | $\frac{1}{T}\sum_t \mathbf{1}\{IC_t > 0\}$ | $\ge 0.55$ |
| t 统计量 | $\bar{IC} / (\text{std}(IC)/\sqrt{T})$ | $\ge 2.0$ |
| IC 衰减 | 滞后期 1/2/3 的 IC 递减速率 | 前 2 期不应过快 |

实践中优先使用 Rank IC（Spearman 相关），对极端值更稳健。

### 7.2 分组回测

将因子值从高到低分为 5 档（Q1 最高），分别构造等权组合，计算每档累计超额收益。要求 Q1 显著优于 Q5，且各档之间具有单调性。

### 7.3 因子相关性筛查

对所有候选因子计算两两 Rank 相关系数，若 $\vert\rho\vert > 0.70$，则保留 IC_IR 更高的一个，剔除另一个，避免合成时信号冗余。

### 7.4 市场分段稳健性

分别统计以下子区间的 IC 均值：
- 牛市（市场涨幅 > 20% 的年份或区间）
- 熊市（市场跌幅 > 20%）
- 震荡市（其余）

要求因子在 3 种市场环境下 IC 符号一致（允许量级不同），说明因子信号具有跨市场泛化能力。

---

## 八、因子合成与打分模型

### 8.1 大类因子内部合成（第一层）

对同类因子先在类内合成，减少冗余，再跨类合成。内部合成方式：

- **等权合成**：简单平均，基准对照
- **IC 加权合成**：$w_k = \bar{IC}_k / \sum_j \bar{IC}_j$
- **IC_IR 加权合成**：$w_k = (\bar{IC}_k / \sigma_{IC_k}) / \sum_j (\bar{IC}_j / \sigma_{IC_j})$

### 8.2 跨类合成（第二层）

将 8 大类因子的大类得分合成为最终综合得分，支持以下方法：

**方法一：线性加权（推荐起步）**

$$\text{Score}_i = \sum_{c=1}^{C} w_c \cdot \tilde{f}_{c,i}$$

权重 $w_c$ 按各大类的历史 IC_IR 动态更新（滚动 24 月）。

**方法二：最大化复合 IC_IR**

$$w^* = \Sigma^{-1} \mu_{IC}$$

其中 $\mu_{IC}$ 为各因子历史 IC 均值向量，$\Sigma$ 为 IC 协方差矩阵，类似均值-方差有效前沿形式，在最大化组合 IC_IR 的意义下最优。

**方法三：机器学习排序（LightGBM）**

- 输入特征：横截面预处理后的因子值矩阵
- 目标变量：下期超额收益的分位数（排序问题）或连续超额（回归问题）
- 训练方式：滚动样本（每月用过去 5 年数据训练，预测当期）
- 防过拟合：特征重要性筛选、树深度限制、交叉验证、Early Stopping
- 注意：截面样本量少（通常 300-500 只基金）时容易过拟合，建议作为对照组而非主策略

---

## 九、组合构建

### 9.1 候选池筛选

基于综合得分取前 25%-30% 作为候选池（约 80-120 只），再叠加质化过滤：
- 剔除规模骤增或骤降（季度变化 > ±50%）
- 剔除近期有监管处罚或舆情风险的基金公司
- 剔除与已持有基金风格/持仓高度重合的基金

### 9.2 权重分配方案

| 方案 | 描述 | 适用场景 |
|---|---|---|
| 等权配置 | 5-10 只基金均分权重 | 基准/对照组 |
| 波动率倒数加权 | $w_i \propto 1/\sigma_i$ | 风险优先场景 |
| 风险平价 | 每只基金贡献等额组合风险 | 均衡风险来源 |
| 均值-方差优化 | 在预期收益（得分代理）约束下最小化组合方差 | 收益优先场景 |

### 9.3 约束条件

- 单基金权重上限：20%
- 单基金经理权重上限：30%（防止经理集中风险）
- 单基金公司权重上限：40%
- 持仓基金数量：5-15 只
- 与基准的最大跟踪误差：≤ 8%（可配置）
- 风格中性约束（可选）：SIZE / VALUE 暴露不超过基准 ±0.3

### 9.4 调仓机制

- **定期调仓**：季度或月度
- **临时触发调仓**：基金经理变更、规模异常、持仓大幅漂移
- **实施时滞**：考虑开放式基金申赎到账 T+7（货币基金）/ T+2（场外），调仓日期与信号日期之间留出合理缓冲

---

## 十、回测与评估

### 10.1 回测设置

| 参数 | 建议值 | 说明 |
|---|---|---|
| 样本总区间 | 2015.01 - 至今 | 涵盖完整牛熊周期 |
| 样本内（训练） | 2015.01 - 2020.12 | 用于因子筛选与模型参数估计 |
| 样本外（测试） | 2021.01 - 至今 | 策略真实评估区间 |
| 调仓频率 | 月度或季度 | 季度对应基金披露节奏 |
| 申购费率 | 0.12%（C 类） | 含销售服务费 |
| 冲击成本 | 0.1% | 规模较大时需调高 |
| 申赎时滞 | T+2（默认）/ T+7 | 按产品类型配置 |
| 基准指数 | 中证偏股基金指数 | 也可用偏股混合型基金平均 |

### 10.2 评估指标体系

| 类别 | 指标 |
|---|---|
| 绝对收益 | 年化收益、最大回撤、Calmar、波动率、Sharpe |
| 相对收益 | 年化超额、信息比率、跟踪误差、月度胜率 |
| 稳健性 | 滚动 12 月超额、超额回撤、信号稳定性 |
| 风险分布 | 收益分布偏度/峰度、尾部 CVaR |

### 10.3 业绩归因

**Brinson 归因：** 拆分选基贡献（选哪些基金）与配置贡献（各类基金权重如何分配）。

**因子归因：** 分析策略超额中有多少来自风格（如持续配置价值风格），有多少来自纯选基 alpha，引导迭代方向。

### 10.4 稳健性分析

- 参数敏感性测试：变化持仓基金数（5/8/10/15）、调仓频率（月/季）、IC 回看窗口（12/24 月）
- 剔除头部贡献者后的表现（去掉最大贡献基金，看剩余是否仍有超额）
- 不同市场状态下分段统计
- 与等权基金指数、随机选基组合对比

---

## 十一、风险管理与监控

### 11.1 策略层面预警

| 预警信号 | 触发条件 | 处理方式 |
|---|---|---|
| 组合大幅回撤 | 策略净值从高点回撤 > 10% | 降低权益敞口至 80% |
| 单基金异常下跌 | 单月跌幅 > -10% 且偏离基准 > 5% | 触发人工审查，必要时提前调仓 |
| 基金经理变更 | 任何持仓基金经理变更 | 立即列入观察名单，下次调仓时重新评估 |
| 规模骤变 | 季度规模变化 > ±50% | 规模骤增导致灵活度下降，需降权 |

### 11.2 定期复盘

- **月度**：组合收益归因，信号有效性跟踪（IC 是否下降）
- **季度**：全量因子 IC 回测，剔除失效因子，引入新因子
- **年度**：完整策略绩效报告，评估因子权重是否需要大幅调整

---

## 十二、技术实施方案

### 12.1 技术栈

| 层次 | 工具 |
|---|---|
| 数据获取 | Wind Python API / 聚源 JuDataAPI / Choice |
| 数据存储 | PostgreSQL（结构化数据） + Parquet（时序因子快照） |
| 因子计算 | Python：numpy / pandas / scipy / statsmodels |
| 机器学习 | LightGBM / scikit-learn |
| 回测引擎 | 自建（基于 pandas，支持按信号日分步推进） |
| 组合优化 | cvxpy（凸优化） |
| 可视化 | Plotly + Streamlit（交互式监控面板） |
| 任务调度 | APScheduler（月末自动触发因子计算与调仓信号生成） |

### 12.2 核心设计约定

- 所有因子继承 `FactorBase`，只需实现 `_compute_one(fund_ret, ctx) -> float`，基类负责循环、对齐、异常捕获
- `FactorContext` 是数据总线，在一次调仓计算中只构建一次，所有因子共享
- 持仓型因子（行业/持仓模块）继承 `HoldingFactorBase`，使用最近一次披露快照而非净值序列
- 因子 `direction` 标记为 `1`（越大越好）/ `-1`（越小越好）/ `0`（中性，不直接参与打分）
- 所有因子计算对 NaN 容忍，`min_obs` 控制最少有效观测数，不达标返回 NaN 而非报错

---

## 十三、项目时间线

| 阶段 | 主要工作 | 里程碑 | 预计周期 |
|---|---|---|---|
| **阶段一：数据工程** | Wind 接入、基金池构建、数据质量验证 | 基金池 300+ 只，净值/持仓清洗完成 | 2 周 |
| **阶段二：因子库建设** | 全部 46 个因子开发与单元测试 | 因子库可运行，输出横截面因子矩阵 | 4 周 |
| **阶段三：因子检验** | IC 检验、分组回测、相关性筛查、市场分段检验 | 筛选出 25-35 个有效因子 | 2 周 |
| **阶段四：合成模型** | IC 加权/ICIR 加权/最大化 ICIR 三套方案 | 综合打分模型输出 TOP-N 排名 | 2 周 |
| **阶段五：回测框架** | 组合构建、调仓逻辑、费率、时滞 | 完整回测引擎上线 | 2 周 |
| **阶段六：验证与调优** | 样本外回测、稳健性分析、参数敏感性 | 策略绩效满足目标要求 | 3 周 |
| **阶段七：上线运维** | 自动化调度、监控面板、季度复盘报告 | 系统完全自动化运行 | 2 周 |

**总周期：约 4 个月**（单人节奏）；团队协作可压缩至 2-3 个月。

---

## 十四、使用示例

### 快速启动（模拟数据，无需真实数据源）

```bash
pip install numpy pandas scipy statsmodels
cd fund_factors
python -m examples.demo
```

### 接入真实数据

```python
from base.factor_base import FactorContext
from factors.performance import SharpeRatio, InformationRatio, CalmarRatio
from factors.manager import TM_Alpha, TM_Gamma
from factors.persistence import HurstExponent, QuarterlyWinStreak
from preprocess.processor import preprocess_factor, icir_weighted_combine
import pandas as pd

# 从 Wind 或数据库加载（用户自行实现 data_loader）
ctx = FactorContext(
    nav_df=load_nav(),           # (date, fund_id) -> adj_nav, ret
    bench_df=load_bench(),       # (date, bench_id) -> ret
    holding_df=load_holding(),   # (date, fund_id) -> stock_id, weight, industry
    fund_info_df=load_info(),    # fund_id -> setup_date, manager_start_date ...
    rf=0.02,
    eval_date=pd.Timestamp("2024-12-31"),
    lookback=252,
)

factors = [SharpeRatio(), InformationRatio(), CalmarRatio(),
           TM_Alpha(), TM_Gamma(), HurstExponent(), QuarterlyWinStreak()]

panel = pd.concat({f.name: f.compute(ctx) for f in factors}, axis=1)
processed = pd.concat(
    {f.name: preprocess_factor(panel[f.name], method='zscore', direction=f.direction)
     for f in factors}, axis=1
)

from preprocess.processor import equal_weight_combine
score = equal_weight_combine(processed).sort_values(ascending=False)
print(score.head(10))   # TOP-10 基金
```

### 扩展新因子

```python
from base.factor_base import FactorBase, FactorContext

class ReturnSkewness(FactorBase):
    """收益偏度：正偏度意味着偶发上涨，整体分布更有利"""
    name = "RET_SKEW"
    category = "RISK"
    direction = 1   # 越大越好

    def _compute_one(self, fund_ret, ctx) -> float:
        return float(fund_ret.skew())
```

---

## 附录：因子速查表

| 类别 | 因子代码 | 方向 | 数据依赖 |
|---|---|---|---|
| 业绩 | RET_3M / 6M / 12M / 24M / 36M | ↑ | 净值 |
| 业绩 | ANN_RET | ↑ | 净值 |
| 业绩 | EXC_RET_BENCH | ↑ | 净值 + 基准 |
| 业绩 | WIN_RATE_M | ↑ | 净值 + 基准 |
| 风险 | VOL_ANN | ↓ | 净值 |
| 风险 | MDD | ↑（越接近 0 越好） | 净值 |
| 风险 | MDD_DAYS | ↓ | 净值 |
| 风险 | DOWNSIDE_VOL | ↓ | 净值 |
| 风险 | VAR_95 / CVAR_95 | ↓ | 净值 |
| 风险 | TRACK_ERROR | ↓ | 净值 + 基准 |
| 风险 | BETA_BENCH | 中性 | 净值 + 基准 |
| 风险调整 | SHARPE / SORTINO / CALMAR | ↑ | 净值 |
| 风险调整 | INFO_RATIO / TREYNOR | ↑ | 净值 + 基准 |
| 风格 | BETA_SIZE / VALUE / GROWTH / MOM / QUALITY / VOL | 中性 | 净值 + 风格因子 |
| 风格 | STYLE_DRIFT | ↓ | 净值 + 风格因子 |
| 风格 | STYLE_R2 | 中性 | 净值 + 风格因子 |
| 行业 | IND_HHI / TOP3_IND | ↓ | 持仓 |
| 行业 | IND_NUM | ↑ | 持仓 |
| 行业 | IND_DEVIATION / IND_ROTATION | 中性 | 持仓 + 基准行业权重 |
| 持仓 | TOP10_CONC | 中性 | 持仓 |
| 持仓 | STOCK_HHI | ↓ | 持仓 |
| 持仓 | STOCK_NUM | ↑ | 持仓 |
| 持仓 | TURNOVER | ↓ | 持仓（相邻两期） |
| 持仓 | OVERLAP_PEER | ↓ | 持仓 + 基金信息 |
| 持仓 | CROWDING | ↓ | 持仓 + 抱团股池 |
| 经理 | ALPHA_TM / ALPHA_HM | ↑ | 净值 + 基准 |
| 经理 | GAMMA_TM / BETA2_HM | ↑ | 净值 + 基准 |
| 经理 | BRINSON_SEL | ↑ | 持仓 + 个股收益 |
| 经理 | MANAGER_TENURE | ↑ | 基金信息 |
| 持续性 | RET_HURST | ↑ | 净值 |
| 持续性 | WIN_QUARTERS | ↑ | 净值 + 基准 |
| 持续性 | RANK_STABILITY | ↑ | 净值 |
| 持续性 | ROLLING_IR_MEAN | ↑ | 净值 + 基准 |
| 持续性 | RET_AUTOCORR | ↑ | 净值 |

---

*最后更新：2026-05-05*

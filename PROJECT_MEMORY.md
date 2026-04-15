# 德国房产投资 RL 模拟器 — 完整项目记忆文档

> **文档用途**：本文档记录了该项目从零开始的完整设计决策、架构定义、文件清单和开发规划。任何 AI 助手读完此文档后，应能完整接续开发工作，无需重新澄清背景。
>
> **最后更新**：2026年4月（对话框一）

---

## 目录

1. 产品背景与核心理解
2. 技术约束与关键决策
3. 系统架构（Phase 1 + Phase 2）
4. MDP 正式定义
5. 完整文件清单与优先级
6. 已完成文件说明（可直接复用）
7. 待开发文件说明
8. 风险清单
9. 开发阶段规划
10. 训练环境说明
11. 论文定位

---

## 1. 产品背景与核心理解

### 1.1 产品是什么

**德国房产投资财务人生规划模拟器**。

用户描述自己当前的财务状态（有没有房、有多少现金、年收入多少），系统模拟接下来 10-15 年里，在不同的决策序列下，最终财务结果的差异。输出是一张**策略结果表**，用户用 filter 自己筛选，系统不推荐特定策略（德国法律限制）。

### 1.2 核心用户故事

> 我，一个普通投资者，当前有 12 万欧现金，年收入 8 万欧，想在慕尼黑买一套 40 万欧的公寓出租。我想知道：在德国税法约束下，不同的持有、融资、出租、翻修、退出方案组合，10 年后我的实际净收益是多少？

### 1.3 产品输出形式

- 一张策略结果表（每行 = 一条完整的决策序列，每列 = 该序列的财务指标）
- 用户可 filter（IRR、FLAG、投机税、净退出收益等）
- 用户可保存感兴趣的方案到账号
- **不排序、不推荐**（排序 = 涉嫌税务结构推荐，违规）
- 每个策略附带 Decision Log（每一步动作对应的税法条款和量化影响）

### 1.4 产品不做什么

- 不提供"你应该选这个方案"的定制化建议
- 不预测房价涨跌（sale_price 由用户输入）
- 不处理商业地产、REIT、股票等非住宅房产投资

### 1.5 场景定义

**场景 A（准备买房）**：用户当前无房产，State 里 properties 列表为空，liquid_cash 是全部自有资金。

**场景 B（已有房产）**：用户已持有 1 套或多套房产，State 包含真实历史数据（已持有年数、当前贷款余额、已产生的 AfA 累计值）。Agent 从真实当前状态出发搜索后续策略。

两个场景**共用同一套系统**，State 格式完全相同，只是初始值不同。

---

## 2. 技术约束与关键决策

### 2.1 已确定的技术决策

| 决策项 | 结论 | 原因 |
|---|---|---|
| DL 框架 | RL（强化学习）| 硕士论文要求，导师方向 |
| MDP 框架 | Markov Decision Process | 时序决策问题天然适合 |
| 主算法 | PPO（Proximal Policy Optimization）| 离散动作空间，训练稳定 |
| 对比算法 | DQN + A2C + Random Search | 消融实验，论文实验章节 |
| 可解释性 | Decision Log | 每步 state→action→reward，对应税法条款 |
| 仿真时间粒度 | 年（不是月）| 德国税法按年申报，月粒度无税法依据且慢 12 倍 |
| Episode 步数 | 固定上限（15步）+ 早停奖励 | 避免 early stopping trap |
| 起点策略 | 随机起点（Random Reset）| 泛化能力，服务不同用户 |
| 多资产支持 | Phase 1: 1套，Phase 2: 最多3套 | 控制复杂度，论文说明简化假设 |
| 可解释性方法 | Decision Log（非 SHAP/LIME）| 满足德国法律可追溯要求，够用 |
| 税务计算 | 自实现（不使用 gettsim）| gettsim 覆盖面不够，德语文档复杂 |
| 税法参数管理 | 数据驱动（tax_params.json）| 税法变化只改 JSON，代码零修改 |
| 训练硬件 | MacBook Pro M4 Pro 24GB | 本地训练，MPS 加速 |

### 2.2 关键产品约束

- **不允许给出定制化策略推荐**：输出必须是多方案对比表，不能说"选这个"
- **德国法律可解释性**：每个建议必须能追溯到具体税法条款
- **论文要求**：系统必须包含完整的 DL（RL）模型设计，是论文核心贡献

### 2.3 税法规则覆盖范围

已实现的规则（tax_engine.py）：

- §32a EStG：累进 Einkommensteuer（含 Ehegattensplitting，Solidaritätszuschlag）
- §7 Abs.4 EStG：AfA 建筑折旧（旧楼2%，Neubau post-2023 3%，Denkmal 9%/7%）
- §6 Abs.1 Nr.1a EStG：15% 资本化规则（前3年大修超15%触发）
- §21 Abs.2 EStG：66%/50% 租金规则（低于市场租金的抵扣削减）
- §23 EStG：投机税（持有<10年按边际税率，≥10年免税）
- §9 EStG：Werbungskosten 汇总（利息、AfA、维修、管理费等）
- GrEStG：各州 Grunderwerbsteuer

---

## 3. 系统架构

### 3.1 整体分层（Phase 1 + Phase 2）

```
┌─────────────────────────────────────────────────────┐
│ 用户输入层                                            │
│  个人财务状态 | 现有房产清单 | 目标/约束              │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│ 状态引擎层（新增）                                    │
│  PersonalState | PropertyState×N | ActionMask        │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│ 仿真引擎层（已有代码升级）                            │
│  tax_engine.py（不变）                               │
│  finance_engine.py（不变）                           │
│  action_engine.py（新增：执行单个动作）              │
│  world_model.py（新增核心：动作→状态转移→reward）    │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│ RL 环境层                                            │
│  env.py（Gymnasium 接口）                            │
│  reward.py（Reward 函数设计）                        │
│  policy_net.py（PPO/DQN/A2C 共用 backbone）         │
│  train.py + evaluate.py                              │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│ 输出层（已有代码，小改动）                            │
│  策略结果表 | Decision Log | report.py               │
└─────────────────────────────────────────────────────┘
```

### 3.2 数据流（单次推理）

```
用户输入参数
    → personal_state.py 构建初始 State
    → env.reset(initial_state)
    → Agent 观测 State（含 ActionMask）
    → Agent 选择 Action（非法动作被屏蔽）
    → env.step(action)
        → action_engine.py 执行动作，更新 State
        → world_model.py 推进一年，调用 tax_engine + finance_engine
        → reward.py 计算即时 reward
        → 返回 (new_state, reward, done, info)
    → 循环 15 步（或提前终止）
    → Episode 结束，记录 Decision Log
    → output_formatter.py 生成结果表一行
```

### 3.3 原有文件的角色变化

| 文件 | 原来角色 | 新角色 |
|---|---|---|
| `simulator.py` | 顶层仿真入口 | 降级为 world_model.py 的内部工具 |
| `property_model.py` | 单资产参数载体 | 拆成 PersonalState + PropertyState |
| `sampler.py` | 生成参数组合 | 改为生成随机初始 State |
| `output_formatter.py` | 输出参数行 | 输出动作序列 + 结果 |
| `decision_log.py` | 记录参数调整 | 记录真实动作序列 + 税法引用 |
| `tax_engine.py` | 不变 | 不变，直接复用 |
| `finance_engine.py` | 不变 | 不变，直接复用 |
| `validators.py` | 参数校验 | 被 action_mask.py 补充（动作前提校验）|

---

## 4. MDP 正式定义

### 4.1 State Space S

每个 time step 的完整财务快照：

```python
@dataclass
class PersonalState:
    # 个人财务
    current_year: int               # 当前模拟年份
    liquid_cash: float              # 可动用现金（欧元）
    annual_income: float            # 年工资收入
    filing_status: str              # "single" | "married"

    # 房产列表（Phase 1: 最多1套，Phase 2: 最多3套）
    properties: list[PropertyState]

    # 累计指标（用于 §23 判断）
    years_elapsed: int              # 从模拟开始已过年数

@dataclass
class PropertyState:
    status: str                     # "none"|"owned_vacant"|"owned_renting"|"sold"
    purchase_year: int
    purchase_price: float
    state: str                      # 德国联邦州
    building_type: str
    current_loan_balance: float
    annual_rate: float
    cumulative_afa: float           # 已累计折旧金额（§23 时需要）
    cumulative_renovation: float    # 已累计维修支出（15%规则检查）
    current_rent_annual: float      # 当前年租金（0 = 未出租）
    market_rent_annual: float       # 市场参考租金
    years_owned: int                # 已持有年数（§23 关键）
```

State 向量化为固定长度的 float 数组，供 Policy Network 消费。Phase 1 约 25 维，Phase 2 约 55 维（支持3套）。

### 4.2 Action Space A

Phase 1（单资产，8个动作类型）：

```python
class ActionType(Enum):
    DO_NOTHING        = 0   # 永远合法
    BUY_PROPERTY      = 1   # 前提：liquid_cash >= 首付 + Nebenkosten
    START_RENTING     = 2   # 前提：有房产且 status == "owned_vacant"
    ADJUST_RENT       = 3   # 前提：status == "owned_renting"
    DO_RENOVATION     = 4   # 前提：有房产（任意状态）
    REFINANCE         = 5   # 前提：loan_balance > 0
    EXTRA_REPAYMENT   = 6   # 前提：loan_balance > 0
    SELL_PROPERTY     = 7   # 前提：status in ["owned_vacant", "owned_renting"]
```

每个动作带参数（档位化，对应 V1.0 规范的离散分级）：

```python
@dataclass
class Action:
    action_type: ActionType
    params: dict    # e.g. {"ltv": 0.8, "rate": 0.035} for BUY_PROPERTY
```

### 4.3 Action Mask

在每个 State 下，`action_mask.py` 输出一个 boolean 数组，形状为 `[n_actions]`。PPO/DQN 在 softmax 之前将非法动作设为 -inf，确保 Agent 永远不会选非法动作。

### 4.4 Reward Function R(s, a, s')

```python
# 即时 reward（每年末）
r_annual = net_cashflow_after_tax(s')       # 正值 = 现金流为正

# FLAG 惩罚（触发时一次性扣分）
r_flags = (
    - λ1 * FLAG_15_PERCENT_HIT              # 建议 λ1 = 0.3 × 年租金
    - λ2 * FLAG_RENT_TOO_LOW                # 建议 λ2 = 0.1 × 年租金
    - λ3 * FLAG_TAX_WASTE                   # 建议 λ3 = 0.2 × 年租金
    - λ4 * FLAG_NEGATIVE_CASHFLOW           # 建议 λ4 = 0.1 × 年租金
)

# 终止 reward（Episode 结束时）
r_exit = net_exit_proceeds - speculation_tax    # 卖出时
r_hold = cumulative_equity_built                # 持有到期时（未卖出）

# 总 reward（每步）
R = r_annual + r_flags

# Episode 结束时额外加
R_terminal = r_exit 或 r_hold
```

**重要设计原则**：λ 值的调优是 Reward Shaping 的核心，需要实验。从简单开始（只用 r_annual + r_exit），确认 Agent 能学到基本合理的策略后，再加 FLAG 惩罚。

### 4.5 Transition Function T(s, a)

确定性转移：给定 State s 和 Action a，新 State s' 是唯一确定的（德国税法是确定性规则，无随机性）。这是 RL 在此问题上的优势——环境稳定，Agent 专注于学习 reward landscape 的结构。

### 4.6 Episode 定义

```
起点：random reset（随机生成合法的初始 PersonalState）
终点：max_steps = 15（对应15年）
      或 Agent 选择 SELL_PROPERTY 后再 DO_NOTHING 5步
早停：连续 5步 reward 变化 < 0.01 × annual_income，给收敛奖励并结束
折扣因子 γ：0.95（推荐起点，论文中做敏感性分析）
```

---

## 5. 完整文件清单与优先级

### 图例

- P0：没有它系统无法运行
- P1：没有它功能不完整
- P2：质量保障，可后补

### 5.1 已完成文件（层一，直接复用）

| 文件 | 状态 | 作用 | 复用方式 |
|---|---|---|---|
| `tax_params.json` | ✅ 完成 | 2025/2026 税法参数，数据驱动 | 直接复用，按年追加新数据 |
| `tax_engine.py` | ✅ 完成 | 德国税法计算（7个函数）| 直接复用，world_model 调用 |
| `finance_engine.py` | ✅ 完成 | 贷款摊销、现金流、IRR | 直接复用 |
| `validators.py` | ✅ 完成 | 参数合法性校验 | 被 action_mask 补充 |
| `config.json` | ✅ 完成 | 全局业务常量 | 直接复用 |
| `rule_registry.json` | ✅ 完成 | 税法条款注册表 | Decision Log 查询 |
| `tests/test_tax_engine.py` | ✅ 完成 | 税法规则边界测试 | 直接复用 |
| `tests/test_finance_engine.py` | ✅ 完成 | 金融计算精度测试 | 直接复用 |
| `demo.py` | ✅ 完成 | 三场景演示脚本 | 参考，不直接复用 |

**已完成文件的关键测试结果**（44个测试全部通过）：
- Grundfreibetrag 2025: €12,096 / 2026: €12,348
- AfA 旧楼 2%，Neubau post-2023 3%
- 15% 规则：前3年 > 购价15% 触发资本化
- §23：持有 < 10年征税，≥ 10年免税（9年 vs 11年 IRR 差 2.70%）
- 年利率 3.5%，300k贷款，2% Sondertilgung：12年还清，节省利息 €20,778

### 5.2 Phase 1 待开发文件

#### 状态引擎层

| 文件 | 优先级 | 作用 | 输入 | 输出 |
|---|---|---|---|---|
| `personal_state.py` | P0 | PersonalState + PropertyState 数据类定义，State 向量化方法 | 用户输入 dict | PersonalState 对象 + float 数组 |
| `action_space.py` | P0 | ActionType 枚举，Action 数据类，动作参数的离散档位定义 | — | Action 对象 |
| `action_mask.py` | P0 | 在每个 State 下计算合法动作 boolean 向量 | PersonalState | np.array[bool]，形状 [n_actions] |

#### 仿真引擎层

| 文件 | 优先级 | 作用 | 输入 | 输出 |
|---|---|---|---|---|
| `action_engine.py` | P0 | 执行单个动作，更新 PersonalState，返回即时财务变化 | PersonalState + Action | (new_PersonalState, immediate_financials: dict) |
| `world_model.py` | P0 | 核心调度器：动作 → 调用 tax+finance → 推进一年 → 返回新 State + reward 分量 | PersonalState + Action | (new_PersonalState, reward_components: dict) |

#### RL 环境层

| 文件 | 优先级 | 作用 | 输入 | 输出 |
|---|---|---|---|---|
| `env.py` | P0 | Gymnasium 标准接口（reset/step/render），集成 action_mask | initial_state 或 None（随机）| observation, reward, done, truncated, info |
| `reward.py` | P0 | Reward 函数：即时现金流 + FLAG 惩罚 + 终止奖励，λ 权重可配置 | world_model 输出 | float（scalar reward）+ 分项 dict |
| `policy_net.py` | P1 | Actor-Critic backbone，支持 PPO/DQN/A2C 三种训练算法切换 | observation tensor | action logits + value estimate |
| `train.py` | P1 | 训练循环，CLI 切换算法，记录训练曲线，保存 checkpoint | env + policy_net + 超参数 | model.pt + training_log.json |
| `evaluate.py` | P1 | 对比实验：PPO vs DQN vs A2C vs Random，输出论文用对比表格 | 4个 model.pt + 测试 State 集 | evaluation_report.json |

#### 输出层（升级）

| 文件 | 优先级 | 作用 | 输入 | 输出 |
|---|---|---|---|---|
| `decision_log.py` | P1 | 记录完整动作序列 + 对应税法条款 + 量化影响 | Episode 历史 + rule_registry | list[DecisionEntry]，可序列化 |
| `output_formatter.py` | P1 | 升级：输出从"参数行"改为"动作序列 + 财务结果" | Episode 结果 | CSV/JSON 一行 |

#### 基础设施

| 文件 | 优先级 | 作用 |
|---|---|---|
| `tests/test_personal_state.py` | P1 | State 向量化正确性，ActionMask 边界测试 |
| `tests/test_action_engine.py` | P1 | 每个动作的前提检查，State 更新正确性 |
| `tests/test_world_model.py` | P1 | 完整 episode 数字验证 |
| `requirements.txt` | P2 | gymnasium, torch, stable-baselines3, numpy, pandas, pytest |

### 5.3 Phase 2 追加文件

| 文件 | 优先级 | 作用 |
|---|---|---|
| `multi_asset_state.py` | P0 | 扩展 PersonalState 支持最多3套房产，可变长度 State padding |
| `portfolio_reward.py` | P0 | 多资产 Reward：跨资产现金流合并，整体税务合并计算 |
| `scenarios.json` | P1 | 预设对比场景（9年vs11年退出，高LTV vs 低LTV 等）|
| `main.py` | P1 | 升级：从参数CLI改为接收用户完整财务状态的CLI入口 |

---

## 6. 已完成文件详细说明

### 6.1 tax_engine.py — 核心函数列表

```python
class TaxEngine:
    def calc_income_tax(zvE, year, filing_status) -> dict
        # 返回: einkommensteuer, solidaritaetszuschlag, total_tax,
        #        effective_rate, marginal_rate, year, law_ref

    def calc_afa(building_value, movable_value, year_of_purchase,
                 simulation_year, building_type) -> dict
        # 返回: building_afa, movable_afa, total_afa, rate_used

    def check_15pct_rule(renovation_cumulative_net, purchase_price_net,
                         years_since_purchase, simulation_year) -> dict
        # 返回: triggered(bool), limit_amount, impact, law_ref

    def check_rent_rule(actual_rent_annual, market_rent_annual,
                        simulation_year) -> dict
        # 返回: deduction_ratio(1.0/0.5/0.0), zone, flag_rent_too_low

    def calc_speculation_tax(sale_price, original_purchase_price,
                             cumulative_afa_claimed, holding_years,
                             annual_income_in_exit_year, filing_status,
                             simulation_year) -> dict
        # 返回: speculation_tax, taxable_gain, tax_free(bool)

    def calc_werbungskosten(interest_paid, afa_total, renovation_deductible,
                            management_costs, insurance_costs, other_costs,
                            deduction_ratio, simulation_year) -> dict
        # 返回: total_deductible, net_deductible(分项), law_ref

    def calc_grunderwerbsteuer(purchase_price, state, year) -> dict
        # 返回: grunderwerbsteuer, rate, law_ref
```

**重要**：TaxEngine 的所有数字来自 tax_params.json，代码零硬编码。追加新年份税法只需在 JSON 里添加新的年份键。

### 6.2 finance_engine.py — 核心函数列表

```python
def build_amortization_schedule(principal, annual_rate, holding_years,
                                sondertilgung_rate, refi_year, refi_rate,
                                purchase_year) -> AmortizationSchedule
    # 返回 AmortizationSchedule，含每年的 YearlyLoanState

def calc_purchase_costs(purchase_price, grunderwerbsteuer_rate,
                        notar_rate, makler_rate, include_makler) -> dict

def calc_equity_and_loan(purchase_price, total_nebenkosten,
                         equity_amount) -> dict
    # 返回: loan_amount, ltv, equity_ratio

def calc_annual_cashflow(rental_income_gross, non_deductible_costs,
                         loan_payment, tax_refund, sondertilgung) -> dict
    # 返回: net_cashflow, flag_negative_cashflow

def calc_exit_proceeds(sale_price, remaining_loan_balance,
                       makler_sell_rate, notar_sell_rate,
                       speculation_tax) -> dict
    # 返回: net_proceeds, gross_proceeds, selling_costs

def calc_irr(cashflows: list[float]) -> dict
    # Newton-Raphson，返回: irr, converged

def calc_npv(cashflows: list[float], discount_rate: float) -> dict
```

### 6.3 tax_params.json 结构

```json
{
  "2025": {
    "einkommensteuer": {
      "grundfreibetrag": 12096,
      "brackets": [...],
      "splitting_divisor": 2
    },
    "solidaritaetszuschlag": {"rate": 0.055, "freigrenze_single": 18130, ...},
    "afa": {"gebaeude": {"standard_rate": 0.02, "neubau_post_2023_rate": 0.03}, ...},
    "thresholds": {
      "renovation": {"window_years": 3, "limit_rate": 0.15},
      "rent": {"full_deduction_threshold": 0.66, "half_deduction_threshold": 0.50},
      "speculation_tax": {"free_after_years": 10}
    },
    "grunderwerbsteuer": {"by_state": {"Bayern": 0.035, "Berlin": 0.06, ...}}
  },
  "2026": {
    "einkommensteuer": {"grundfreibetrag": 12348, "brackets": [...]}
    // 只写变化的字段，其余自动继承 2025
  }
}
```

---

## 7. 待开发文件详细说明

### 7.1 personal_state.py

**职责**：定义 PersonalState 和 PropertyState 数据类，以及 State 向量化方法（供 Policy Network 消费）。

**关键设计点**：
- `PropertyState.status` 是状态机的核心，取值 `none | owned_vacant | owned_renting | sold`
- Phase 1 固定1套房产，Phase 2 扩展到最多3套（不足的用零向量 padding）
- 向量化时离散字段（state, building_type, filing_status）用 one-hot encoding

**示例接口**：
```python
@dataclass
class PersonalState:
    current_year: int
    liquid_cash: float
    annual_income: float
    filing_status: str          # "single" | "married"
    properties: list[PropertyState]
    years_elapsed: int

    def to_observation(self) -> np.ndarray:
        """返回固定长度的 float 数组，供 Policy Network 输入"""

    @classmethod
    def random(cls, config: dict) -> "PersonalState":
        """生成随机合法的初始状态，用于 Random Reset"""
```

### 7.2 action_space.py

**职责**：定义所有动作类型及其参数选项（离散档位）。

**动作参数档位（Phase 1）**：
```python
BUY_PROPERTY 参数：
    ltv: [0.80, 0.90]
    annual_rate: [0.025, 0.035, 0.040, 0.050]
    sondertilgung_rate: [0.0, 0.01, 0.02]
    land_ratio: [0.20, 0.30, 0.40]
    asset_split: [0, 10000, 20000]

START_RENTING 参数：
    rental_ratio: [0.66, 0.80, 0.95, 1.0]

DO_RENOVATION 参数：
    amount: [10000, 20000, 40000, 65000]

SELL_PROPERTY 参数：
    sale_price_multiplier: [0.9, 1.0, 1.1, 1.2, 1.3]
    （相对于 purchase_price 的倍数，用户可覆盖）
```

**总动作数**（Phase 1）：约 40-50 个离散动作（各参数档位的笛卡尔积子集）

### 7.3 action_mask.py

**职责**：在每个 State 下计算合法动作的 boolean 向量。这是解决"没买房不能出租"问题的核心文件。

**前提依赖规则**：
```python
def compute_mask(state: PersonalState) -> np.ndarray:
    mask = np.zeros(N_ACTIONS, dtype=bool)

    mask[DO_NOTHING] = True  # 永远合法

    # 买房：需要有足够现金
    if state.liquid_cash >= MIN_EQUITY_FOR_BUY:
        mask[BUY_PROPERTY_variants] = True

    # 出租：需要有房且空置
    if any(p.status == "owned_vacant" for p in state.properties):
        mask[START_RENTING_variants] = True

    # 翻修：需要有房（任意状态）
    if any(p.status in ["owned_vacant","owned_renting"] for p in state.properties):
        mask[DO_RENOVATION_variants] = True

    # 再融资：需要有贷款
    if any(p.current_loan_balance > 0 for p in state.properties):
        mask[REFINANCE_variants] = True

    # 卖房：需要有房
    if any(p.status in ["owned_vacant","owned_renting"] for p in state.properties):
        mask[SELL_PROPERTY_variants] = True

    return mask
```

**交付标准**：`tests/test_action_mask.py` 覆盖所有前提依赖的边界情况。

### 7.4 action_engine.py

**职责**：执行单个动作，更新 PersonalState，返回即时财务变化（不含税务计算，税务由 world_model 调用 tax_engine 处理）。

**示例接口**：
```python
def execute_buy(state: PersonalState, action_params: dict) -> tuple[PersonalState, dict]:
    """
    执行买房动作：
    - 扣除首付 + Nebenkosten
    - 新增 PropertyState（status = "owned_vacant"）
    - 建立贷款摊销计划
    返回：新 State + {cash_out, loan_created, nebenkosten}
    """

def execute_start_renting(state, params) -> tuple[PersonalState, dict]:
def execute_renovation(state, params) -> tuple[PersonalState, dict]:
def execute_sell(state, params, tax_engine) -> tuple[PersonalState, dict]:
    """卖房需要调用 tax_engine 计算投机税"""
```

### 7.5 world_model.py

**职责**：整个仿真的核心调度器。接收当前 State 和 Action，调用 action_engine + tax_engine + finance_engine，推进一年，返回新 State 和 reward 分量。

**每年推进逻辑**：
```
1. 执行动作（action_engine）
2. 推进贷款一年（finance_engine.amortization step）
3. 计算当年税务（tax_engine）
4. 检测 FLAGS（flag_system）
5. 计算年度现金流（finance_engine.calc_annual_cashflow）
6. 更新 PropertyState 的所有 cumulative 字段
7. 返回 (new_state, reward_components)
```

### 7.6 env.py

**职责**：Gymnasium 标准接口，集成 action_mask。

```python
class RealEstateEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "json"]}

    def __init__(self, config: dict, initial_state: PersonalState = None):
        self.observation_space = spaces.Box(...)    # 固定维度 float 数组
        self.action_space = spaces.Discrete(N_ACTIONS)

    def reset(self, seed=None, options=None):
        # options["initial_state"] 可以传入真实用户状态（场景B）
        # 否则随机生成（场景A）
        return observation, info

    def step(self, action: int):
        # 检查 action_mask（非法动作应该已被 Policy 屏蔽）
        # 调用 world_model.step()
        return observation, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        # Stable-Baselines3 MaskablePPO 的接口
        return compute_mask(self.current_state)
```

**注意**：使用 `sb3-contrib` 的 `MaskablePPO`，而不是标准 PPO，因为需要 action masking 支持。

### 7.7 reward.py

**职责**：Reward 函数，λ 权重从 config.json 读取（不硬编码）。

**推荐的 λ 初始值（需要实验调优）**：
```python
LAMBDA = {
    "FLAG_15_PERCENT_HIT":    0.3,   # × annual_income
    "FLAG_RENT_TOO_LOW":      0.1,
    "FLAG_TAX_WASTE":         0.2,
    "FLAG_NEGATIVE_CASHFLOW": 0.1,
}
```

**Reward Shaping 开发顺序**（重要！）：
1. 第一阶段：只用 `net_cashflow + exit_proceeds`，验证 Agent 能学到基本策略
2. 第二阶段：加入 FLAG 惩罚
3. 第三阶段：加入收敛奖励和早停逻辑

### 7.8 policy_net.py

**职责**：Actor-Critic backbone，支持三种算法。

```python
class RealEstateActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dims: list[int]):
        # 推荐起点：[256, 256] 两层 MLP
        # 原因：State 维度小（25-55），不需要 Transformer
        # 论文可以做消融：MLP vs Transformer backbone

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 返回：action_logits, value_estimate
```

**训练算法**：使用 `stable-baselines3` 和 `sb3-contrib`：
```python
# PPO（主模型）
from sb3_contrib import MaskablePPO
model = MaskablePPO("MlpPolicy", env, verbose=1, device="mps")

# DQN（对比模型1）
from stable_baselines3 import DQN
# 注：DQN 不原生支持 action masking，需要自定义 wrapper

# A2C（对比模型2）
from stable_baselines3 import A2C
```

### 7.9 decision_log.py（升级版）

**每步记录格式**：
```json
{
  "year": 2026,
  "year_index": 2,
  "action": "DO_RENOVATION",
  "action_params": {"amount": 20000},
  "state_before": {"liquid_cash": 45000, "property_status": "owned_renting"},
  "state_after": {"liquid_cash": 25000},
  "reward_components": {
    "net_cashflow": -3200,
    "flag_penalties": 0,
    "flag_15pct_triggered": false
  },
  "rule_anchor": {
    "law": "§6 Abs.1 Nr.1a EStG",
    "description": "Yr2维修 €20,000 低于15%限额 €60,000，可直接作为Werbungskosten抵扣",
    "quantified_impact": "节省税款约 €8,400（按42%边际税率）"
  }
}
```

---

## 8. 风险清单

### 风险一：Reward Shaping 难度 ⚠️ 高风险

**问题**：Reward 函数设计不好，Agent 会学出奇怪的策略（永远 do_nothing，或频繁买卖套利）。

**缓解方案**：
- 严格按照 Section 7.7 的"三阶段"开发顺序
- 每阶段用"Sanity Check 场景"验证：Agent 应该在 §23 10年门槛后才卖房（可验证）
- 保存每隔 10 万步的 checkpoint，出问题可以回滚

**时间影响**：Reward 调优可能占总开发时间的 30%，提前预留。

### 风险二：Action Mask 边界条件 ⚠️ 中等风险

**问题**："现金不够付首付"的判断依赖于"用户选的 LTV"，而 LTV 是动作参数的一部分，形成循环依赖。

**缓解方案**：
- action_mask 只做保守估计（假设最高 LTV 方案，看现金够不够）
- 真正的合法性检查在 action_engine 里做（执行时再精确校验）
- 测试文件必须覆盖"现金刚好够/差一点"的边界情况

### 风险三：State 维度随房产数量变化 ⚠️ 中等风险

**问题**：标准 RL 库假设 observation_space 维度固定，但支持 N 套房产时 State 长度变化。

**缓解方案**：
- Phase 1 固定 1 套，observation 维度固定
- Phase 2 用 padding（不足 3 套的用零向量填充）
- 论文中说明 N≤3 的假设，留 future work

### 风险四：训练稳定性 ⚠️ 中等风险

**问题**：RL 训练天然不稳定，同样代码不同随机种子结果差异大。

**缓解方案**：
- 每个实验跑 5 个随机种子，报告均值±标准差
- 使用 `stable-baselines3` 的成熟实现，不自己写训练循环
- 先在小规模（10 万 steps）验证收敛，再扩展到 100 万

### 风险五：M4 Pro MPS 兼容性 ⚠️ 低风险

**问题**：部分 PyTorch 操作在 MPS 上有 bug（MPS 相对 CUDA 较新）。

**缓解方案**：
- `device = "mps" if torch.backends.mps.is_available() else "cpu"`
- 如果 MPS 出问题，用 CPU 训练（慢 3-5 倍，但 100 万步 CPU 也在 5-10 小时内）
- `stable-baselines3` v2.x 已有官方 MPS 支持

### 风险六：论文范围 vs 实现范围的落差 ⚠️ 低风险但需管理

**问题**：论文描述"支持任意套数"，MVP 只做 1-3 套。

**缓解方案**：在论文 Section 3（方法论）明确写出 N≤3 的简化假设和工程原因，不试图掩盖。Reviewer 接受合理的简化，但不接受不解释的简化。

---

## 9. 开发阶段规划

### Phase 1（单套房产，论文 MVP）

**目标**：完成可以训练和对比 PPO/DQN/A2C/Random 的完整系统，支持场景 A（买房）。

#### 阶段一（第1-2周）：状态引擎

交付物：
- `personal_state.py` — PersonalState + PropertyState + 向量化
- `action_space.py` — ActionType 枚举 + 参数档位
- `action_mask.py` — 合法动作 boolean 向量
- `tests/test_personal_state.py`
- `tests/test_action_mask.py`

验收标准：所有测试通过；ActionMask 在"未买房"状态下正确屏蔽出租/翻修/卖房动作。

#### 阶段二（第3-4周）：仿真引擎

交付物：
- `action_engine.py` — 8种动作的执行逻辑
- `world_model.py` — 核心调度器
- `tests/test_action_engine.py`
- `tests/test_world_model.py`

验收标准：手动构造一个"买房→出租→翻修→卖房"的动作序列，验证最终 IRR 和现有 demo.py 的结果在 0.1% 以内吻合。

#### 阶段三（第5周）：RL 环境

交付物：
- `env.py` — Gymnasium 接口
- `reward.py` — 第一阶段只用现金流 reward
- `tests/test_world_model.py` 补充 episode 测试

验收标准：`env.step()` 和 `env.reset()` 通过 `gymnasium.utils.env_checker`；Random Agent 在环境里能跑完整 episode 不报错。

#### 阶段四（第6周）：训练

交付物：
- `policy_net.py`
- `train.py`（支持 --algo ppo/dqn/a2c）

验收标准：PPO 在 10 万 steps 内显示 reward 有上升趋势（不需要收敛，只需要方向对）。

#### 阶段五（第7-8周）：实验与输出

交付物：
- `evaluate.py`（4模型对比）
- `decision_log.py`（升级版）
- `output_formatter.py`（升级版）
- 完整实验报告数据

验收标准：evaluate.py 输出论文 Table 3 格式的对比数据；Decision Log 包含每步动作的税法引用。

### Phase 2（多套房产，完整产品）

**目标**：支持最多3套房产，场景 B（已有房产），完整的再融资和 Sondertilgung 动作。

#### 阶段六（第9-10周）：多资产扩展

交付物：
- `multi_asset_state.py` — 扩展 PersonalState 支持3套，padding 逻辑
- `portfolio_reward.py` — 跨资产现金流合并
- `scenarios.json` — 预设对比场景
- 更新 `action_mask.py` 支持多资产

#### 阶段七（第11-12周）：产品化

交付物：
- `main.py`（升级为用户完整财务状态入口）
- `report.py`（人类可读报告）
- 完整的 requirements.txt 和 README.md

---

## 10. 训练环境说明

**硬件**：MacBook Pro M4 Pro，24GB 统一内存

**训练时长估算**（Phase 1 单套房产）：

| 模型 | 100 万 steps 估算 | 备注 |
|---|---|---|
| PPO（MPS）| ~1.5 小时 | MPS 加速约 2× |
| DQN（MPS）| ~2 小时 | Experience Replay 额外开销 |
| A2C（MPS）| ~1.3 小时 | 最轻量 |
| 全部跑完 | ~5 小时 | 可以睡前开始 |
| 5个种子 × 3模型 | ~25 小时 | 论文完整实验，分两晚 |

**依赖安装**：
```bash
pip install torch torchvision torchaudio  # M4 原生支持
pip install gymnasium
pip install stable-baselines3
pip install sb3-contrib  # MaskablePPO
pip install numpy pandas pytest
```

**训练命令示例**：
```bash
# 训练 PPO
python train.py --algo ppo --steps 1000000 --seed 42

# 对比实验
python evaluate.py --algos ppo dqn a2c random --n_eval 1000

# 用真实用户数据推理（场景B）
python main.py --state user_state.json --algo ppo --model checkpoints/ppo_best.pt
```

---

## 11. 论文定位

**论文题目候选**：
> "A Markov Decision Process Framework for Real Estate Investment Strategy Optimization under German Tax Constraints"

**核心贡献**：
1. 将德国税法约束形式化为 MDP 的 Reward 函数和 Action Mask
2. Decision Log 作为 RL 策略的可解释性机制，满足法律可追溯要求
3. PPO vs DQN vs A2C 在确定性离散动作空间下的系统对比

**实验章节结构**：
- Table 1：各算法的 sample efficiency（相同 simulator 调用次数下的最高 IRR）
- Table 2：FLAG 触发率（RL 是否学会规避违规策略）
- Table 3：§23 cliff effect 验证（Agent 是否学会在10年后卖房）
- Figure 1：训练曲线（reward 随 steps 的收敛情况）
- Figure 2：Decision Log 示例（一条完整的动作序列及税法引用）

**学术定位**：
- 相关工作：Real estate portfolio optimization, RL for financial planning, Tax-aware investment strategies
- 区别点：现有工作不处理德国税法的离散规则约束（15%规则、§23等），本文首次将其形式化为 MDP

---

## 附录：关键数字参考

| 指标 | 数值 | 来源 |
|---|---|---|
| Grundfreibetrag 2025 | €12,096 | §32a EStG |
| Grundfreibetrag 2026 | €12,348 | §32a EStG |
| AfA 旧楼 | 2%/年 | §7 Abs.4 EStG |
| AfA Neubau post-2023 | 3%/年 | §7 Abs.4 EStG（2023新规）|
| 15% 规则窗口 | 前3年 | §6 Abs.1 Nr.1a EStG |
| 15% 限额 | 购价的15%（不含增值税）| §6 Abs.1 Nr.1a EStG |
| 租金抵扣上限 | 66% 市场租金 | §21 Abs.2 EStG |
| §23 免税年限 | 持有 ≥ 10年 | §23 Abs.1 S.1 Nr.1 EStG |
| 9年 vs 11年 IRR 差距 | +2.70% | demo.py 验证结果 |
| Solidaritätszuschlag | 5.5% | SolzG |
| Bayern GrESt | 3.5% | Landesrecht |
| Berlin GrESt | 6.0% | Landesrecht |
| 典型 Makler | 3.57% | 含增值税，买卖双方各付一半（2020新规）|

---

*文档结束。下一个对话框从"Phase 1 阶段一：personal_state.py"开始。*

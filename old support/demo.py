"""
demo.py
-------
可直接运行的演示脚本。

用法:
    python demo.py

展示三个场景:
  1. 标准出租 10年持有
  2. 15% 规则触发 vs 规避对比
  3. §23 投机税: 第9年 vs 第11年卖出对比

所有文件需在同一目录下:
    tax_params.json
    tax_engine.py
    finance_engine.py
    property_model.py
    simulator.py
    demo.py
"""

from simulator import run_simulation


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def divider(title: str):
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)


def print_flags(flags: dict):
    print()
    print("  FLAGS:")
    for name, data in flags.items():
        icon = "🔴" if data["triggered"] else "🟢"
        print(f"    {icon}  {name}")
        if data["triggered"]:
            print(f"         → {data['detail']}")


def print_yearly_table(snapshots):
    print()
    header = (
        f"  {'Yr':>3}  {'Rent':>8}  {'Interest':>9}  "
        f"{'AfA':>8}  {'WK Total':>9}  {'VuV':>9}  "
        f"{'TaxΔ':>8}  {'NetCF':>9}  {'Balance':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for s in snapshots:
        neg = " ◄" if s.cashflow["flag_negative_cashflow"] else ""
        print(
            f"  {s.year_index:>3}  "
            f"{s.actual_rent_annual:>8,.0f}  "
            f"{s.loan_state.annual_interest:>9,.0f}  "
            f"{s.afa_result['total_afa']:>8,.0f}  "
            f"{s.werbungskosten['total_deductible']:>9,.0f}  "
            f"{s.verpachtung_income:>9,.0f}  "
            f"{s.tax_delta:>8,.0f}  "
            f"{s.cashflow['net_cashflow']:>9,.0f}  "
            f"{s.loan_state.closing_balance:>10,.0f}"
            f"{neg}"
        )


def print_summary(result, label: str = ""):
    er = result.exit_result
    print(f"\n  {'IRR':<28}: {result.irr():>8.2%}")
    print(f"  {'NPV @3%':<28}: €{result.npv():>10,.0f}")
    print(f"  {'Total tax saved':<28}: €{result.total_tax_saved:>10,.0f}")
    print(f"  {'Total interest paid':<28}: €{result.total_interest_paid:>10,.0f}")
    print(f"  {'Total net cashflow (ops)':<28}: €{result.total_net_cashflow:>10,.0f}")
    print(f"  {'Sale price':<28}: €{er.sale_price:>10,.0f}")
    print(f"  {'Remaining loan at exit':<28}: €{er.remaining_loan_balance:>10,.0f}")
    print(f"  {'Speculation tax':<28}: €{er.speculation_tax_result['speculation_tax']:>10,.0f}",
          "(EXEMPT)" if er.speculation_tax_result["tax_free"] else "")
    print(f"  {'Net exit proceeds':<28}: €{er.exit_proceeds['net_proceeds']:>10,.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# Base parameter set (shared across scenarios)
# ─────────────────────────────────────────────────────────────────────────────

BASE = dict(
    purchase_price    = 400_000,
    state             = "Bayern",
    purchase_year     = 2024,
    personal_income   = 80_000,
    filing_status     = "single",
    market_rent_annual= 18_000,
    building_type     = "standard",
    usage             = "full_rental",
    rental_ratio      = 0.95,
    renovation_year   = 5,
    renovation_amount = 20_000,
    asset_split       = 10_000,
    land_ratio        = 0.25,
    equity_amount     = 120_000,
    annual_rate       = 0.035,
    sondertilgung_rate= 0.02,
    refi_year         = None,
    refi_rate         = None,
    holding_years     = 10,
)

SALE_PRICE = 480_000   # assumed exit price across all scenarios


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1 — Standard case
# ─────────────────────────────────────────────────────────────────────────────

divider("Scenario 1 — Standard: 10yr hold, full rental, 2% Sondertilgung")

print(f"""
  Property  : €{BASE['purchase_price']:,.0f}  |  State: {BASE['state']}
  Equity    : €{BASE['equity_amount']:,.0f}  |  Loan rate: {BASE['annual_rate']:.1%}
  Rent      : {BASE['rental_ratio']:.0%} of market (€{BASE['market_rent_annual']:,.0f}/yr)
  Sonder    : {BASE['sondertilgung_rate']:.0%}/yr  |  Reno: Yr{BASE['renovation_year']} €{BASE['renovation_amount']:,.0f}
  Hold      : {BASE['holding_years']} years  |  Sale: €{SALE_PRICE:,.0f}
""")

r1 = run_simulation(BASE, sale_price=SALE_PRICE)
print_summary(r1)
print_flags(r1.flags)
print_yearly_table(r1.snapshots)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2 — 15% rule: triggered vs avoided
# ─────────────────────────────────────────────────────────────────────────────

divider("Scenario 2 — §6 15% Rule: Yr2 big reno (triggered) vs Yr4 (safe)")

# Triggered: large renovation in year 2 (within 3yr window, >15% of price)
params_triggered = {**BASE,
    "renovation_year": 2,
    "renovation_amount": 65_000,   # 65k > 15% × 400k = 60k  → capitalised
    "sondertilgung_rate": 0.0,
    "holding_years": 10,
}

# Avoided: same budget but in year 4 (outside 3yr window)
params_avoided = {**BASE,
    "renovation_year": 4,
    "renovation_amount": 65_000,   # same amount, year 4 → fully deductible
    "sondertilgung_rate": 0.0,
    "holding_years": 10,
}

r_trig  = run_simulation(params_triggered, sale_price=SALE_PRICE)
r_avoid = run_simulation(params_avoided,   sale_price=SALE_PRICE)

print(f"\n  Renovation: €{65_000:,.0f}  |  Triggered: Yr2  |  Avoided: Yr4\n")

cols = ["IRR", "NPV @3%", "Total tax saved", "FLAG_15_PERCENT_HIT"]
print(f"  {'Metric':<28}  {'Triggered (Yr2)':>18}  {'Avoided (Yr4)':>16}  {'Δ':>10}")
print("  " + "-" * 76)

irr_diff = r_avoid.irr() - r_trig.irr()
npv_diff = r_avoid.npv() - r_trig.npv()
tax_diff = r_avoid.total_tax_saved - r_trig.total_tax_saved

print(f"  {'IRR':<28}  {r_trig.irr():>17.2%}  {r_avoid.irr():>15.2%}  {irr_diff:>+9.2%}")
print(f"  {'NPV @3%':<28}  €{r_trig.npv():>16,.0f}  €{r_avoid.npv():>14,.0f}  €{npv_diff:>+8,.0f}")
print(f"  {'Total tax saved':<28}  €{r_trig.total_tax_saved:>16,.0f}  €{r_avoid.total_tax_saved:>14,.0f}  €{tax_diff:>+8,.0f}")
print(f"  {'FLAG_15_PERCENT_HIT':<28}  {'YES':>18}  {'NO':>16}")

print("\n  15% rule triggered detail:")
print(f"    → {r_trig.flags['FLAG_15_PERCENT_HIT']['detail']}")
print(f"\n  When avoided (Yr4), €65,000 fully deductible as Werbungskosten in that year.")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3 — §23 Exit timing: year 9 vs year 11
# ─────────────────────────────────────────────────────────────────────────────

divider("Scenario 3 — §23 Speculation Tax: Sell at Yr9 vs Yr11")

params_9yr  = {**BASE, "holding_years": 9,  "sondertilgung_rate": 0.01}
params_11yr = {**BASE, "holding_years": 11, "sondertilgung_rate": 0.01}

SALE_HIGH = 500_000   # slightly higher sale price to make gain meaningful

r9  = run_simulation(params_9yr,  sale_price=SALE_HIGH)
r11 = run_simulation(params_11yr, sale_price=SALE_HIGH)

print(f"\n  Same property, same strategy — only exit year changes.")
print(f"  Sale price: €{SALE_HIGH:,.0f}  |  §23 threshold: 10 years\n")

print(f"  {'Metric':<30}  {'Exit Yr 9':>14}  {'Exit Yr 11':>14}  {'Δ':>10}")
print("  " + "-" * 72)

metrics = [
    ("IRR",            f"{r9.irr():.2%}",           f"{r11.irr():.2%}",
     f"{r11.irr()-r9.irr():+.2%}"),
    ("NPV @3%",        f"€{r9.npv():,.0f}",         f"€{r11.npv():,.0f}",
     f"€{r11.npv()-r9.npv():+,.0f}"),
    ("Speculation tax",f"€{r9.exit_result.speculation_tax_result['speculation_tax']:,.0f}",
     f"€{r11.exit_result.speculation_tax_result['speculation_tax']:,.0f}", ""),
    ("Tax-free exit",  str(r9.exit_result.speculation_tax_result["tax_free"]),
     str(r11.exit_result.speculation_tax_result["tax_free"]), ""),
    ("Net exit proceeds",
     f"€{r9.exit_result.exit_proceeds['net_proceeds']:,.0f}",
     f"€{r11.exit_result.exit_proceeds['net_proceeds']:,.0f}",
     f"€{r11.exit_result.exit_proceeds['net_proceeds']-r9.exit_result.exit_proceeds['net_proceeds']:+,.0f}"),
]

for label, v9, v11, delta in metrics:
    print(f"  {label:<30}  {v9:>14}  {v11:>14}  {delta:>10}")

spec_tax = r9.exit_result.speculation_tax_result["speculation_tax"]
irr_gap  = r11.irr() - r9.irr()
print(f"""
  Interpretation:
    Waiting 2 extra years eliminates €{spec_tax:,.0f} in speculation tax.
    IRR improvement: {irr_gap:+.2%}
    This is the §23 "cliff effect" — a core insight for the RL reward design.
""")

print("=" * 64)
print("  Demo complete. All three scenarios ran successfully.")
print("=" * 64)

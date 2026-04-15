"""
output_formatter.py
-------------------
Converts a SimulationResult into one flat dict (= one row in the results table).

The results table is what the user sees and filters.
Each column is either:
  - An input dimension (what strategy was used)
  - An output metric (what the simulation produced)
  - A FLAG (risk indicator, user can filter on these)
  - A yearly detail (cashflow per year, for drill-down)

Design:
  - No business logic here — only reshaping and formatting.
  - Column names are stable (defined in COLUMN_REGISTRY below).
  - output_formatter also writes results_table.csv and results_table.json.
"""

from __future__ import annotations
import csv
import json
import math
from pathlib import Path
from dataclasses import dataclass
from simulator import SimulationResult


# ---------------------------------------------------------------------------
# Column registry
# ---------------------------------------------------------------------------
# Each entry: (column_name, label, visible_to_user, filterable)
# visible=False means it's in the JSON but not the default CSV view.

COLUMNS = [
    # ── Input dimensions ──────────────────────────────────────────────────
    ("purchase_price",      "Purchase Price (€)",          True,  True),
    ("state",               "State",                       True,  True),
    ("purchase_year",       "Purchase Year",               True,  True),
    ("personal_income",     "Personal Income (€)",         True,  True),
    ("filing_status",       "Filing Status",               True,  True),
    ("market_rent_annual",  "Market Rent/yr (€)",          True,  True),
    ("building_type",       "Building Type",               True,  True),
    ("usage",               "Usage",                       True,  True),
    ("holding_years",       "Holding Years",               True,  True),
    ("rental_ratio",        "Rental Ratio",                True,  True),
    ("renovation_year",     "Renovation Year",             True,  True),
    ("renovation_amount",   "Renovation Amount (€)",       True,  True),
    ("asset_split",         "Movables Split (€)",          True,  True),
    ("land_ratio",          "Land Ratio",                  True,  True),
    ("equity_amount",       "Equity (€)",                  True,  True),
    ("annual_rate",         "Loan Rate",                   True,  True),
    ("sondertilgung_rate",  "Sondertilgung Rate",          True,  True),
    ("refi_year",           "Refi Year",                   True,  True),
    ("refi_rate",           "Refi Rate",                   True,  True),
    ("_group_tag",          "Group Tag",                   True,  True),

    # ── Return metrics ────────────────────────────────────────────────────
    ("irr",                 "IRR",                         True,  True),
    ("npv",                 "NPV @3% (€)",                 True,  True),
    ("irr_converged",       "IRR Converged",               False, False),

    # ── Aggregates ────────────────────────────────────────────────────────
    ("total_tax_saved",     "Total Tax Saved (€)",         True,  True),
    ("total_interest_paid", "Total Interest Paid (€)",     True,  True),
    ("total_net_cashflow",  "Total Net Cashflow (€)",      True,  True),

    # ── Exit ──────────────────────────────────────────────────────────────
    ("sale_price",          "Sale Price (€)",              True,  True),
    ("remaining_loan_balance", "Remaining Loan (€)",       True,  True),
    ("speculation_tax",     "Speculation Tax (€)",         True,  True),
    ("tax_free_exit",       "Tax-Free Exit",               True,  True),
    ("net_exit_proceeds",   "Net Exit Proceeds (€)",       True,  True),
    ("gross_exit_proceeds", "Gross Exit Proceeds (€)",     True,  True),
    ("selling_costs",       "Selling Costs (€)",           True,  True),

    # ── FLAGS ─────────────────────────────────────────────────────────────
    ("FLAG_15_PERCENT_HIT",          "⚠ 15% Rule Hit",       True,  True),
    ("FLAG_15_PERCENT_HIT_year",     "15% Rule Hit Year",     True,  True),
    ("FLAG_15_PERCENT_HIT_detail",   "15% Rule Detail",       False, False),
    ("FLAG_RENT_TOO_LOW",            "⚠ Rent Too Low",        True,  True),
    ("FLAG_RENT_TOO_LOW_year",       "Rent Low Year",         True,  True),
    ("FLAG_RENT_TOO_LOW_detail",     "Rent Low Detail",       False, False),
    ("FLAG_TAX_WASTE",               "⚠ Tax Waste",           True,  True),
    ("FLAG_TAX_WASTE_year",          "Tax Waste Year",        True,  True),
    ("FLAG_TAX_WASTE_detail",        "Tax Waste Detail",      False, False),
    ("FLAG_NEGATIVE_CASHFLOW",       "⚠ Negative CF",         True,  True),
    ("FLAG_NEGATIVE_CASHFLOW_year",  "Neg CF Year",           True,  True),
    ("FLAG_NEGATIVE_CASHFLOW_detail","Neg CF Detail",         False, False),
    ("any_flag",                     "Any Flag",              True,  True),

    # ── Yearly detail (not in CSV summary; available in JSON drill-down) ──
    ("yearly_snapshots",    "Yearly Detail",               False, False),
]

# Quick lookup: column_name -> (label, visible, filterable)
COLUMN_META = {col[0]: col[1:] for col in COLUMNS}

# CSV visible columns only
CSV_COLUMNS = [col[0] for col in COLUMNS if col[2]]


# ---------------------------------------------------------------------------
# Format one SimulationResult → one flat dict
# ---------------------------------------------------------------------------

def format_row(result: SimulationResult) -> dict:
    """
    Flatten a SimulationResult into one dict.
    Keys match COLUMN_REGISTRY.
    """
    row = {}

    # ── Input params (direct copy from result.params) ──
    for col in [c[0] for c in COLUMNS if c[0] in result.params or
                c[0] == "_group_tag"]:
        row[col] = result.params.get(col)

    # ── Return metrics ──
    irr = result.irr()
    row["irr"]           = None if math.isnan(irr) else round(irr, 6)
    row["npv"]           = result.npv()
    row["irr_converged"] = result.irr_result.get("converged", False)

    # ── Aggregates ──
    row["total_tax_saved"]     = result.total_tax_saved
    row["total_interest_paid"] = result.total_interest_paid
    row["total_net_cashflow"]  = result.total_net_cashflow

    # ── Exit ──
    er = result.exit_result.to_dict()
    row["sale_price"]             = er["sale_price"]
    row["remaining_loan_balance"] = er["remaining_loan_balance"]
    row["speculation_tax"]        = er["speculation_tax"]
    row["tax_free_exit"]          = er["tax_free_exit"]
    row["net_exit_proceeds"]      = er["net_exit_proceeds"]
    row["gross_exit_proceeds"]    = er["gross_exit_proceeds"]
    row["selling_costs"]          = er["selling_costs"]

    # ── FLAGS ──
    for flag_name, flag_data in result.flags.items():
        row[flag_name]                  = flag_data["triggered"]
        row[f"{flag_name}_year"]        = flag_data["year_index"]
        row[f"{flag_name}_detail"]      = flag_data["detail"]
    row["any_flag"] = result.any_flag()

    # ── Yearly detail (for JSON drill-down) ──
    row["yearly_snapshots"] = [s.to_dict() for s in result.snapshots]

    return row


# ---------------------------------------------------------------------------
# Batch formatting and file output
# ---------------------------------------------------------------------------

@dataclass
class ResultsTable:
    rows: list[dict]
    run_id: str
    fixed_inputs: dict

    @property
    def n(self) -> int:
        return len(self.rows)

    def to_csv(self, path: str | Path) -> None:
        """Write visible columns to CSV. IRR formatted as percentage."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=CSV_COLUMNS,
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in self.rows:
                csv_row = {}
                for col in CSV_COLUMNS:
                    val = row.get(col)
                    # Format IRR as percentage string for readability
                    if col == "irr" and val is not None:
                        csv_row[col] = f"{val:.2%}"
                    elif col == "npv" and val is not None:
                        csv_row[col] = round(val, 2)
                    else:
                        csv_row[col] = val
                writer.writerow(csv_row)

    def to_json(self, path: str | Path) -> None:
        """Write full data (including yearly_snapshots) to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            "run_id": self.run_id,
            "fixed_inputs": self.fixed_inputs,
            "total_cases": self.n,
            "cases": self.rows,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    def filter(
        self,
        min_irr: float = None,
        max_irr: float = None,
        no_flags: bool = False,
        group_tag: str = None,
        max_speculation_tax: float = None,
        min_net_exit: float = None,
    ) -> "ResultsTable":
        """
        In-memory filter for quick analysis.
        The real product will do this on the frontend,
        but this is useful for demo and testing.
        """
        filtered = self.rows
        if min_irr is not None:
            filtered = [r for r in filtered
                        if r.get("irr") is not None and r["irr"] >= min_irr]
        if max_irr is not None:
            filtered = [r for r in filtered
                        if r.get("irr") is not None and r["irr"] <= max_irr]
        if no_flags:
            filtered = [r for r in filtered if not r.get("any_flag")]
        if group_tag is not None:
            filtered = [r for r in filtered
                        if r.get("_group_tag") == group_tag]
        if max_speculation_tax is not None:
            filtered = [r for r in filtered
                        if (r.get("speculation_tax") or 0) <= max_speculation_tax]
        if min_net_exit is not None:
            filtered = [r for r in filtered
                        if (r.get("net_exit_proceeds") or 0) >= min_net_exit]
        return ResultsTable(
            rows=filtered,
            run_id=self.run_id,
            fixed_inputs=self.fixed_inputs,
        )

    def summary_stats(self) -> dict:
        """Quick stats for console output."""
        irrs = [r["irr"] for r in self.rows if r.get("irr") is not None]
        flags = [r for r in self.rows if r.get("any_flag")]
        return {
            "total_rows":     self.n,
            "irr_min":        round(min(irrs), 4) if irrs else None,
            "irr_max":        round(max(irrs), 4) if irrs else None,
            "irr_mean":       round(sum(irrs) / len(irrs), 4) if irrs else None,
            "rows_with_flags": len(flags),
            "flag_rate":      round(len(flags) / self.n, 3) if self.n else 0,
        }


def build_results_table(
    simulation_results: list[SimulationResult],
    run_id: str,
    fixed_inputs: dict,
) -> ResultsTable:
    """Convert a list of SimulationResults into a ResultsTable."""
    rows = [format_row(r) for r in simulation_results]
    return ResultsTable(rows=rows, run_id=run_id, fixed_inputs=fixed_inputs)

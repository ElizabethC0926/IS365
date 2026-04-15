"""
main.py
-------
Command-line entry point for the real estate investment simulator.

Usage examples:
    # Basic run with required inputs
    python main.py --price 400000 --state Bayern --income 80000

    # Full options
    python main.py \\
        --price 400000 \\
        --state Bayern \\
        --income 80000 \\
        --purchase-year 2024 \\
        --filing married \\
        --market-rent 18000 \\
        --building standard \\
        --sale-price 480000 \\
        --output results/ \\
        --max-cases 500 \\
        --workers 4

    # Dry run (validate inputs, show what would be sampled, no simulation)
    python main.py --price 400000 --state Bayern --income 80000 --dry-run

Output:
    results/results_<run_id>.csv   — flat table, one row per strategy
    results/results_<run_id>.json  — full data with yearly snapshots
    results/run_<run_id>.log       — run metadata and timing
"""

from __future__ import annotations
import argparse
import datetime
import json
import multiprocessing
import sys
import time
from pathlib import Path

from sampler import generate_cases
from simulator import run_simulation
from output_formatter import build_results_table


# ---------------------------------------------------------------------------
# Worker function (must be top-level for multiprocessing)
# ---------------------------------------------------------------------------

def _simulate_one(args: tuple) -> dict | None:
    """
    Worker function for multiprocessing.Pool.
    Returns the SimulationResult serialised as a summary dict,
    or None if the simulation raises an unexpected error.
    """
    case, sale_price, tax_params_path = args
    try:
        from simulator import run_simulation

        PROPERTY_CASE_KEYS = {
            'purchase_price', 'state', 'purchase_year', 'personal_income',
            'filing_status', 'market_rent_annual', 'building_type', 'usage',
            'holding_years', 'rental_ratio', 'renovation_year',
            'renovation_amount', 'asset_split', 'land_ratio', 'equity_amount',
            'annual_rate', 'sondertilgung_rate', 'refi_year', 'refi_rate',
        }
        sim_params = {k: v for k, v in case.items() if k in PROPERTY_CASE_KEYS}
        result = run_simulation(
            sim_params,
            sale_price=sale_price,
            tax_params_path=tax_params_path,
        )
        result.params['_group_tag'] = case.get('_group_tag', 'grid')
        return result
    except Exception as exc:
        return None   # silently skip; counted in error tally


# ---------------------------------------------------------------------------
# Progress printer
# ---------------------------------------------------------------------------

def _progress(done: int, total: int, start_time: float, prefix: str = "") -> None:
    elapsed = time.time() - start_time
    pct = done / total * 100 if total else 0
    rate = done / elapsed if elapsed > 0 else 0
    eta  = (total - done) / rate if rate > 0 else 0
    bar_len = 30
    filled  = int(bar_len * done / total) if total else 0
    bar = "█" * filled + "░" * (bar_len - filled)
    print(
        f"\r  {prefix}[{bar}] {done}/{total} ({pct:.0f}%)  "
        f"{rate:.1f}/s  ETA {eta:.0f}s   ",
        end="", flush=True,
    )
    if done == total:
        print()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="German Real Estate Investment Strategy Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required fixed inputs
    req = p.add_argument_group("required property inputs")
    req.add_argument("--price",    type=float, required=True,
                     metavar="EUR", help="Purchase price in EUR")
    req.add_argument("--state",    type=str,   required=True,
                     metavar="STATE",
                     help="German Bundesland (e.g. Bayern, Berlin)")
    req.add_argument("--income",   type=float, required=True,
                     metavar="EUR", help="Annual personal income in EUR")

    # Optional fixed inputs
    opt = p.add_argument_group("optional property inputs")
    opt.add_argument("--purchase-year", type=int, default=2025,
                     metavar="YEAR", help="Year of purchase (default: 2025)")
    opt.add_argument("--filing",   type=str, default="single",
                     choices=["single", "married"],
                     help="Tax filing status (default: single)")
    opt.add_argument("--market-rent", type=float, default=None,
                     metavar="EUR",
                     help="Annual market rent EUR. "
                          "Default: 4.5%% of purchase price.")
    opt.add_argument("--building", type=str, default="standard",
                     choices=["standard", "neubau_post_2023", "denkmal"],
                     help="Building type (default: standard)")
    opt.add_argument("--sale-price", type=float, default=None,
                     metavar="EUR",
                     help="Assumed sale price at exit. "
                          "Default: purchase price (no appreciation).")

    # Run options
    run = p.add_argument_group("run options")
    run.add_argument("--output",    type=str, default="results",
                     metavar="DIR", help="Output directory (default: results/)")
    run.add_argument("--max-cases", type=int, default=None,
                     metavar="N",
                     help="Cap grid cases (forced groups always included). "
                          "Default: full grid.")
    run.add_argument("--workers",   type=int,
                     default=max(1, multiprocessing.cpu_count() - 1),
                     metavar="N",
                     help="Parallel workers (default: CPU count - 1)")
    run.add_argument("--dry-run",   action="store_true",
                     help="Validate inputs and show sample count, no simulation.")
    run.add_argument("--no-forced", action="store_true",
                     help="Skip forced analysis groups (15%% collision etc.)")
    run.add_argument("--tax-params", type=str, default="tax_params.json",
                     metavar="PATH",
                     help="Path to tax_params.json (default: tax_params.json)")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args   = parser.parse_args(argv)

    run_id     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Fixed inputs ──────────────────────────────────────────────────────
    market_rent = args.market_rent or round(args.price * 0.045, 2)
    sale_price  = args.sale_price  or args.price

    fixed_inputs = dict(
        purchase_price     = args.price,
        state              = args.state,
        purchase_year      = args.purchase_year,
        personal_income    = args.income,
        filing_status      = args.filing,
        market_rent_annual = market_rent,
        building_type      = args.building,
    )

    # ── Header ────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  German RE Investment Simulator")
    print("=" * 60)
    print(f"  Run ID       : {run_id}")
    print(f"  Purchase     : €{args.price:>12,.0f}  |  {args.state}")
    print(f"  Income       : €{args.income:>12,.0f}  |  {args.filing}")
    print(f"  Market rent  : €{market_rent:>12,.0f}/yr")
    print(f"  Sale price   : €{sale_price:>12,.0f}")
    print(f"  Building     : {args.building}")
    print(f"  Workers      : {args.workers}")
    print()

    # ── Sampling ──────────────────────────────────────────────────────────
    print("  [1/3] Sampling strategy space...")
    t0 = time.time()

    sample_result = generate_cases(
        fixed_inputs,
        include_forced_groups=not args.no_forced,
        max_cases=args.max_cases,
    )

    t_sample = time.time() - t0
    print(f"        {sample_result.total_valid} valid cases "
          f"({sample_result.total_invalid} invalid filtered) "
          f"in {t_sample:.1f}s")
    for group, count in sorted(sample_result.group_counts.items()):
        print(f"        ↳ {group:<35}: {count:>4}")
    print()

    if args.dry_run:
        print("  Dry run — stopping before simulation.")
        print("  (Remove --dry-run to run the full simulation)")
        return 0

    # ── Simulation ────────────────────────────────────────────────────────
    print(f"  [2/3] Running {sample_result.total_valid} simulations "
          f"({args.workers} workers)...")
    t1      = time.time()
    cases   = sample_result.cases
    n_total = len(cases)

    sim_args = [
        (case, sale_price, args.tax_params)
        for case in cases
    ]

    results      = []
    error_count  = 0

    if args.workers == 1:
        # Single-process mode (easier to debug)
        for i, arg in enumerate(sim_args):
            r = _simulate_one(arg)
            if r is not None:
                results.append(r)
            else:
                error_count += 1
            _progress(i + 1, n_total, t1, prefix="  ")
    else:
        with multiprocessing.Pool(processes=args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_simulate_one, sim_args)):
                if r is not None:
                    results.append(r)
                else:
                    error_count += 1
                _progress(i + 1, n_total, t1, prefix="  ")

    t_sim = time.time() - t1
    print(f"        {len(results)} succeeded, "
          f"{error_count} errors  in {t_sim:.1f}s  "
          f"({len(results)/t_sim:.0f} cases/s)")
    print()

    if not results:
        print("  ERROR: No simulations succeeded. Check inputs.")
        return 1

    # ── Output ────────────────────────────────────────────────────────────
    print("  [3/3] Writing results...")

    table = build_results_table(
        results,
        run_id=run_id,
        fixed_inputs=fixed_inputs,
    )

    csv_path  = output_dir / f"results_{run_id}.csv"
    json_path = output_dir / f"results_{run_id}.json"
    log_path  = output_dir / f"run_{run_id}.log"

    table.to_csv(csv_path)
    table.to_json(json_path)

    # Summary stats
    stats = table.summary_stats()

    # Log file
    log = {
        "run_id":         run_id,
        "fixed_inputs":   fixed_inputs,
        "sale_price":     sale_price,
        "total_sampled":  sample_result.total_valid,
        "total_simulated":len(results),
        "errors":         error_count,
        "timing_seconds": {
            "sampling":   round(t_sample, 2),
            "simulation": round(t_sim, 2),
            "total":      round(time.time() - t0, 2),
        },
        "summary_stats":  stats,
        "group_counts":   sample_result.group_counts,
        "output_files": {
            "csv":  str(csv_path),
            "json": str(json_path),
        },
    }
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    # ── Final summary ─────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Results Summary")
    print("=" * 60)
    print(f"  Total strategies : {stats['total_rows']:>6,}")
    print(f"  IRR range        : "
          f"{stats['irr_min']:.2%} – {stats['irr_max']:.2%}")
    print(f"  IRR mean         : {stats['irr_mean']:.2%}")
    print(f"  Rows with flags  : {stats['rows_with_flags']:>6,} "
          f"({stats['flag_rate']:.0%})")
    print()
    print(f"  CSV  → {csv_path}")
    print(f"  JSON → {json_path}")
    print(f"  Log  → {log_path}")
    print()

    # Quick filter previews
    clean = table.filter(
        no_flags=False,
        min_irr=0.03,
        max_speculation_tax=0,
    )
    print(f"  Quick filter: IRR>3% + no speculation tax → {clean.n} strategies")

    if clean.rows:
        top3 = sorted(
            clean.rows,
            key=lambda r: r.get("irr") or 0,
            reverse=True,
        )[:3]
        print("  Top 3 by IRR (no spec tax):")
        for row in top3:
            flags_hit = [
                k.replace("FLAG_", "") for k in
                ["FLAG_15_PERCENT_HIT", "FLAG_RENT_TOO_LOW",
                 "FLAG_TAX_WASTE", "FLAG_NEGATIVE_CASHFLOW"]
                if row.get(k)
            ]
            flag_str = ", ".join(flags_hit) if flags_hit else "none"
            print(
                f"    IRR={row['irr']:.2%}  "
                f"hold={row['holding_years']}yr  "
                f"rate={row['annual_rate']:.1%}  "
                f"net_exit=€{row['net_exit_proceeds']:,.0f}  "
                f"flags=[{flag_str}]"
            )

    print()
    print(f"  Total time: {time.time() - t0:.1f}s")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

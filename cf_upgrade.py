#cf-upgrader test chatGPT guided version to see what works 
#!/usr/bin/env python3
"""
cf_upgrader.py

MVP CF upgrade assistant:
- Runs cf-checker (cfchecks) for a target CF version
- Parses errors/warnings
- Applies a small set of safe fixes + interactive prompts
- Logs changes
- Supports single file or directory, with pilot mode

Requires:
  - Python 3.9+
  - netCDF4
  - cf-checker installed so 'cfchecks' is on PATH (your fork)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from netCDF4 import Dataset  # type: ignore
except Exception as e:
    print("ERROR: netCDF4 is required. Install with: pip install netCDF4", file=sys.stderr)
    raise

# ----------------------------
# Utilities
# ----------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False), encoding="utf-8")

def safe_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return repr(x)

def normalize_conventions_value(val: str) -> List[str]:
    # e.g. "CF-1.8, ACDD-1.3" -> ["CF-1.8", "ACDD-1.3"]
    parts = [p.strip() for p in val.split(",") if p.strip()]
    return parts

def join_conventions(parts: List[str]) -> str:
    return ", ".join(parts)

# ----------------------------
# cf-checker runner + parsing
# ----------------------------

@dataclass
class CFMessage:
    severity: str  # "ERROR" or "WARN" or other
    section: Optional[str]  # CF section reference like "2.6.1" if present
    text: str

# Common-ish patterns:
#   "ERROR (2.6.1): some message"
#   "WARN  (3.1): something"
MSG_RE = re.compile(r"^(ERROR|WARN)\s*\(([^)]+)\)\s*:\s*(.*)$")

def run_cfchecks(file_path: Path, target_cf: str, cfchecks_cmd: str = "cfchecks") -> Tuple[int, str]:
    """
    Runs cfchecks and returns (returncode, combined_output).
    Tries:
      1) cfchecks -v <target> <file>
      2) python -m cfchecker.cfchecks -v <target> <file>  (fallback)
    """
    cmd1 = [cfchecks_cmd, "-v", target_cf, str(file_path)]
    try:
        p = subprocess.run(cmd1, capture_output=True, text=True)
        out = (p.stdout or "") + (p.stderr or "")
        # if command not found, returncode=127 on some systems; also might throw FileNotFoundError
        return p.returncode, out
    except FileNotFoundError:
        pass

    # fallback if cfchecks not on PATH
    cmd2 = [sys.executable, "-m", "cfchecker.cfchecks", "-v", target_cf, str(file_path)]
    p = subprocess.run(cmd2, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode, out

def parse_cfchecks_output(output: str) -> List[CFMessage]:
    msgs: List[CFMessage] = []
    for line in output.splitlines():
        m = MSG_RE.match(line.strip())
        if m:
            sev, sec, txt = m.group(1), m.group(2), m.group(3)
            msgs.append(CFMessage(severity=sev, section=sec, text=txt))
        else:
            # keep non-matching lines? For MVP we ignore unless they look like errors.
            continue
    return msgs

# ----------------------------
# Snapshot + diff (metadata only)
# ----------------------------

def snapshot_metadata(nc_path: Path) -> Dict[str, Any]:
    """
    Snapshot global attrs + variable attrs (no data), plus dims.
    """
    snap: Dict[str, Any] = {}
    with Dataset(nc_path, "r") as ds:
        snap["global_attributes"] = {k: safe_str(getattr(ds, k)) for k in ds.ncattrs()}
        snap["dimensions"] = {d: len(ds.dimensions[d]) for d in ds.dimensions}
        vars_obj: Dict[str, Any] = {}
        for vname, var in ds.variables.items():
            vars_obj[vname] = {
                "dimensions": list(var.dimensions),
                "dtype": safe_str(var.dtype),
                "attributes": {k: safe_str(getattr(var, k)) for k in var.ncattrs()},
                "shape": list(var.shape),
            }
        snap["variables"] = vars_obj
    return snap

def diff_snapshots(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """
    Small, readable diff: created vars, deleted vars, changed global attrs,
    changed var attrs (only where different).
    """
    d: Dict[str, Any] = {}

    bga = before.get("global_attributes", {})
    aga = after.get("global_attributes", {})
    changed_ga = {}
    for k in sorted(set(bga) | set(aga)):
        if bga.get(k) != aga.get(k):
            changed_ga[k] = {"before": bga.get(k), "after": aga.get(k)}
    d["global_attributes_changed"] = changed_ga

    bvars = before.get("variables", {})
    avars = after.get("variables", {})
    created = sorted(set(avars) - set(bvars))
    deleted = sorted(set(bvars) - set(avars))
    d["variables_created"] = created
    d["variables_deleted"] = deleted

    changed_var_attrs: Dict[str, Any] = {}
    common = sorted(set(bvars) & set(avars))
    for v in common:
        bva = bvars[v].get("attributes", {})
        ava = avars[v].get("attributes", {})
        ch = {}
        for k in sorted(set(bva) | set(ava)):
            if bva.get(k) != ava.get(k):
                ch[k] = {"before": bva.get(k), "after": ava.get(k)}
        if ch:
            changed_var_attrs[v] = ch
    d["variable_attributes_changed"] = changed_var_attrs

    return d

# ----------------------------
# Answer cache (pilot mode)
# ----------------------------

def load_answers(path: Path) -> Dict[str, Any]:
    return read_json(path) or {}

def save_answers(path: Path, answers: Dict[str, Any]) -> None:
    write_json(path, answers)

def ask_choice(prompt: str, options: List[str], default: Optional[int], interactive: bool) -> Optional[str]:
    """
    options: list of strings to display. returns chosen option string, or None if skipped.
    """
    if not interactive:
        return None

    print("\n" + prompt)
    for i, opt in enumerate(options, start=1):
        print(f"  {i}) {opt}")
    if default is not None:
        print(f"Press Enter for default [{default}]")

    while True:
        raw = input("> ").strip()
        if raw == "" and default is not None:
            idx = default
        else:
            if raw.lower() in ("s", "skip", "q", "quit"):
                return None
            if not raw.isdigit():
                print("Enter a number, or 'skip'.")
                continue
            idx = int(raw)

        if 1 <= idx <= len(options):
            return options[idx - 1]
        print("Invalid choice.")

def ask_text(prompt: str, default: Optional[str], interactive: bool) -> Optional[str]:
    if not interactive:
        return None
    if default is not None:
        prompt = f"{prompt} [default: {default}]"
    print("\n" + prompt)
    raw = input("> ").strip()
    if raw == "" and default is not None:
        return default
    if raw.lower() in ("s", "skip", "q", "quit"):
        return None
    return raw

# ----------------------------
# Fix application
# ----------------------------

@dataclass
class Change:
    timestamp: str
    action: str
    target: str
    before: Optional[str]
    after: Optional[str]
    note: Optional[str] = None
    user_supplied: bool = False

def ensure_global_attr(ds: Dataset, name: str, value: str, changes: List[Change], note: Optional[str] = None) -> None:
    before = safe_str(getattr(ds, name)) if name in ds.ncattrs() else None
    if before == value:
        return
    setattr(ds, name, value)
    changes.append(Change(
        timestamp=utc_now_iso(),
        action="set_global_attribute",
        target=name,
        before=before,
        after=value,
        note=note,
        user_supplied=False,
    ))

def append_history(ds: Dataset, entry: str, changes: List[Change]) -> None:
    old = safe_str(getattr(ds, "history")) if "history" in ds.ncattrs() else ""
    new = (old + "\n" if old else "") + entry
    setattr(ds, "history", new)
    changes.append(Change(
        timestamp=utc_now_iso(),
        action="append_history",
        target="history",
        before=old if old else None,
        after=new,
        note=None,
        user_supplied=False,
    ))

def ensure_conventions_contains(ds: Dataset, target_cf: str, changes: List[Change]) -> None:
    """
    Ensure global Conventions contains "CF-x.y" exactly once.
    """
    want = f"CF-{target_cf}"
    if "Conventions" in ds.ncattrs():
        parts = normalize_conventions_value(safe_str(getattr(ds, "Conventions")))
    else:
        parts = []
    if want in parts:
        return
    # remove other CF-* tokens if present? conservative: keep them, add new.
    # But if multiple CF versions appear, CF-checker might complain.
    # In "upgrade" mode, replace any CF-* with the target.
    new_parts = [p for p in parts if not p.startswith("CF-")]
    new_parts.insert(0, want)
    new_val = join_conventions(new_parts)
    before = join_conventions(parts) if parts else None
    setattr(ds, "Conventions", new_val)
    changes.append(Change(
        timestamp=utc_now_iso(),
        action="update_conventions",
        target="Conventions",
        before=before,
        after=new_val,
        note="Replaced/inserted CF version token to match target.",
        user_supplied=False,
    ))

def add_feature_type(ds: Dataset, feature_type: str, changes: List[Change], user_supplied: bool) -> None:
    before = safe_str(getattr(ds, "featureType")) if "featureType" in ds.ncattrs() else None
    setattr(ds, "featureType", feature_type)
    changes.append(Change(
        timestamp=utc_now_iso(),
        action="set_global_attribute",
        target="featureType",
        before=before,
        after=feature_type,
        note="Set per user input to satisfy DSG requirements.",
        user_supplied=user_supplied,
    ))

def create_grid_mapping_var(ds: Dataset, var_name: str, grid_mapping_name: str, params: Dict[str, Any],
                            changes: List[Change], user_supplied: bool) -> None:
    """
    Create an empty scalar int variable for CRS/grid mapping with CF attributes.
    Very common pattern: int crs; crs:grid_mapping_name="lambert_conformal_conic"; ...
    """
    if var_name in ds.variables:
        # update attributes only
        v = ds.variables[var_name]
        before = safe_str(getattr(v, "grid_mapping_name")) if "grid_mapping_name" in v.ncattrs() else None
        setattr(v, "grid_mapping_name", grid_mapping_name)
        changes.append(Change(
            timestamp=utc_now_iso(),
            action="set_variable_attribute",
            target=f"{var_name}.grid_mapping_name",
            before=before,
            after=grid_mapping_name,
            note=None,
            user_supplied=user_supplied,
        ))
    else:
        v = ds.createVariable(var_name, "i4")  # scalar
        v[...] = 0
        changes.append(Change(
            timestamp=utc_now_iso(),
            action="create_variable",
            target=var_name,
            before=None,
            after=f"i4 scalar; grid_mapping_name={grid_mapping_name}",
            note="Created grid mapping / CRS variable.",
            user_supplied=user_supplied,
        ))
        setattr(v, "grid_mapping_name", grid_mapping_name)

    # set params
    for k, val in params.items():
        before = safe_str(getattr(ds.variables[var_name], k)) if k in ds.variables[var_name].ncattrs() else None
        setattr(ds.variables[var_name], k, val)
        changes.append(Change(
            timestamp=utc_now_iso(),
            action="set_variable_attribute",
            target=f"{var_name}.{k}",
            before=before,
            after=safe_str(val),
            note=None,
            user_supplied=user_supplied,
        ))

def ensure_grid_mapping_attr_on_data_vars(ds: Dataset, grid_mapping_var: str, changes: List[Change]) -> None:
    """
    Conservative: only set grid_mapping attribute on variables that already reference it
    via cf-checker message (we don't know which ones). Here we do nothing automatically.
    You can extend this later.
    """
    # Intentionally no-op in MVP; safer to only create missing mapping var.
    return

# ----------------------------
# Issue handlers (MVP)
# ----------------------------

def infer_missing_grid_mapping_var_from_msgs(msgs: List[CFMessage]) -> Optional[str]:
    """
    If cf-checker says a grid_mapping variable is missing, try to extract its name.
    Handles common message phrasing.
    """
    # Example-ish phrases vary; we match: grid_mapping 'crs' not defined / missing variable 'crs'
    patterns = [
        re.compile(r"grid_mapping\s+'([^']+)'\s+.*not\s+defined", re.IGNORECASE),
        re.compile(r"grid_mapping\s+attribute.*'([^']+)'.*not\s+defined", re.IGNORECASE),
        re.compile(r"Missing.*grid_mapping.*variable\s+'([^']+)'", re.IGNORECASE),
        re.compile(r"grid_mapping\s+variable\s+'([^']+)'\s+is\s+missing", re.IGNORECASE),
    ]
    for m in msgs:
        if m.severity != "ERROR":
            continue
        for pat in patterns:
            mm = pat.search(m.text)
            if mm:
                return mm.group(1)
    return None

def has_missing_feature_type(msgs: List[CFMessage]) -> bool:
    for m in msgs:
        if m.severity == "ERROR" and re.search(r"\bfeatureType\b", m.text):
            if re.search(r"Missing|required|must", m.text, re.IGNORECASE):
                return True
    return False

def has_conventions_problem(msgs: List[CFMessage]) -> bool:
    for m in msgs:
        if re.search(r"\bConventions\b", m.text, re.IGNORECASE):
            return True
    return False

def count_errors(msgs: List[CFMessage]) -> int:
    return sum(1 for m in msgs if m.severity == "ERROR")

# ----------------------------
# Main upgrade routine
# ----------------------------

def upgrade_one_file(
    in_path: Path,
    out_path: Path,
    target_cf: str,
    interactive: bool,
    dry_run: bool,
    answers: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Returns a dict report including:
      - pass/fail
      - cfchecks output
      - changes
      - diff preview
    """
    report: Dict[str, Any] = {
        "input": str(in_path),
        "output": str(out_path),
        "target_cf": target_cf,
        "started": utc_now_iso(),
        "dry_run": dry_run,
        "interactive": interactive,
        "iterations": [],
        "changes": [],
        "final": {},
    }

    # Make a working copy to modify
    if not dry_run:
        ensure_dir(out_path.parent)
        shutil.copy2(in_path, out_path)
    else:
        # In dry-run, we still snapshot "after" as same as before
        pass

    before_snap = snapshot_metadata(in_path)
    report["before_snapshot"] = before_snap

    max_iters = 10
    cumulative_changes: List[Change] = []

    for it in range(1, max_iters + 1):
        # Run checker
        rc, out = run_cfchecks(out_path if not dry_run else in_path, target_cf)
        msgs = parse_cfchecks_output(out)

        iter_info = {
            "iteration": it,
            "cfchecks_returncode": rc,
            "errors": count_errors(msgs),
            "messages": [asdict(m) for m in msgs],
            "raw_output_excerpt": "\n".join(out.splitlines()[:200]),  # keep it bounded
        }
        report["iterations"].append(iter_info)

        if count_errors(msgs) == 0:
            break  # passes checker for this target version

        # Apply fixes for this iteration (MVP set)
        iter_changes: List[Change] = []

        if not dry_run:
            with Dataset(out_path, "r+") as ds:
                # 1) Conventions fix (always safe)
                if has_conventions_problem(msgs) or True:
                    ensure_conventions_contains(ds, target_cf, iter_changes)

                # 2) featureType for DSG (prompt)
                if has_missing_feature_type(msgs):
                    # answers cache key
                    key = "featureType"
                    ft = answers.get(key)
                    if not ft:
                        ft = ask_choice(
                            "CF upgrade needs a DSG featureType (global attribute 'featureType'). Choose:",
                            ["point", "timeSeries", "profile", "trajectory", "timeSeriesProfile", "trajectoryProfile"],
                            default=2,
                            interactive=interactive,
                        )
                        if ft:
                            answers[key] = ft

                    if ft:
                        add_feature_type(ds, ft, iter_changes, user_supplied=True)

                # 3) Missing grid_mapping variable referenced
                missing_gm = infer_missing_grid_mapping_var_from_msgs(msgs)
                if missing_gm:
                    # allow cached projection choice + params
                    proj_key = f"grid_mapping.{missing_gm}.grid_mapping_name"
                    gm_name = answers.get(proj_key)
                    if not gm_name:
                        gm_name = ask_choice(
                            f"cf-checker indicates grid_mapping '{missing_gm}' is referenced but missing.\n"
                            f"We can create a CRS/grid mapping variable named '{missing_gm}'. Choose grid_mapping_name:",
                            [
                                "latitude_longitude",
                                "rotated_latitude_longitude",
                                "lambert_conformal_conic",
                                "polar_stereographic",
                                "mercator",
                            ],
                            default=1,
                            interactive=interactive,
                        )
                        if gm_name:
                            answers[proj_key] = gm_name

                    params: Dict[str, Any] = {}
                    if gm_name:
                        # Minimal parameter prompting; you can extend later.
                        if gm_name == "rotated_latitude_longitude":
                            p1 = ask_text("Enter grid_north_pole_latitude (e.g. 37.5)", answers.get(proj_key + ".grid_north_pole_latitude"), interactive)
                            p2 = ask_text("Enter grid_north_pole_longitude (e.g. 177.5)", answers.get(proj_key + ".grid_north_pole_longitude"), interactive)
                            if p1 is not None:
                                answers[proj_key + ".grid_north_pole_latitude"] = p1
                                params["grid_north_pole_latitude"] = float(p1)
                            if p2 is not None:
                                answers[proj_key + ".grid_north_pole_longitude"] = p2
                                params["grid_north_pole_longitude"] = float(p2)

                        elif gm_name == "lambert_conformal_conic":
                            sp = ask_text("Enter standard_parallel (one value or two comma-separated, e.g. 30,60)", answers.get(proj_key + ".standard_parallel"), interactive)
                            lon0 = ask_text("Enter longitude_of_central_meridian (e.g. -95)", answers.get(proj_key + ".longitude_of_central_meridian"), interactive)
                            lat0 = ask_text("Enter latitude_of_projection_origin (e.g. 40)", answers.get(proj_key + ".latitude_of_projection_origin"), interactive)
                            if sp is not None:
                                answers[proj_key + ".standard_parallel"] = sp
                                vals = [float(x.strip()) for x in sp.split(",") if x.strip()]
                                params["standard_parallel"] = vals if len(vals) > 1 else vals[0]
                            if lon0 is not None:
                                answers[proj_key + ".longitude_of_central_meridian"] = lon0
                                params["longitude_of_central_meridian"] = float(lon0)
                            if lat0 is not None:
                                answers[proj_key + ".latitude_of_projection_origin"] = lat0
                                params["latitude_of_projection_origin"] = float(lat0)

                        elif gm_name == "polar_stereographic":
                            lon0 = ask_text("Enter straight_vertical_longitude_from_pole (e.g. 0)", answers.get(proj_key + ".straight_vertical_longitude_from_pole"), interactive)
                            lat_ts = ask_text("Enter standard_parallel OR latitude_of_projection_origin (e.g. 70)", answers.get(proj_key + ".standard_parallel"), interactive)
                            if lon0 is not None:
                                answers[proj_key + ".straight_vertical_longitude_from_pole"] = lon0
                                params["straight_vertical_longitude_from_pole"] = float(lon0)
                            if lat_ts is not None:
                                answers[proj_key + ".standard_parallel"] = lat_ts
                                params["standard_parallel"] = float(lat_ts)

                        # latitude_longitude typically needs none beyond grid_mapping_name
                        create_grid_mapping_var(ds, missing_gm, gm_name, params, iter_changes, user_supplied=True)

                # 4) Always append history for traceability (safe)
                append_history(ds, f"{utc_now_iso()} cf-upgrader: attempted upgrade to CF-{target_cf}", iter_changes)

        cumulative_changes.extend(iter_changes)

        # If no changes made this iter, stop to avoid infinite loop
        if not iter_changes:
            break

    report["changes"] = [asdict(c) for c in cumulative_changes]
    report["finished"] = utc_now_iso()

    # Snapshot after + diff preview
    if dry_run:
        after_snap = before_snap
    else:
        after_snap = snapshot_metadata(out_path)
    report["after_snapshot"] = after_snap
    report["preview_diff"] = diff_snapshots(before_snap, after_snap)

    # Final status: last iteration
    last_iter = report["iterations"][-1] if report["iterations"] else {}
    report["final"] = {
        "passes_cfchecks": (last_iter.get("errors", 999) == 0),
        "final_errors": last_iter.get("errors"),
        "cfchecks_returncode": last_iter.get("cfchecks_returncode"),
    }
    return report

# ----------------------------
# CLI / batch orchestration
# ----------------------------

def discover_netcdf_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    files = []
    for ext in ("*.nc", "*.nc4", "*.cdf"):
        files.extend(path.rglob(ext))
    return sorted(set(files))

def main() -> int:
    ap = argparse.ArgumentParser(description="CF upgrade assistant using cf-checker (cfchecks).")
    ap.add_argument("input", help="NetCDF file or directory")
    ap.add_argument("--target-cf", required=True, help="Target CF version like 1.8 or 1.12")
    ap.add_argument("--out-dir", default="cf_upgraded_out", help="Output directory (for directory input)")
    ap.add_argument("--in-place", action="store_true", help="Modify files in place (dangerous; no backup in MVP)")
    ap.add_argument("--dry-run", action="store_true", help="Do not modify; still run checker and show planned diffs")
    ap.add_argument("--non-interactive", action="store_true", help="Do not prompt; only apply safe automatic fixes")
    ap.add_argument("--pilot", action="store_true", help="For directory: run first file interactively, then reuse answers")
    ap.add_argument("--answers", default="cf_upgrader_answers.json", help="Path to cached answers JSON")
    ap.add_argument("--log-dir", default="cf_upgrader_logs", help="Directory for JSON logs")
    args = ap.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    target_cf = args.target_cf.strip()
    interactive = not args.non_interactive
    dry_run = args.dry_run

    files = discover_netcdf_files(in_path)
    if not files:
        print(f"No NetCDF files found at: {in_path}", file=sys.stderr)
        return 2

    answers_path = Path(args.answers).expanduser().resolve()
    answers = load_answers(answers_path)

    log_dir = Path(args.log_dir).expanduser().resolve()
    ensure_dir(log_dir)

    run_report = {
        "started": utc_now_iso(),
        "input": str(in_path),
        "target_cf": target_cf,
        "files_total": len(files),
        "results": [],
    }

    # Determine processing plan
    for idx, f in enumerate(files):
        # choose output path
        if args.in_place:
            out_path = f
        else:
            if in_path.is_file():
                out_path = Path(args.out_dir).expanduser().resolve() / f.name
            else:
                rel = f.relative_to(in_path)
                out_path = Path(args.out_dir).expanduser().resolve() / rel

        # Pilot logic: first file interactive; rest use cached answers and optionally non-interactive
        this_interactive = interactive
        if args.pilot and in_path.is_dir() and idx > 0:
            this_interactive = False  # apply cached answers without prompting

        print(f"\n=== [{idx+1}/{len(files)}] {f} ===")
        if not args.in_place:
            print(f"Output: {out_path}")

        rep = upgrade_one_file(
            in_path=f,
            out_path=out_path,
            target_cf=target_cf,
            interactive=this_interactive,
            dry_run=dry_run,
            answers=answers,
        )

        # Persist per-file log
        log_name = f"{f.stem}.cf-upgrader.log.json"
        log_path = log_dir / log_name
        write_json(log_path, rep)
        print(f"Log: {log_path}")

        # Print preview diff summary
        diff = rep.get("preview_diff", {})
        ga = diff.get("global_attributes_changed", {})
        created = diff.get("variables_created", [])
        vch = diff.get("variable_attributes_changed", {})

        print("\n--- Preview diff (metadata) ---")
        if ga:
            print("Global attributes changed:")
            for k, vv in ga.items():
                print(f"  - {k}: {vv.get('before')}  ->  {vv.get('after')}")
        if created:
            print("Variables created:", ", ".join(created))
        if vch:
            print("Variable attribute changes:")
            # print top few for readability
            shown = 0
            for vname, ch in vch.items():
                print(f"  - {vname}:")
                for ak, vv in list(ch.items())[:10]:
                    print(f"      {ak}: {vv.get('before')} -> {vv.get('after')}")
                shown += 1
                if shown >= 10:
                    break
        if not ga and not created and not vch:
            print("(no metadata changes detected)")

        status = rep.get("final", {}).get("passes_cfchecks", False)
        print(f"\nResult: {'PASS' if status else 'NOT PASS'} for CF-{target_cf}")

        run_report["results"].append({
            "file": str(f),
            "output": str(out_path),
            "passes_cfchecks": status,
            "final_errors": rep.get("final", {}).get("final_errors"),
            "log": str(log_path),
        })

        # After first file in pilot mode, save answers
        if args.pilot and idx == 0:
            save_answers(answers_path, answers)
            print(f"\nSaved pilot answers to: {answers_path}")
            print("Continuing remaining files using cached answers (no prompts).")

    run_report["finished"] = utc_now_iso()
    run_summary_path = log_dir / "cf-upgrader.run-summary.json"
    write_json(run_summary_path, run_report)
    print(f"\nRun summary: {run_summary_path}")

    # Save answers at end too
    save_answers(answers_path, answers)

    # Exit code: 0 if all passed
    all_pass = all(r.get("passes_cfchecks") for r in run_report["results"])
    return 0 if all_pass else 1

if __name__ == "__main__":
    raise SystemExit(main())

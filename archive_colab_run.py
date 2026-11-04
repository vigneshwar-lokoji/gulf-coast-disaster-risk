"""Rescue a Colab run into artifacts/runs/ as a normal archived run.

The run that produced the headline numbers predates the archiving in
pipeline.py, so its only surviving record is the console output embedded in
the downloaded notebook plus whatever is still sitting in the Drive folder.
Drive is not an archive: the tag is still `run`, so the next full run
overwrites every one of those files in place.

This reads the notebook, writes a manifest in the same shape pipeline.py
emits, and copies in any files recovered from Drive. Re-runnable: point it at
a --drive-dir later and the recovered files fold into the same run directory.

    python archive_colab_run.py "~/Downloads/pipeline (1).ipynb" \
        --drive-dir ~/Desktop/crisk_drive_upload/crisk/artifacts

Later Colab runs do not need this. They archive themselves, because
pipeline.py now writes into artifacts/runs/<id>/ on the Drive mount.
"""

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def notebook_console(nb):
    """Every stream output, in cell order, as one log."""
    out = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        txt = "".join(
            "".join(o.get("text", []))
            for o in c.get("outputs", [])
            if o.get("output_type") == "stream"
        )
        if txt.strip():
            out.append(f"===== cell {i} =====\n{txt.rstrip()}\n")
    return "\n".join(out)


def need(pattern, text, what, cast=str, group=1):
    """Parse or fail loudly. A manifest with a silent null is worse than none."""
    m = re.search(pattern, text, re.M)
    if not m:
        raise SystemExit(f"could not find {what} in the notebook output.\n"
                         f"pattern: {pattern}")
    return cast(m.group(group))


def parse(log):
    """Pull the scalars worth having structured. console.log holds the rest."""
    versions = {
        name: need(rf"^{name}\s+(\S+)", log, f"{name} version")
        for name in ("python", "numpy", "pandas", "xgboost")
    }
    torch_v, cuda = re.search(r"^torch\s+(\S+)\s+cuda=(\S+)", log,
                              re.M).groups()
    versions["torch"] = torch_v

    splits = []
    for name, lo, hi, rows, counties, fpos, frate, spos, srate in re.findall(
        r"^\s*(train|val|test)\s+(\d{4})-(\d{4})\s+(\d+)\s+(\d+)\s+"
        r"(\d+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s*$", log, re.M
    ):
        splits.append({"split": name, "years": f"{lo}-{hi}", "rows": int(rows),
                       "counties": int(counties), "flood_pos": int(fpos),
                       "flood_rate%": float(frate), "storm_pos": int(spos),
                       "storm_rate%": float(srate)})
    if len(splits) != 3:
        raise SystemExit(f"expected 3 split rows, parsed {len(splits)}")

    # The summary table is the one place every model's test AP appears beside
    # the baseline it has to beat, so it is parsed rather than the per-cell
    # prints which differ in format between XGBoost and the TFT.
    models = {}
    for haz, model, ap, vs in re.findall(
        r"^\s*(flood|storm)\s+(baseline: \S+|xgboost|tft)\s+"
        r"([\d.]+)\s+(?:[\d.]+|NaN)\s+([\d.]+)\s+\S+\s*$", log, re.M
    ):
        models.setdefault(haz, {})[model] = {"test_ap": float(ap),
                                             "vs_baseline": float(vs)}
    if set(models) != {"flood", "storm"}:
        raise SystemExit(f"expected flood and storm in the summary, "
                         f"got {sorted(models)}")

    fit = {}
    for haz, it, secs in re.findall(
        r"=== xgboost (\w+) ===\n\s+best_iteration=(\d+)\s+([\d.]+)s", log
    ):
        fit[f"xgb:{haz}"] = {"best_iteration": int(it), "seconds": float(secs)}
    for haz, ep, vl, secs in re.findall(
        r"=== tft (\w+) ===\n\s+epochs=(\d+)\s+best_val_loss=([\d.]+)\s+([\d.]+)s",
        log
    ):
        fit[f"tft:{haz}"] = {"epochs_run": int(ep), "best_val_loss": float(vl),
                             "seconds": float(secs)}

    warm = dict(re.findall(r"warm-start (\w+): xgb=(\w+)\s+tft=(\w+)", log)
                and [(h, {"xgb": x == "True", "tft": t == "True"})
                     for h, x, t in re.findall(
                         r"warm-start (\w+): xgb=(\w+)\s+tft=(\w+)", log)])

    # Every cell prints the path it wrote, so the log itself is the list of
    # what this run produced. Comparing that against what comes back off Drive
    # is the only way to know whether the recovery is complete -- Colab's Drive
    # mount buffers writes, and a file the log says was written is not
    # necessarily a file that survived the session.
    wrote = sorted({Path(p.strip()).name
                    for p in re.findall(r"^\s*wrote (\S.*)$", log, re.M)})

    return {
        "wrote_per_log": wrote,
        "versions": versions,
        "cuda_available": cuda == "True",
        "gpu_banner": need(r"^torch\s+\S+\s+cuda=\S+", log, "torch banner",
                           group=0),
        "root": need(r"^root\s+(.+)$", log, "root"),
        "panel_line": need(r"^panel\s+(.+)$", log, "panel line"),
        "tag": need(r"^smoke \S+\s+tag (\S+)", log, "tag"),
        "smoke": need(r"^smoke (\S+)", log, "smoke flag") == "True",
        "warm_start_at_entry": warm,
        "splits": splits,
        "fit": fit,
        "models": models,
    }


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", type=Path)
    ap.add_argument("--drive-dir", type=Path, default=None,
                    help="an artifacts/ directory pulled off Drive, to fold "
                         "its results/ and plots/ into the run")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--runs-dir", type=Path,
                    default=HERE / "artifacts" / "runs")
    args = ap.parse_args()

    nb_path = args.notebook.expanduser()
    nb = json.loads(nb_path.read_text())
    log = notebook_console(nb)
    if not log.strip():
        raise SystemExit(f"{nb_path} has no saved cell output -- nothing to "
                         f"rescue. Download the notebook with outputs intact.")
    facts = parse(log)

    # Colab leaves no wall clock in the output, so the notebook's own mtime is
    # the best available stamp. Flagged as approximate rather than presented as
    # if the run itself had reported it.
    stamp = datetime.fromtimestamp(nb_path.stat().st_mtime, timezone.utc)
    run_id = args.run_id or f"{stamp:%Y%m%dT%H%M%SZ}-{facts['tag']}-colab"
    run_dir = args.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "console.log").write_text(log)
    shutil.copy2(nb_path, run_dir / "notebook.ipynb")

    recovered, missing_note = {}, None
    if args.drive_dir:
        src = args.drive_dir.expanduser()
        for sub in ("results", "plots"):
            for f in sorted((src / sub).glob(f"*_{facts['tag']}.*")):
                dest = run_dir / sub / f.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
                recovered[f"{sub}/{f.name}"] = sha256(f)
    else:
        missing_note = ("results and plots not recovered: pass --drive-dir "
                        "pointing at an artifacts/ directory pulled off Drive")

    got_names = {Path(k).name for k in recovered}
    lost = [n for n in facts["wrote_per_log"] if n not in got_names]
    if lost:
        missing_note = (
            f"{len(lost)} of {len(facts['wrote_per_log'])} files the run "
            f"reported writing were not recovered: {', '.join(lost)}")

    manifest = {
        "run_id": run_id,
        "tag": facts["tag"],
        "smoke": facts["smoke"],
        "env": "colab",
        "host": "colab (GPU runtime)",
        "root": facts["root"],
        "retro_ingested": True,
        "ingested_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "started_utc": None,
        "finished_utc": None,
        "approx_run_date": stamp.isoformat(timespec="seconds"),
        "approx_run_date_basis": "mtime of the downloaded notebook",
        "provenance": (
            "Rescued from the downloaded notebook's embedded cell output by "
            "archive_colab_run.py. This run predates the run-archiving in "
            "pipeline.py, so there is no manifest written at run time. Every "
            "figure in this manifest is parsed from console.log, which is the "
            "verbatim notebook output and is included alongside it."
        ),
        "git": {"commit": None, "branch": None, "dirty": None,
                "note": "Colab ran an uploaded notebook, not a git checkout"},
        "versions": facts["versions"],
        "cuda_available": facts["cuda_available"],
        "panel": {"line_as_printed": facts["panel_line"],
                  "sha256": None,
                  "note": "panel hashing was added after this run; the "
                          "notebook records only the path"},
        "splits": facts["splits"],
        "fit": facts["fit"],
        "warm_start_at_entry": facts["warm_start_at_entry"],
        # Both hazards reported xgb=False tft=False at entry, so nothing was
        # inherited and the numbers stand on this run alone.
        "self_contained": not any(
            v for d in facts["warm_start_at_entry"].values() for v in d.values()),
        "headline": facts["models"],
        "recovered_from_drive": recovered,
        "incomplete": missing_note,
        "inventory": sorted(str(p.relative_to(run_dir))
                            for p in run_dir.rglob("*") if p.is_file()),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"archived {run_id}")
    print(f"  {run_dir}")
    print(f"  cold start: {manifest['self_contained']}")
    print(f"  console.log {len(log):,} chars, {len(manifest['inventory'])} files")
    for haz, ms in facts["models"].items():
        for name, v in ms.items():
            print(f"  {haz:6s} {name:30s} AP={v['test_ap']:.5f}  "
                  f"vs_baseline={v['vs_baseline']:.2f}x")
    if missing_note:
        print(f"  INCOMPLETE: {missing_note}")


if __name__ == "__main__":
    main()

"""Run the notebook workflow in order, with a fresh kernel for each stage."""

import argparse
from pathlib import Path
from time import perf_counter

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
NOTEBOOKS = (
    "01_data.ipynb",
    "02_model.ipynb",
    "03_scoring.ipynb",
    "04_drift.ipynb",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebook",
        nargs="?",
        choices=NOTEBOOKS,
        help="Run one stage; omit to run all four in order. Earlier outputs must exist.",
    )
    args = parser.parse_args()
    selected = (args.notebook,) if args.notebook else NOTEBOOKS
    destination = ROOT / "output" / "executed"
    destination.mkdir(parents=True, exist_ok=True)

    for name in selected:
        saved = destination / name
        saved.unlink(missing_ok=True)
        notebook = nbformat.read(ROOT / "notebooks" / name, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        started = perf_counter()
        print(f"Running {name} ...", flush=True)
        try:
            NotebookClient(
                notebook,
                timeout=600,
                kernel_name="python3",
                resources={"metadata": {"path": str(ROOT)}},
            ).execute()
        finally:
            # A failed stage keeps its traceback here and stops the run.
            nbformat.write(notebook, saved)
        print(f"Passed {name} ({perf_counter() - started:.1f}s)", flush=True)

    print("Results: output/; executed notebook copies: output/executed/.")


if __name__ == "__main__":
    main()

# python -m pytest -v -s

import os, sys, glob, subprocess, pathlib, pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = [
    p for p in glob.glob("examples/**/*.py", recursive=True)
    if not os.path.basename(p).startswith("_")
    and os.path.basename(p) != "__init__.py"
]

@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: os.path.relpath(p, REPO))
def test_example_runs(example):
    """Run each example headlessly (no GUI)."""
    print(f"\n🧩 Running example: {example}", flush=True)

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"  # disable GUI
    env["JAXDENTIFY_TEST"] = "1"  # allow examples to reduce runtime under pytest
    env["JAXDENTIFY_FROM_PYTEST"] = "1"  # mark subprocess as launched by pytest
    env["PYTHONUTF8"] = "1"  # avoid Windows cp1252 UnicodeEncodeError
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable, example],
        cwd=REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert result.returncode == 0, (
        f"\n❌ Example failed: {example}\n"
        f"----- OUTPUT -----\n{result.stdout}"
    )

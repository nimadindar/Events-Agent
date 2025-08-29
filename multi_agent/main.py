import sys
import subprocess
from pathlib import Path
from datetime import datetime


PIPELINE = ["ResearchTeam", "PostingTeam"]

def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")

def find_scripts(folder: Path):
    """Yield .py files to run, sorted, excluding dunder and private files."""
    for p in sorted(folder.glob("*.py")):
        name = p.name
        if name == "__init__.py":
            continue
        if name.startswith("_"):
            continue
        yield p

def to_module_path(root_pkg_dir: Path, file_path: Path) -> str:
    """
    Convert a file path under `root_pkg_dir` into a dotted module path.
    E.g., /.../multi_agent/ResearchTeam/arxiv_node.py -> multi_agent.ResearchTeam.arxiv_node
    """
    rel = file_path.relative_to(root_pkg_dir).with_suffix("")       
    parts = [root_pkg_dir.name] + list(rel.parts)                   
    return ".".join(parts)

def run_module(module_path: str, cwd: Path) -> int:
    """Run module in a fresh process; stream output; return exit code."""
    log(f"RUN  -B  -m {module_path}")
    proc = subprocess.Popen(
        [sys.executable, "-m", module_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,            
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    code = proc.returncode
    if code == 0:
        log(f"OK     {module_path}")
    else:
        log(f"ERROR  {module_path} (exit code {code})")
    return code

def main() -> int:
    root_pkg_dir = Path(__file__).resolve().parent     
    project_root = root_pkg_dir.parent                 

    for stage in PIPELINE:
        pkg_dir = root_pkg_dir / stage
        if not pkg_dir.is_dir():
            log(f"ERROR  Missing folder: {pkg_dir}")
            return 1
        if not (root_pkg_dir / "__init__.py").exists():
            log(f"ERROR  Package root missing __init__.py: {root_pkg_dir}")
            return 1
        if not (pkg_dir / "__init__.py").exists():
            log(f"ERROR  Package missing __init__.py: {pkg_dir}")
            return 1

    for stage in PIPELINE:
        stage_dir = root_pkg_dir / stage
        log(f"== Stage: {stage} ==")
        scripts = list(find_scripts(stage_dir))
        if not scripts:
            log(f"WARNING No scripts found in {stage_dir}")
            continue

        for script in scripts:
            if script.resolve() == Path(__file__).resolve():
                continue

            module_path = to_module_path(root_pkg_dir, script)
            code = run_module(module_path, cwd=project_root)
            if code != 0:
                log("Terminating pipeline due to error.")
                return code

    log("Pipeline completed successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

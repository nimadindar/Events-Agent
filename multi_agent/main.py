import glob
import subprocess
import os
import sys

# Timeout in seconds, or None for no timeout
timeout_seconds = None

def to_module(path: str) -> str:
    # Convert "multi_agent/ResearchTeam/arxiv_node.py" -> "multi_agent.ResearchTeam.arxiv_node"
    no_ext = os.path.splitext(path)[0]
    return no_ext.replace(os.sep, ".")

# Run ResearchTeam modules
input_files = sorted(glob.glob('multi_agent/ResearchTeam/*.py'))
for file in input_files:
    if os.path.basename(file) == '__init__.py':
        continue
    print(f"Running {file}...")
    try:
        subprocess.run([sys.executable, '-m', to_module(file)],
                       timeout=timeout_seconds, check=True, cwd='.')
        print(f"Finished {file}")
    except subprocess.TimeoutExpired:
        print(f"Timeout expired for {file}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {file}: {e}")

# Run PostingTeam modules
output_files = sorted(glob.glob('multi_agent/PostingTeam/*.py'))
for file in output_files:
    if os.path.basename(file) == '__init__.py':
        continue
    print(f"Running {file}...")
    try:
        subprocess.run([sys.executable, '-m', to_module(file)],
                       timeout=timeout_seconds, check=True, cwd='.')
        print(f"Finished {file}")
    except subprocess.TimeoutExpired:
        print(f"Timeout expired for {file}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {file}: {e}")

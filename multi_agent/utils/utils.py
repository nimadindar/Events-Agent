import re
import time
from pathlib import Path
from datetime import datetime

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.graph import MessagesState


class State(MessagesState):
    next: str


class DebugHandler(BaseCallbackHandler):
    def __init__(self, log_filename: str = "debug.log"):
        output_dir = Path("./saved")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file: Path = output_dir / log_filename
        self._llm_t0 = None
        self._tool_t0 = None

    def _ts(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def _log(self, message: str):
        line = f"[{self._ts()}] {message}"
        print(line)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def on_llm_start(self, serialized, prompts, **kwargs):
        self._llm_t0 = time.perf_counter()
        self._log("\n[LLM START]")
        for i, p in enumerate(prompts):
            self._log(f"Prompt {i}:\n{p}\n")

    def on_llm_end(self, response, **kwargs) -> None:
        elapsed = None
        if self._llm_t0 is not None:
            elapsed = time.perf_counter() - self._llm_t0
            self._llm_t0 = None
        usage = getattr(response, "llm_output", {})   
        if elapsed is not None:
            self._log(f"[LLM END] elapsed={elapsed:.3f}s | usage: {usage}")
        else:
            self._log(f"[LLM END] usage: {usage}")

    def on_agent_action(self, action, **kwargs):

        tool_name = getattr(action, "tool", "<unknown>")
        tool_input = getattr(action, "tool_input", "")

        preview = str(tool_input)
        if len(preview) > 500:
            preview = preview[:500] + " ..."
        self._log(f"[AGENT] Plan: call tool '{tool_name}' with input: {preview}")

    def on_agent_finish(self, finish, **kwargs):

        out = getattr(finish, "return_values", {})
        preview = str(out)
        if len(preview) > 500:
            preview = preview[:500] + " ..."
        self._log(f"[AGENT] Finished with: {preview}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        self._tool_t0 = time.perf_counter()
        self._log(f"\n[TOOL START] {serialized.get('name')}")
        self._log(f"Input: {input_str}")

    def on_tool_end(self, output, **kwargs):
        output_str = str(output)
        preview = output_str[:500]
        suffix = " ..." if len(output_str) > 500 else ""
        elapsed = None
        if self._tool_t0 is not None:
            elapsed = time.perf_counter() - self._tool_t0
            self._tool_t0 = None
        if elapsed is not None:
            self._log(f"[TOOL END] elapsed={elapsed:.3f}s | Output: {preview}{suffix}")
        else:
            self._log(f"[TOOL END] Output: {preview}{suffix}")

    def on_chain_start(self, serialized, inputs, **kwargs):
        name = None
        if isinstance(serialized, dict):
            name = serialized.get("name") or serialized.get("id")
        elif isinstance(serialized, str):
            name = serialized
        else:
            name = type(serialized).__name__
        try:
            keys = list(inputs.keys()) if isinstance(inputs, dict) else type(inputs).__name__
        except Exception:
            keys = "<unknown>"
        self._log(f"\n[CHAIN START] {name} | Inputs keys: {keys}")


    def on_chain_end(self, outputs, **kwargs):
        keys = list(outputs.keys()) if isinstance(outputs, dict) else type(outputs).__name__
        self._log(f"[CHAIN END] Outputs: {keys}")

 
def normalize_url(url):

    match = re.search(r"(\d{4}\.\d{4,5})(v\d)?", url)
    if match:
        return match.group(1)  
    return url


def parse_publish_date(d: str):

    if not isinstance(d, str):
        return datetime.min
    for fmt in ("%d-%m-%Y", "%d-%m-%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(d, fmt)
        except Exception:
            pass
    return datetime.min
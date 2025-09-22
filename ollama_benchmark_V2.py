#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, csv, datetime, json, os, platform, socket, subprocess, time
import psutil
from typing import Dict, Any, List, Optional

# =========================
# HW / System helpers
# =========================

def get_hw_info() -> Dict[str, Any]:
    cpu_model = platform.processor() or platform.machine() or "unknown"
    try:
        freq = psutil.cpu_freq().max if psutil.cpu_freq() else None
    except Exception:
        freq = None
    return {
        "cpu_model": cpu_model,
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "cpu_freq_MHz": freq,
        "mem_total_MB": psutil.virtual_memory().total // (1024**2),
        "swap_total_MB": psutil.swap_memory().total // (1024**2),
    }

def get_cpu_temp() -> Optional[float]:
    paths = [
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/class/hwmon/hwmon0/temp1_input",
    ]
    for p in paths:
        try:
            with open(p) as f:
                v = f.read().strip()
                milli = int(v)
                return milli / 1000.0
        except Exception:
            continue
    return None

def find_ollama_proc() -> Optional[psutil.Process]:
    candidates = []
    for p in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
        try:
            name = (p.info.get("name") or "").lower()
            cmd = " ".join(p.info.get("cmdline") or []).lower()
            if "ollama" in name or "ollama" in cmd:
                if "serve" in cmd or name == "ollama":
                    candidates.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if not candidates:
        return None
    best = candidates[0]
    best_cpu = -1.0
    for p in candidates:
        try:
            c = p.cpu_percent(interval=0.0)
            if c > best_cpu:
                best = p
                best_cpu = c
        except psutil.NoSuchProcess:
            continue
    return best

def sample_resources(root_proc: Optional[psutil.Process], prime: bool=False):
    """Return (cpu_percent_sum, mem_MB_sum) across the ollama serve process + children."""
    if root_proc is None:
        return 0.0, 0
    try:
        procs = [root_proc] + root_proc.children(recursive=True)
        cpu = 0.0
        mem = 0
        for p in procs:
            if not p.is_running():
                continue
            try:
                if prime:
                    _ = p.cpu_percent(None)
                    continue
                cpu += p.cpu_percent(None)
                mem += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if prime:
            return 0.0, 0
        return cpu, mem // (1024**2)
    except psutil.NoSuchProcess:
        return 0.0, 0

# =========================
# Swap / VM / Disk helpers
# =========================

def _read_meminfo_swap_used_mb() -> int:
    try:
        with open("/proc/meminfo") as f:
            kv = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    k = parts[0].rstrip(":")
                    v = parts[1]
                    if v.isdigit():
                        kv[k] = int(v)
        total = kv.get("SwapTotal", 0) // 1024
        free = kv.get("SwapFree", 0) // 1024
        return max(0, total - free)
    except Exception:
        return 0

def _read_vmstat_pswp():
    pswpin = pswpout = 0
    try:
        with open("/proc/vmstat") as f:
            for line in f:
                parts = line.split()
                if len(parts) != 2:
                    continue
                if parts[0] == "pswpin":
                    pswpin = int(parts[1])
                elif parts[0] == "pswpout":
                    pswpout = int(parts[1])
    except Exception:
        pass
    return pswpin, pswpout

def _disk_snapshot():
    snap = {}
    try:
        io = psutil.disk_io_counters(perdisk=True)
        for dev, d in io.items():
            snap[dev] = {
                "read_bytes": int(d.read_bytes),
                "write_bytes": int(d.write_bytes),
                "read_count": int(d.read_count),
                "write_count": int(d.write_count),
                "read_time_ms": int(getattr(d, "read_time", 0)),
                "write_time_ms": int(getattr(d, "write_time", 0)),
            }
    except Exception:
        pass
    return snap

def _disk_delta_mb(before, after):
    flat = {}
    total_read = 0
    total_write = 0
    for dev, a in after.items():
        b = before.get(dev, {"read_bytes":0,"write_bytes":0,"read_count":0,"write_count":0,"read_time_ms":0,"write_time_ms":0})
        d_read = max(0, a["read_bytes"] - b["read_bytes"])
        d_write = max(0, a["write_bytes"] - b["write_bytes"])
        total_read += d_read
        total_write += d_write
        flat[f"dev_{dev}_read_MB"] = d_read / (1024*1024)
        flat[f"dev_{dev}_write_MB"] = d_write / (1024*1024)
        flat[f"dev_{dev}_read_count"] = max(0, a["read_count"] - b["read_count"])
        flat[f"dev_{dev}_write_count"] = max(0, a["write_count"] - b["write_count"])
        flat[f"dev_{dev}_read_time_ms"] = max(0, a["read_time_ms"] - b["read_time_ms"])
        flat[f"dev_{dev}_write_time_ms"] = max(0, a["write_time_ms"] - b["write_time_ms"])
    flat["disk_read_MB_total"] = total_read / (1024*1024)
    flat["disk_write_MB_total"] = total_write / (1024*1024)
    return flat

# =========================
# Ollama helpers
# =========================

def get_model_size_bytes(model: str) -> Optional[int]:
    try:
        proc = subprocess.run(["ollama", "show", model, "--json"], capture_output=True, text=True, check=False)
        if proc.returncode == 0 and proc.stdout.strip():
            data = json.loads(proc.stdout)
            if isinstance(data, dict):
                if "size" in data and isinstance(data["size"], int):
                    return data["size"]
    except Exception:
        pass
    return None

# =========================
# Streaming run via API (curl) with error handling
# =========================

def run_prompt_via_api(model: str, prompt: str) -> Dict[str, Any]:
    url = "http://127.0.0.1:11434/api/generate"
    payload = json.dumps({"model": model, "prompt": prompt, "stream": True})
    cmd = ["curl", "-sS", "-N", "-H", "Content-Type: application/json", "-d", payload, url]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    ollama_proc = find_ollama_proc()
    _ = sample_resources(ollama_proc, prime=True)

    _pswpin0, _pswpout0 = _read_vmstat_pswp()
    _disk0 = _disk_snapshot()
    _swap_used_max = _read_meminfo_swap_used_mb()
    _iowait_samples: List[float] = []

    cpu_samples: List[float] = []
    mem_samples: List[int] = []
    temp_samples: List[float] = []
    response_text_chunks: List[str] = []

    t_start = time.time()
    first_token_time = None
    last_token_time = None
    eval_count = None
    eval_duration_ns = None

    for line in iter(proc.stdout.readline, ''):
        now = time.time()
        s = line.strip()
        if not s:
            cpu, mem = sample_resources(ollama_proc)
            cpu_samples.append(cpu)
            mem_samples.append(mem)
            t = get_cpu_temp()
            if t is not None:
                temp_samples.append(t)
            try:
                ct = psutil.cpu_times_percent(interval=0, percpu=False)
                _iowait_samples.append(getattr(ct, "iowait", 0.0))
            except Exception:
                pass
            try:
                su = _read_meminfo_swap_used_mb()
                if su > _swap_used_max:
                    _swap_used_max = su
            except Exception:
                pass
            continue

        try:
            data = json.loads(s)
        except json.JSONDecodeError:
            continue

        chunk = data.get("response", "")
        if chunk:
            if first_token_time is None:
                first_token_time = now
            last_token_time = now
            response_text_chunks.append(chunk)

        if data.get("done", False):
            eval_count = data.get("eval_count", eval_count)
            eval_duration_ns = data.get("eval_duration", eval_duration_ns)
            break

        cpu, mem = sample_resources(ollama_proc)
        cpu_samples.append(cpu)
        mem_samples.append(mem)
        t = get_cpu_temp()
        if t is not None:
            temp_samples.append(t)

    proc.stdout.close()
    stderr_data = ""
    try:
        stderr_data = proc.stderr.read().strip()
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
        return {"exit_code": 1, "response_text": "[ERROR] プロセスが強制終了しました"}

    # --- Error handling ---
    if proc.returncode != 0:
        msg = f"[ERROR] ollama 実行に失敗しました (exit={proc.returncode})."
        if "model not found" in stderr_data.lower() or "no such model" in stderr_data.lower():
            msg = f"[ERROR] モデル '{model}' が未pullです。先に以下を実行してください:\n  ollama pull {model}"
        print(msg)
        return {"exit_code": proc.returncode, "response_text": msg}

    # 応答が全くなかった場合
    if not response_text_chunks:
        msg = f"[ERROR] モデル '{model}' から応答が得られませんでした。未pullまたはロード失敗の可能性があります。\n  ollama pull {model}"
        print(msg)
        return {"exit_code": 1, "response_text": msg}

    # --- Normal case ---
    t_end = time.time()
    _pswpin1, _pswpout1 = _read_vmstat_pswp()
    _disk1 = _disk_snapshot()
    _pswpin_delta = max(0, _pswpin1 - _pswpin0)
    _pswpout_delta = max(0, _pswpout1 - _pswpout0)
    _disk_flat = _disk_delta_mb(_disk0, _disk1)
    _cpu_iowait_avg = (sum(_iowait_samples)/len(_iowait_samples)) if _iowait_samples else None
    _cpu_iowait_max = (max(_iowait_samples) if _iowait_samples else None)

    thinking_time = (first_token_time - t_start) if first_token_time else None
    if first_token_time and last_token_time:
        generation_time = max(0.0, last_token_time - first_token_time)
    else:
        generation_time = max(0.0, t_end - t_start)

    resp_text = "".join(response_text_chunks)
    if eval_count is not None and eval_duration_ns:
        tokens_generated = int(eval_count)
        gen_time_for_rate = max(generation_time, eval_duration_ns / 1e9)
        tokens_per_sec = tokens_generated / max(gen_time_for_rate, 1e-6)
    else:
        tokens_generated = len(resp_text) if resp_text else 0
        tokens_per_sec = (tokens_generated / generation_time) if generation_time > 0 else None

    return {
        "exit_code": 0,
        "thinking_time": thinking_time,
        "generation_time": generation_time,
        "response_text": resp_text,
        "tokens_generated": tokens_generated,
        "tokens_per_sec": tokens_per_sec,
        "cpu_percent_avg": (sum(cpu_samples)/len(cpu_samples)) if cpu_samples else None,
        "cpu_percent_max": (max(cpu_samples) if cpu_samples else None),
        "mem_MB": (max(mem_samples) if mem_samples else None),
        "cpu_temp_avg": (sum(temp_samples)/len(temp_samples)) if temp_samples else None,
        "cpu_temp_max": (max(temp_samples) if temp_samples else None),
        "swap_used_MB_max": _swap_used_max,
        "pswpin_delta": _pswpin_delta,
        "pswpout_delta": _pswpout_delta,
        "cpu_iowait_pct_avg": _cpu_iowait_avg,
        "cpu_iowait_pct_max": _cpu_iowait_max,
        **_disk_flat,
    }

# =========================
# One run wrapper
# =========================

def run_prompt(model: str, prompt: str, phase: str, run_id: int, hwinfo: Dict[str,Any]) -> Dict[str,Any]:
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    api_result = run_prompt_via_api(model, prompt)

    row: Dict[str,Any] = {
        "timestamp": start_time,
        "hostname": socket.gethostname(),
        "phase": phase,
        "model_name": model,
        "model_size_bytes": get_model_size_bytes(model),
        "prompt": prompt,
        "run_id": run_id,
        "exit_code": api_result.get("exit_code", 0),
        "thinking_time": api_result.get("thinking_time"),
        "generation_time": api_result.get("generation_time"),
        "tokens_generated": api_result.get("tokens_generated"),
        "tokens_per_sec": api_result.get("tokens_per_sec"),
        "cpu_percent_avg": api_result.get("cpu_percent_avg"),
        "cpu_percent_max": api_result.get("cpu_percent_max"),
        "mem_MB": api_result.get("mem_MB"),
        "cpu_temp_avg": api_result.get("cpu_temp_avg"),
        "cpu_temp_max": api_result.get("cpu_temp_max"),
        "num_threads": os.getenv("OLLAMA_NUM_THREADS", "unset"),
        "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "unset"),
        "response_text": api_result.get("response_text"),
        **hwinfo,
        "swap_used_MB_max": api_result.get("swap_used_MB_max"),
        "pswpin_delta": api_result.get("pswpin_delta"),
        "pswpout_delta": api_result.get("pswpout_delta"),
        "cpu_iowait_pct_avg": api_result.get("cpu_iowait_pct_avg"),
        "cpu_iowait_pct_max": api_result.get("cpu_iowait_pct_max"),
    }
    for k, v in api_result.items():
        if k.startswith("dev_") or k in ("disk_read_MB_total","disk_write_MB_total"):
            row[k] = v
    return row

# =========================
# CLI / Main
# =========================

def read_prompts(prompts_path: Optional[str]) -> List[str]:
    if not prompts_path:
        return ["こんにちは、今日の東京の天気はどうですか？"]
    lines: List[str] = []
    with open(prompts_path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                lines.append(s)
    return lines or ["こんにちは、今日の東京の天気はどうですか？"]

def main():
    parser = argparse.ArgumentParser(description="Ollama benchmark with swap/disk/iowait metrics (V2.2).")
    parser.add_argument("--model", required=True, help="Model name, e.g., gemma3:4b")
    parser.add_argument("--prompts", default=None, help="Path to prompts.txt (one prompt per line)")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per prompt (default: 3)")
    parser.add_argument("--output", default=None, help="Output CSV path (default auto)")
    parser.add_argument("--coldish", action="store_true", help="Use 'cold-ish' label for the first run")
    args = parser.parse_args()

    prompts = read_prompts(args.prompts)
    hwinfo = get_hw_info()

    if args.output:
        outfile = args.output
    else:
        outfile = f"ollama_benchmark_V2.2_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"

    with open(outfile, "w", newline="", encoding="utf-8-sig") as f:
        writer = None
        for prompt in prompts:
            for run_id in range(1, max(1, args.runs) + 1):
                phase = "cold-ish" if (args.coldish and run_id == 1) else ("cold" if run_id == 1 else "warm")
                row = run_prompt(args.model, prompt, phase, run_id, hwinfo)
                if writer is None:
                    fieldnames = list(row.keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                writer.writerow(row)

    print(f"結果を {outfile} に保存しました (UTF-8 BOM付き)")

if __name__ == "__main__":
    main()

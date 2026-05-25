import time
import requests
import csv
import os
import signal
import sys
import traceback
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

ENERGY_URL = os.getenv("ENERGY_URL")
metrics_file = os.environ.get("METRICS_CSV")

# Sampling interval (milliseconds) — default 1000ms (1s). Set METRICS_INTERVAL_MS=100 for 10Hz.
METRICS_INTERVAL_MS = float(os.environ.get("METRICS_INTERVAL_MS", "1000"))
SAMPLE_INTERVAL = max(METRICS_INTERVAL_MS, 1.0) / 1000.0  # seconds, clamp to >=1ms

# HTTP timeout tuned for high-rate scrapes: up to 2x the interval, bounded [0.2s, 5s]
HTTP_TIMEOUT = max(0.2, min(5.0, 2.0 * SAMPLE_INTERVAL))

def parse_value_from_line(line: str) -> float:
    parts = line.rsplit(None, 1)
    if len(parts) != 2:
        raise ValueError(f"Unexpected metric line format: {line!r}")
    value_str = parts[1].strip()
    return float(value_str)

def log(msg: str) -> None:
    ts = datetime.now().isoformat()
    print(f"[{ts}] {msg}", flush=True)

# Signal handler to ensure clean exit
def signal_handler(sig, frame):
    log(f"Metrics collection stopping due to signal {sig}")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Define which DCGM metrics we actually want to capture.
FIELD_NAMES = [
    "timestamp",
    "gpu_util",          # DCGM_FI_DEV_GPU_UTIL
    "fb_used",           # DCGM_FI_DEV_FB_USED
    "fb_free",           # DCGM_FI_DEV_FB_FREE
    "temperature",       # DCGM_FI_DEV_TEMPERATURE
    "power_usage",       # DCGM_FI_DEV_POWER_USAGE
    "energy_consumption" # DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION
]

if not ENERGY_URL:
    log("ERROR: ENERGY_URL is not set.")
    sys.exit(2)
if not metrics_file:
    log("ERROR: METRICS_CSV is not set.")
    sys.exit(2)

session = requests.Session()

def scrape_once():
    """
    Scrape /metrics and return (data_dict, missing_keys).
    Early‑exits once all desired metrics are found.
    """
    response = session.get(ENERGY_URL, timeout=HTTP_TIMEOUT)
    if response.status_code != 200:
        raise requests.RequestException(f"Non-200 from exporter: {response.status_code}")

    metrics_lines = response.text.splitlines()

    data = {
        "gpu_util": None,
        "fb_used": None,
        "fb_free": None,
        "temperature": None,
        "power_usage": None,
        "energy_consumption": None,
    }

    found = 0
    target = len(data)

    for line in metrics_lines:
        # Order roughly by likelihood/placement; adjust as you like
        if data["gpu_util"] is None and line.startswith("DCGM_FI_DEV_GPU_UTIL"):
            try:
                data["gpu_util"] = parse_value_from_line(line)
                found += 1
            except Exception:
                log(f"Failed parsing GPU_UTIL line: '{line}'")
        elif data["fb_used"] is None and line.startswith("DCGM_FI_DEV_FB_USED"):
            try:
                data["fb_used"] = parse_value_from_line(line)
                found += 1
            except Exception:
                log(f"Failed parsing FB_USED line: '{line}'")
        elif data["fb_free"] is None and line.startswith("DCGM_FI_DEV_FB_FREE"):
            try:
                data["fb_free"] = parse_value_from_line(line)
                found += 1
            except Exception:
                log(f"Failed parsing FB_FREE line: '{line}'")
        elif data["temperature"] is None and line.startswith("DCGM_FI_DEV_GPU_TEMP"):
            try:
                data["temperature"] = parse_value_from_line(line)
                found += 1
            except Exception:
                log(f"Failed parsing TEMPERATURE line: '{line}'")
        elif data["power_usage"] is None and line.startswith("DCGM_FI_DEV_POWER_USAGE"):
            try:
                data["power_usage"] = parse_value_from_line(line)
                found += 1
            except Exception:
                log(f"Failed parsing POWER_USAGE line: '{line}'")
        elif data["energy_consumption"] is None and line.startswith("DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION"):
            try:
                data["energy_consumption"] = parse_value_from_line(line)
                found += 1
            except Exception:
                log(f"Failed parsing ENERGY_CONSUMPTION line: '{line}'")

        if found == target:
            break

    missing = [k for k, v in data.items() if v is None]
    return data, missing

# Main loop (drift‑free scheduler)
with open(metrics_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(FIELD_NAMES)
    f.flush()

    log(f"Starting DCGM scrape: interval={SAMPLE_INTERVAL:.3f}s, timeout={HTTP_TIMEOUT:.3f}s, url={ENERGY_URL}")

    next_t = time.perf_counter()
    while True:
        try:
            data, missing = scrape_once()
            if missing:
                log(f"Some metrics missing in this iteration: {missing}")

            writer.writerow([
                datetime.now().isoformat(),
                data["gpu_util"],
                data["fb_used"],
                data["fb_free"],
                data["temperature"],
                data["power_usage"],
                data["energy_consumption"],
            ])
            f.flush()
        except requests.exceptions.RequestException as re:
            log(f"Request error when querying exporter: {re}")
        except Exception as e:
            log(f"Unexpected error in metrics loop: {e}")
            traceback.print_exc()

        # Drift‑free sleep to hit the target cadence
        next_t += SAMPLE_INTERVAL
        sleep_for = next_t - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            # We fell behind; resync to now to avoid accumulating delay
            next_t = time.perf_counter()

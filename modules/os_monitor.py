import time
import threading
import statistics
from pynput import keyboard
import pygetwindow as gw


class OSMonitor(threading.Thread):
    """
    Monitors OS-level activity:
      - Keystroke dynamics: dwell time + flight time analysis
      - Keystroke rate baseline detection (calibration → deviation alert)
      - Window switching detection
      - Suspicious typing speed (macro/copy-paste detection)
    """

    BASELINE_DURATION   = 30    # seconds to collect baseline
    DEVIATION_MULT      = 12.0  # flag if burst is >12× baseline mean (raised to allow heavy bursts)
    MAX_KEY_RATE        = 35    # hard cap (presses/sec) before baseline is ready
    FLAG_COOLDOWN       = 30.0  # seconds between duplicate flags (raised from 15s)

    # Dwell / flight dynamics thresholds — more tolerant to avoid false positives
    PASTE_FLIGHT_MAX    = 15    # ms — super fast flight (not slow pauses)
    MACRO_DWELL_STD_MAX = 3     # ms std-dev: too Robot-consistent = macro (was 5ms)
    MIN_DYNAMICS_SAMPLE = 40    # need at least this many events (was 20)
    PASTE_HIT_COUNT     = 4     # ≥ this many of the last 5 flights must be sub-threshold

    def __init__(self, callback):
        super().__init__()
        self.daemon   = True
        self.callback = callback
        self.running  = False

        self.allowed_window_title = None

        # Keystroke rate baseline
        self._key_times         = []
        self._burst_rates       = []
        self._baseline_start    = None
        self._baseline_ready    = False
        self._baseline_mean     = None
        self._baseline_std      = None
        self._last_flagged_ts   = 0

        # Keystroke dynamics (dwell + flight) — fed from browser via receive_keystroke_event
        self._dwell_times   = []   # key-hold durations in ms
        self._flight_times  = []   # pause-between-keys in ms
        self._dyn_lock      = threading.Lock()

        # Public status string read by /api/stats
        self.keystroke_status = "Normal"

    # ── Baseline helpers ──────────────────────────────────────────────────────
    def set_browser_baseline(self, key_count: int, duration: int):
        """
        Pre-load baseline from the browser check page.
        key_count: total keys pressed during baseline period
        duration:  seconds of collection
        """
        if duration > 0 and key_count > 0:
            mean = key_count / duration
            self._baseline_mean  = mean
            self._baseline_std   = max(mean * 0.3, 1.0)
            self._baseline_ready = True
            print(f"[OSMonitor] Browser baseline: {mean:.2f} kps ({key_count} keys/{duration}s)")

    def _record_burst(self, rate: float):
        """Called once per second to build/apply rate baseline."""
        elapsed = time.time() - self._baseline_start

        if not self._baseline_ready:
            if rate > 0:
                self._burst_rates.append(rate)
            if elapsed >= self.BASELINE_DURATION:
                if len(self._burst_rates) >= 5:
                    self._baseline_mean = statistics.mean(self._burst_rates)
                    self._baseline_std  = (
                        statistics.stdev(self._burst_rates)
                        if len(self._burst_rates) > 1 else 1.0
                    )
                    self._baseline_ready = True
                    print(f"[OSMonitor] Rate baseline: mean={self._baseline_mean:.1f}, std={self._baseline_std:.2f}")
                else:
                    self._baseline_ready = True
                    print("[OSMonitor] Not enough baseline data; using hard cap only.")
        else:
            if self._baseline_mean is not None:
                threshold = self._baseline_mean + self.DEVIATION_MULT * max(self._baseline_std, 1.0)
                if rate > threshold:
                    self._flag(f"Unusual typing speed ({rate:.0f} kps vs baseline {self._baseline_mean:.0f})")
            if rate > self.MAX_KEY_RATE:
                self._flag("Suspicious typing speed (possible macro/copy-paste)")

    def _flag(self, msg: str):
        now = time.time()
        if now - self._last_flagged_ts > self.FLAG_COOLDOWN:
            self._last_flagged_ts = now
            self.keystroke_status = f"⚠ {msg}"
            self.callback(msg)

    # ── Browser keystroke dynamics feed ──────────────────────────────────────
    def receive_keystroke_event(self, dwell_ms: float, flight_ms: float):
        """
        Called by the Flask route /api/report_keystroke.
        dwell_ms  : how long the key was held (keydown→keyup), ms
        flight_ms : pause since the previous keyup, ms (0 for first key)
        """
        with self._dyn_lock:
            if dwell_ms > 0:
                self._dwell_times.append(dwell_ms)
            if flight_ms > 0:
                self._flight_times.append(flight_ms)

            # Keep rolling window of last 60 events
            self._dwell_times  = self._dwell_times[-60:]
            self._flight_times = self._flight_times[-60:]

            n = len(self._dwell_times)
            if n >= self.MIN_DYNAMICS_SAMPLE:
                self._analyse_dynamics()

    def _analyse_dynamics(self):
        """
        Detect copy-paste (single huge flight) and macro / auto-type
        (suspiciously uniform dwell times).  Called under _dyn_lock.
        """
        # 1. Copy-paste / macro burst: require PASTE_HIT_COUNT of last 5 flights to be less than threshold
        if self._flight_times and len(self._flight_times) >= 5:
            recent_flights = self._flight_times[-5:]
            hit_count = sum(1 for f in recent_flights if f < self.PASTE_FLIGHT_MAX)
            if hit_count >= self.PASTE_HIT_COUNT:
                min_flight = min(recent_flights)
                self._flag(f"Possible macro burst detected ({hit_count}/5 gaps <{self.PASTE_FLIGHT_MAX}ms, min={min_flight:.0f}ms)")
                return

        # 2. Macro: dwell times suspiciously uniform
        if len(self._dwell_times) >= self.MIN_DYNAMICS_SAMPLE:
            std = statistics.stdev(self._dwell_times) if len(self._dwell_times) > 1 else 99
            if std < self.MACRO_DWELL_STD_MAX:
                self._flag(f"Possible macro/auto-type detected (dwell std={std:.1f} ms)")
                return

        # All clear
        self.keystroke_status = "Normal"

    # ── Global keyboard listener (rate tracking only) ─────────────────────────
    def on_press(self, key):
        now = time.time()
        self._key_times.append(now)
        self._key_times = [t for t in self._key_times if now - t <= 1.0]

    # ── Window checker ────────────────────────────────────────────────────────
    def check_active_window(self):
        try:
            active = gw.getActiveWindow()
            if active and self.allowed_window_title:
                title = active.title
                if self.allowed_window_title not in title:
                    self.callback(f"Window switched: '{title}'")
        except Exception as e:
            print(f"[OSMonitor] Window check error: {e}")

    # ── Thread run ────────────────────────────────────────────────────────────
    def run(self):
        self.running = True
        self._baseline_start = time.time()

        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

        time.sleep(2)
        try:
            active = gw.getActiveWindow()
            if active:
                self.allowed_window_title = active.title
                print(f"[OSMonitor] Locked onto window: {self.allowed_window_title}")
        except Exception:
            pass

        last_burst_check = time.time()

        while self.running:
            now = time.time()
            if now - last_burst_check >= 1.0:
                burst = len([t for t in self._key_times if now - t <= 1.0])
                self._record_burst(float(burst))
                last_burst_check = now

            self.check_active_window()
            time.sleep(1.0)

    def stop(self):
        self.running = False
        if hasattr(self, 'listener'):
            self.listener.stop()

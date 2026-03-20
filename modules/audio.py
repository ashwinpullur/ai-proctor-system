import pyaudio
import numpy as np
import threading
import time


class AudioMonitor(threading.Thread):
    def __init__(self, callback):
        super(AudioMonitor, self).__init__()
        self.daemon = True
        self.callback = callback
        self.CHUNK   = 1024
        self.FORMAT  = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE    = 44100
        self.running = False

        # Audio threshold for "talking" or loud noise
        self.VOLUME_THRESHOLD         = 500
        self.loud_start_time          = None
        self.LOUD_DURATION_THRESHOLD_SEC = 2.0

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _rms(data: bytes) -> float:
        """Return the RMS amplitude of a raw int16 PCM buffer."""
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float64)
        # cast first, then square — np.square does NOT accept a dtype kwarg
        return float(np.sqrt(np.mean(samples ** 2)))

    def _open_stream(self, pa: pyaudio.PyAudio):
        """Open (or re-open) the microphone input stream."""
        return pa.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
        )

    # ── thread entry ──────────────────────────────────────────────────────────
    def run(self):
        self.running = True
        pa     = pyaudio.PyAudio()
        stream = None

        try:
            stream = self._open_stream(pa)
        except Exception as e:
            print(f"[AudioMonitor] Could not open microphone: {e}")
            self.running = False
            pa.terminate()
            return

        while self.running:
            try:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                rms  = self._rms(data)

                if rms > self.VOLUME_THRESHOLD:
                    if self.loud_start_time is None:
                        self.loud_start_time = time.time()
                    elif (time.time() - self.loud_start_time) > self.LOUD_DURATION_THRESHOLD_SEC:
                        self.callback("Suspicious audio detected (Talking/Noise)")
                        self.loud_start_time = time.time()   # reset so it fires ~every 2 s not every frame
                else:
                    self.loud_start_time = None

                time.sleep(0.01)

            except OSError as e:
                # Device disconnected / buffer overflow — try to re-open
                print(f"[AudioMonitor] Stream error ({e}), re-opening…")
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
                time.sleep(1.0)
                try:
                    stream = self._open_stream(pa)
                except Exception as re_err:
                    print(f"[AudioMonitor] Re-open failed: {re_err}")
                    break

            except Exception as e:
                print(f"[AudioMonitor] Unexpected error: {e}")
                break

        # Cleanup
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        pa.terminate()
        self.running = False

    def stop(self):
        self.running = False


# Testing block
if __name__ == "__main__":
    def dummy_callback(msg):
        print(f"ALERT: {msg}")

    monitor = AudioMonitor(callback=dummy_callback)
    monitor.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop()

import time
import torch


class Timer:
    """
    Simple timer with optional device synchronization.
    """

    def __init__(self):
        self.timers_start = []

    def start(self):
        self.timers_start.append(time.perf_counter())

    def stop(self, tag=None):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            torch.mps.synchronize()
        end_t = time.perf_counter()
        start_t = self.timers_start.pop()
        tag = f"{tag}: " if tag else ""
        elapsed_time_s = end_t - start_t
        print(f"{tag}Elapsed {elapsed_time_s}s")
        return elapsed_time_s


_global_timer = Timer()
tic = _global_timer.start
toc = _global_timer.stop

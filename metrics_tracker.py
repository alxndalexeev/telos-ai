import time
from collections import defaultdict
from functools import wraps
from threading import Lock


class MetricsTracker:
    def __init__(self):
        self.lock = Lock()
        self.metrics = defaultdict(lambda: {
            'calls': 0,
            'errors': 0,
            'total_time': 0.0,
            'last_call': None,
        })

    def track(self, func):
        """Decorator to track the usage and performance of a function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            fname = func.__qualname__
            start_time = time.perf_counter()
            with self.lock:
                self.metrics[fname]['calls'] += 1
                self.metrics[fname]['last_call'] = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            except Exception:
                with self.lock:
                    self.metrics[fname]['errors'] += 1
                raise
            finally:
                elapsed = time.perf_counter() - start_time
                with self.lock:
                    self.metrics[fname]['total_time'] += elapsed
        return wrapper

    def get_metrics(self):
        """Return a copy of the current metrics."""
        with self.lock:
            return {k: v.copy() for k, v in self.metrics.items()}

    def print_metrics(self):
        """Prints the collected metrics in a readable format."""
        with self.lock:
            for fname, data in self.metrics.items():
                avg_time = (data['total_time'] / data['calls']) if data['calls'] else 0
                print(f"Function: {fname}")
                print(f"  Calls: {data['calls']}")
                print(f"  Errors: {data['errors']}")
                print(f"  Total Time: {data['total_time']:.6f}s")
                print(f"  Average Time: {avg_time:.6f}s")
                print(f"  Last Call: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data['last_call'])) if data['last_call'] else 'Never'}")
                print("-" * 40)


# Example usage

metrics = MetricsTracker()

@metrics.track
def core_functionality_a(x):
    time.sleep(0.1)
    return x * 2

@metrics.track
def core_functionality_b(y):
    time.sleep(0.05)
    if y < 0:
        raise ValueError("Negative value!")
    return y + 10

if __name__ == "__main__":
    for i in range(5):
        core_functionality_a(i)
        try:
            core_functionality_b(i - 2)
        except Exception:
            pass
    metrics.print_metrics()
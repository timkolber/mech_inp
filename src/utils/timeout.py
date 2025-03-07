import signal
from typing import Optional


class TimeoutException(Exception):
    pass

def timeout_handler(signum: int, frame: Optional[object]) -> None:
    raise TimeoutException()

def setup_loop_timeout(timeout: int) -> None:
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

def deactivate_loop_timeout() -> None:
    signal.alarm(0)
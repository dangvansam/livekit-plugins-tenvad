import os
import platform
import threading
import queue
from ctypes import CDLL, POINTER, byref, c_float, c_int, c_int32, c_size_t, c_void_p

import numpy as np
from .log import logger

SUPPORTED_SAMPLE_RATES = [16000]

_GLOBAL_VAD_LOCK = threading.Lock()
_VAD_THREAD = None
_VAD_QUEUE = None
_VAD_RESULT_QUEUES = {}


def _vad_worker_thread():
    """Dedicated thread for all VAD operations"""
    global _VAD_QUEUE
    logger.info(f"VAD Worker Thread Started: pid={os.getpid()} thread_id={threading.get_ident()}")

    while True:
        try:
            task = _VAD_QUEUE.get()
            if task is None:  # Shutdown signal
                break

            task_id, func, args, kwargs = task
            try:
                result = func(*args, **kwargs)
                _VAD_RESULT_QUEUES[task_id].put(("success", result))
            except Exception as e:
                _VAD_RESULT_QUEUES[task_id].put(("error", e))
        except Exception as e:
            logger.error(f"VAD worker thread error: {e}")


def _ensure_vad_thread():
    """Ensure the VAD worker thread is running"""
    global _VAD_THREAD, _VAD_QUEUE

    if _VAD_THREAD is None or not _VAD_THREAD.is_alive():
        _VAD_QUEUE = queue.Queue()
        _VAD_THREAD = threading.Thread(target=_vad_worker_thread, daemon=True, name="VADWorker")
        _VAD_THREAD.start()


def _execute_on_vad_thread(func, *args, **kwargs):
    """Execute a function on the dedicated VAD thread"""
    global _VAD_QUEUE, _VAD_RESULT_QUEUES

    _ensure_vad_thread()

    task_id = threading.get_ident()
    result_queue = queue.Queue()
    _VAD_RESULT_QUEUES[task_id] = result_queue

    _VAD_QUEUE.put((task_id, func, args, kwargs))

    status, result = result_queue.get()
    del _VAD_RESULT_QUEUES[task_id]

    if status == "error":
        raise result
    return result


class TenVad:
    """TEN VAD implementation loaded from native library"""

    def __init__(self, hop_size: int = 256, threshold: float = 0.5):
        self.hop_size = hop_size
        self.threshold = threshold
        self._lock = threading.Lock()
        self._destroyed = False

        base_dir = os.path.join(os.path.dirname(__file__))

        if platform.system() == "Linux" and platform.machine() == "x86_64":
            git_path = os.path.join(base_dir, "prebuilt/Linux/x64/libten_vad")

        elif platform.system() == "Darwin":
            git_path = os.path.join(base_dir, "prebuilt/macOS/ten_vad.framework/Versions/A/ten_vad")

        elif platform.system().upper() == "WINDOWS":
            if platform.machine().upper() in ["X64", "X86_64", "AMD64"]:
                git_path = os.path.join(base_dir, "prebuilt/Windows/x64/ten_vad.dll")
            else:
                git_path = os.path.join(base_dir, "prebuilt/Windows/x86/ten_vad.dll")
        else:
            raise NotImplementedError(
                f"Unsupported platform: {platform.system()} {platform.machine()}"
            )

        self.vad_library = CDLL(git_path)

        self.vad_handler = c_void_p(0)
        self.out_probability = c_float()
        self.out_flags = c_int32()

        self.vad_library.ten_vad_create.argtypes = [
            POINTER(c_void_p),
            c_size_t,
            c_float,
        ]
        self.vad_library.ten_vad_create.restype = c_int

        self.vad_library.ten_vad_destroy.argtypes = [POINTER(c_void_p)]
        self.vad_library.ten_vad_destroy.restype = c_int

        self.vad_library.ten_vad_process.argtypes = [
            c_void_p,
            c_void_p,
            c_size_t,
            POINTER(c_float),
            POINTER(c_int32),
        ]
        self.vad_library.ten_vad_process.restype = c_int

        self.create_and_init_handler()

    def _create_handler_internal(self):
        """Internal method to create handler - must run on VAD thread"""
        logger.info(f"create_and_init_handler: pid={os.getpid()} thread_id={threading.get_ident()}")

        with _GLOBAL_VAD_LOCK:
            assert (
                self.vad_library.ten_vad_create(
                    byref(self.vad_handler),
                    c_size_t(self.hop_size),
                    c_float(self.threshold),
                )
                == 0
            ), "[TEN VAD]: create handler failure!"

    def create_and_init_handler(self):
        _execute_on_vad_thread(self._create_handler_internal)

    def __del__(self):
        try:
            if hasattr(self, "_destroyed") and hasattr(self, "vad_handler"):
                with _GLOBAL_VAD_LOCK:
                    if not self._destroyed and self.vad_handler and self.vad_handler.value:
                        try:
                            self.vad_library.ten_vad_destroy(byref(self.vad_handler))
                        except:
                            pass
                        finally:
                            self._destroyed = True
                            if hasattr(self, "vad_handler"):
                                self.vad_handler.value = 0
        except:
            pass

    def get_input_data(self, audio_data: np.ndarray):
        audio_data = np.squeeze(audio_data)
        assert len(audio_data.shape) == 1 and audio_data.shape[0] == self.hop_size, (
            f"[TEN VAD]: audio data shape should be [{self.hop_size}]"
        )
        assert audio_data.dtype == np.int16, "[TEN VAD]: audio data type error, must be int16"
        data_pointer = audio_data.__array_interface__["data"][0]
        return c_void_p(data_pointer)

    def _update_threshold_internal(self, threshold: float):
        """Internal method to update threshold - must run on VAD thread"""
        with _GLOBAL_VAD_LOCK:
            # Destroy existing handler
            if self.vad_handler and self.vad_handler.value:
                self.vad_library.ten_vad_destroy(byref(self.vad_handler))
                self.vad_handler.value = 0

            # Update threshold and recreate
            self.threshold = threshold
            self._destroyed = False
            assert (
                self.vad_library.ten_vad_create(
                    byref(self.vad_handler),
                    c_size_t(self.hop_size),
                    c_float(self.threshold),
                )
                == 0
            ), "[TEN VAD]: create handler failure!"

    def update_threshold(self, threshold: float):
        """Update threshold by recreating the handler on the VAD thread"""
        _execute_on_vad_thread(self._update_threshold_internal, threshold)

    def _process_internal(self, audio_data: np.ndarray):
        """Internal method to process audio - must run on VAD thread"""
        
        # logger.info(f"process: pid={os.getpid()} thread_id={threading.get_ident()}")
        
        if self._destroyed:
            raise RuntimeError("VAD instance has been destroyed")
        with _GLOBAL_VAD_LOCK:
            input_pointer = self.get_input_data(audio_data)
            self.vad_library.ten_vad_process(
                self.vad_handler,
                input_pointer,
                c_size_t(self.hop_size),
                byref(self.out_probability),
                byref(self.out_flags),
            )
            return self.out_probability.value, self.out_flags.value

    def process(self, audio_data: np.ndarray):
        """Process audio on the dedicated VAD thread"""
        return _execute_on_vad_thread(self._process_internal, audio_data)


class OnnxModel:
    """Wrapper around TenVad to maintain compatibility with existing VAD interface"""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        activation_threshold: float = 0.5,
    ) -> None:
        self._sample_rate = sample_rate

        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError("TEN VAD only supports 8KHz and 16KHz sample rates")

        self._window_size_samples = 256
        self._ten_vad = TenVad(hop_size=self._window_size_samples, threshold=activation_threshold)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def window_size_samples(self) -> int:
        return self._window_size_samples

    @property
    def context_size(self) -> int:
        return self._context_size

    def update_threshold(self, threshold: float) -> None:
        self._ten_vad.update_threshold(threshold)

    def __call__(self, x: np.ndarray) -> float:
        probability, _ = self._ten_vad.process(x)
        return probability

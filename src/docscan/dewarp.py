from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from .common import DocscanError, elapsed_ms
from .models import DEWARP_MODELS, configure_paddle_cache, model_hashes, model_status


class Dewarper:
    def __init__(self, config: dict):
        self.root = Path(config["model_dir"]).resolve()
        self.device = config["device"]
        self.threads = config["cpu_threads"]
        self.metadata = {"model": DEWARP_MODELS["dewarp"], "device": self.device, "status": "not_loaded"}
        self._predictor = None
        self._failure = None

    def _load(self) -> None:
        if self._predictor is not None:
            return
        if self._failure is not None:
            raise self._failure
        start = perf_counter()
        model = model_status(self.root, DEWARP_MODELS)["dewarp"]
        try:
            if not model["ready"]:
                raise DocscanError("dewarp_model_missing", "Run docscan prepare-models --dewarp to install UVDoc")
            configure_paddle_cache(self.root)
            from paddleocr import TextImageUnwarping

            predictor = TextImageUnwarping(model_name=model["name"], model_dir=model["directory"],
                                           device=self.device, enable_mkldnn=False, cpu_threads=self.threads)
            weights = model_hashes(self.root, DEWARP_MODELS)["dewarp"]
            self._predictor = predictor
            self.metadata.update(status="ok", load_ms=elapsed_ms(start), weights=weights)
        except Exception as error:
            self._failure = error if isinstance(error, DocscanError) else DocscanError("dewarp_model_failed", str(error))
            self.metadata.update(status="error", load_ms=elapsed_ms(start), error=str(self._failure))
            raise self._failure

    def apply(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        self._load()
        start = perf_counter()
        try:
            results = self._predictor.predict(image, batch_size=1)
            if len(results) != 1:
                raise ValueError("UVDoc did not return one page")
            result = np.asarray(results[0]["doctr_img"])
            if result.ndim != 3 or result.shape[2] != 3 or min(result.shape[:2]) < 3 or result.dtype != np.uint8:
                raise ValueError("Unexpected image shape or data type from UVDoc")
        except Exception as error:
            raise DocscanError("dewarp_inference_failed", str(error)) from error
        return result, {"status": "ok", "model": self.metadata["model"], "elapsed_ms": elapsed_ms(start),
                        "input_size": [image.shape[1], image.shape[0]], "output_size": [result.shape[1], result.shape[0]]}

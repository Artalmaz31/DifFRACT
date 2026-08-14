from typing import List, Optional
from datasets import load_dataset


class PromptStream:
    """Endless stream of prompts from a HuggingFace dataset, restarting on exhaustion."""

    def __init__(
        self,
        dataset_id: str,
        column: str = "prompt",
        split: str = "train",
        config_name: Optional[str] = "default",
        min_len: int = 16,
        max_len: int = 512,
    ):
        self.dataset_id = dataset_id
        self.column = column
        self.min_len = min_len
        self.max_len = max_len
        self.dataset = load_dataset(dataset_id, config_name, split=split)
        self._iter = iter(self.dataset)

    def get_prompts(self, n: int) -> List[str]:
        out: List[str] = []
        while len(out) < n:
            try:
                item = next(self._iter)
            except StopIteration:
                self._iter = iter(self.dataset)
                continue
            txt = item.get(self.column, "")
            if txt and len(txt) > self.min_len:
                out.append(txt[: self.max_len])
        return out

    def fixed_validation_batch(self, n: int = 512) -> List[str]:
        out: List[str] = []
        for item in self.dataset:
            if len(out) >= n:
                break
            txt = item.get(self.column, "")
            if txt and len(txt) > self.min_len:
                out.append(txt[: self.max_len])
        return out

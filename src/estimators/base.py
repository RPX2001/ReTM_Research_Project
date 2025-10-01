from abc import ABC, abstractmethod
import torch

class RTMEstimator(ABC):
    @abstractmethod
    def fit(self, source_stft: torch.Tensor, target_stft: torch.Tensor, **kwargs):
        pass
    @abstractmethod
    def predict(self, source_stft: torch.Tensor) -> torch.Tensor:
        pass
    @abstractmethod
    def get_rtm(self) -> torch.Tensor:
        pass
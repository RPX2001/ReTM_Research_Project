import torch
import numpy as np
from .base import RTMEstimator

class ClosedFormRTM(RTMEstimator):
    """Frequency-wise ridge regression estimator.
    Expects source_stft: (batch, freq, time, src_ch) complex
            target_stft: (batch, freq, time, tgt_ch) complex
    Produces rtm: (batch, freq, tgt_ch, src_ch) complex
    """
    def __init__(self, reg=1e-3):
        self.reg = reg
        self.rtm = None

    def fit(self, source_stft, target_stft, mask=None):
        b, F, T, Ns = source_stft.shape
        _, _, _, Nt = target_stft.shape
        self.rtm = torch.zeros((b, F, Nt, Ns), dtype=source_stft.dtype, device=source_stft.device)
        for bi in range(b):
            for f in range(F):
                X = source_stft[bi,f,:,:]  # (T, Ns) if transposed; ensure shape
                Y = target_stft[bi,f,:,:]
                # shape: (T, Ns) and (T, Nt)
                # solve G = (X^H X + reg I)^{-1} X^H Y
                # reshape to matrices
                X_mat = X.view(T, Ns)
                Y_mat = Y.view(T, Nt)
                XtX = X_mat.conj().T @ X_mat
                regm = self.reg * torch.eye(XtX.shape[0], device=XtX.device, dtype=XtX.dtype)
                try:
                    G = torch.linalg.solve(XtX + regm, X_mat.conj().T @ Y_mat)
                except RuntimeError:
                    G = torch.pinverse(XtX + regm) @ (X_mat.conj().T @ Y_mat)
                self.rtm[bi,f,:,:] = G.T
        return self.rtm

    def predict(self, source_stft):
        # apply rtm: out[bi,f,t,nt] = sum_k rtm[bi,f,nt,k] * source_stft[bi,f,t,k]
        b,F,T,Ns = source_stft.shape
        Nt = self.rtm.shape[2]
        out = torch.einsum('bfnk,bftk->bftn', self.rtm, source_stft)
        return out

    def get_rtm(self):
        return self.rtm
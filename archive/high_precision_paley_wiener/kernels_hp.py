"""
Arbitrary-precision (mpmath) prolate spectrum for the band-limited Paley-Wiener kernel -- the
hp extension of kernels.PaleyWienerSincKernel, ARCHIVED (see README.md).

PaleyWienerSincKernelHP subclasses the live float64 kernel and adds the two members that
widths_hp.gelfand_widths_hp needs: kfun_mp (arbitrary-precision K) and prolate_spectrum_hp
(the Slepian spectrum in mpmath).  Everything else -- eval, eval_grad, diagonal, singular_numbers,
the greedy grid cache -- is inherited unchanged, so this class also drives the float64 half of the
comparison figure.
"""

from __future__ import annotations
from mpmath import mp, mpf, matrix, sin, pi, sqrt as msqrt
from numpy.polynomial.legendre import leggauss

from kernels import PaleyWienerSincKernel


def _gauss_legendre_hp(M: int, dps: int):
    """Gauss-Legendre nodes/weights on [-1,1] to `dps` digits: float64 seed + mpmath Newton."""
    mp.dps = dps
    seed, _ = leggauss(M)
    nodes, wts = [], []
    tol = mpf(10) ** (-(dps - 3))
    for x0 in seed:
        x = mpf(float(x0))
        for _ in range(60):
            p0, p1 = mpf(1), x
            for k in range(1, M):
                p0, p1 = p1, ((2 * k + 1) * x * p1 - k * p0) / (k + 1)
            dp = M * (x * p1 - p0) / (x * x - 1)  # P_M'(x)
            dx = p1 / dp
            x -= dx
            if abs(dx) < tol:
                break
        p0, p1 = mpf(1), x
        for k in range(1, M):
            p0, p1 = p1, ((2 * k + 1) * x * p1 - k * p0) / (k + 1)
        dp = M * (x * p1 - p0) / (x * x - 1)
        nodes.append(x)
        wts.append(2 / ((1 - x * x) * dp * dp))
    return nodes, wts


class PaleyWienerSincKernelHP(PaleyWienerSincKernel):
    """Band-limited sinc kernel with the arbitrary-precision prolate members added (kfun_mp,
    prolate_spectrum_hp).  Drop-in for PaleyWienerSincKernel; also carries the float64 interface
    it inherits, so one instance serves both halves of the archived comparison figure.
    """

    def kfun_mp(self, u):
        """K_c(u) = sin(c u)/(pi u) in ARBITRARY PRECISION (mpmath); u and result are mpf, K(0)=c/pi.
        The high-precision sibling of _kfun -- prolate_spectrum_hp / gelfand_widths_hp use it to
        recover the widths past the float64 cliff (mp.dps sets the working precision).  c is the exact
        float64 c (matches the float64 grid), cached as a dps-independent mpf so evals don't reconvert.
        """
        c = getattr(self, "_c_mp", None)
        if c is None:
            c = self._c_mp = mpf(self.c)
        return c / pi if u == 0 else sin(c * u) / (pi * u)

    def prolate_spectrum_hp(self, M: int, dps: int = 50):
        """Slepian prolate spectrum of the L2([-1,1]) operator T_c in mpmath -- the arbitrary-
        precision counterpart of singular_numbers, consumed by gelfand_widths_hp past the float64
        cliff.  Gauss-Legendre Nystrom A = W^1/2 K W^1/2 (K from kfun_mp).  Returns
            lam  : eigenvalues (descending, list of mpf) -- flat ~1 then super-exponential cliff,
            t, w : Gauss-Legendre nodes / weights (mpf),
            V    : V[k][j] = the k-th (descending) eigenvector of A at node j.
        phi_k(t_j) = V[k][j]/sqrt(w_j) are the L2([-1,1])-orthonormal prolate functions on the nodes.
        """
        mp.dps = dps
        t, w = _gauss_legendre_hp(M, dps)
        sw = [msqrt(wi) for wi in w]
        A = matrix(M, M)
        for i in range(M):
            for j in range(i, M):
                v = sw[i] * sw[j] * self.kfun_mp(t[i] - t[j])  # kfun_mp(0)=K(x,x)=c/pi
                A[i, j] = v
                A[j, i] = v
        E, Q = mp.eigsy(A)  # ascending eigenvalues, eigenvectors as columns
        lam = [E[i] for i in range(M)][::-1]  # descending
        V = [[Q[j, M - 1 - k] for j in range(M)] for k in range(M)]
        return lam, t, w, V

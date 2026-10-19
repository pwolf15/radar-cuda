import pytest

from radar.params import RadarParams


@pytest.fixture
def p():
    return RadarParams(
        fc=77e9, slope=30e12, Tc=40e-6, fs=10e6, n_samples=400, n_chirps=128, n_channels=8
    )


@pytest.fixture
def expected_cells(p):
    def _f(targets):
        return {
            (p.n_chirps // 2 + round(v / p.velocity_resolution), round(R / p.range_resolution)): (
                R,
                v,
                theta,
            )
            for R, v, theta, _ in targets
        }

    return _f

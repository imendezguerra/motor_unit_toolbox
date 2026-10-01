import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from motor_unit_toolbox.utils import binary_to_firings, firings_to_binary


def test_firings_to_binary():
    firings = [np.array([0, 3, 10]), np.array([], dtype=int)]
    binary = firings_to_binary(firings, signal_length=6)

    assert binary.shape == (6, 2)
    assert binary.dtype == bool
    np.testing.assert_array_equal(np.nonzero(binary[:, 0])[0], [0, 3])  # 10 is dropped
    assert not binary[:, 1].any()


@settings(max_examples=50, deadline=None)
@given(signal_length=st.integers(min_value=1, max_value=200), data=st.data())
def test_round_trip(signal_length, data):
    n_units = data.draw(st.integers(min_value=0, max_value=5))
    firings = [
        np.array(sorted(data.draw(st.sets(st.integers(0, signal_length - 1)))), dtype=int)
        for _ in range(n_units)
    ]
    recovered = binary_to_firings(firings_to_binary(firings, signal_length))
    assert len(recovered) == n_units
    for orig, rec in zip(firings, recovered):
        np.testing.assert_array_equal(orig, rec)

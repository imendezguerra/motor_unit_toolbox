import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from motor_unit_toolbox.utils import binary_to_firings, firings_to_binary


def test_firings_to_binary_shape_and_values():
    firings = [np.array([0, 3, 5]), np.array([1, 2])]
    binary = firings_to_binary(firings, signal_length=6)

    assert binary.shape == (6, 2)
    assert binary.dtype == bool
    np.testing.assert_array_equal(np.nonzero(binary[:, 0])[0], [0, 3, 5])
    np.testing.assert_array_equal(np.nonzero(binary[:, 1])[0], [1, 2])


def test_firings_beyond_signal_length_are_dropped():
    binary = firings_to_binary([np.array([1, 4, 10, 20])], signal_length=5)
    np.testing.assert_array_equal(np.nonzero(binary[:, 0])[0], [1, 4])


def test_empty_unit():
    binary = firings_to_binary([np.array([], dtype=int), np.array([2])], signal_length=4)
    assert not binary[:, 0].any()
    assert binary[2, 1]


def test_binary_to_firings():
    binary = np.zeros((5, 2), dtype=bool)
    binary[[0, 4], 0] = True
    firings = binary_to_firings(binary)
    assert len(firings) == 2
    np.testing.assert_array_equal(firings[0], [0, 4])
    assert firings[1].size == 0


@settings(max_examples=50, deadline=None)
@given(
    signal_length=st.integers(min_value=1, max_value=200),
    data=st.data(),
)
def test_round_trip(signal_length, data):
    n_units = data.draw(st.integers(min_value=0, max_value=5))
    firings = [
        np.array(
            sorted(data.draw(st.sets(st.integers(0, signal_length - 1), max_size=signal_length))),
            dtype=int,
        )
        for _ in range(n_units)
    ]
    recovered = binary_to_firings(firings_to_binary(firings, signal_length))
    assert len(recovered) == n_units
    for orig, rec in zip(firings, recovered):
        np.testing.assert_array_equal(orig, rec)

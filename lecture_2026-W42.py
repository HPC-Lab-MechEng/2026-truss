import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np

    return (np,)


@app.cell
def _(np):
    #
    # input data structures
    #
    xy = np.array([[0, 0], [1, 0], [0.34, 0.5345]])
    inc = np.array([[0, 1], [1, 2], [2, 0]])
    k = np.array([1.0, 10.0, 100.0])
    return inc, k, xy


@app.cell
def _(np):
    def kmat(inc, xy, k):
        """compute ..."""

        assert np.shape(inc) == (2,)
        assert np.isscalar(k)

        c = xy[inc[1]] - xy[inc[0]]
        c /= np.linalg.norm(c)
        rot = np.zeros((2, 2))
        rot[0] = c
        rot[1] = [-c[1], c[0]]
        np.testing.assert_array_almost_equal_nulp(
            rot @ rot.mT, np.eye(2), nulp=10
        )

        d = np.shape(xy)[1]
        K = np.zeros((2 * d, 2 * d))

        return K

    return (kmat,)


@app.cell
def _(inc, k, kmat, xy):
    for _i in range(len(inc)):
        _ki = kmat(inc[_i], xy, k[_i])
        print(f"elm {_i}")
        print(_ki)
    return


if __name__ == "__main__":
    app.run()

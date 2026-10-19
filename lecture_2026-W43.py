import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np

    return (np,)


@app.cell
def _(np):
    # espilon for tolerances
    eps = np.finfo(np.double).eps
    return (eps,)


@app.cell
def _(np):
    #
    # input data structures
    #
    xy = np.array([[0, 0], [1, 0], [0.25, 0.5671]])
    inc = np.array([[0, 1], [1, 2], [2, 0]])
    k = np.array([1.0, 10.0, 100.0])
    return inc, k, xy


@app.cell
def _(inc, np, xy):
    #
    # N : total number of nodes
    # M : total number of elements
    # D : number of spatial dimensions
    # NxElm = : number of nodes per element
    #
    N, D = np.shape(xy)
    M, NxElm = np.shape(inc)
    DOF = N * D
    return D, DOF, N, NxElm


@app.cell
def _(D, DOF, np, xy):
    #
    # create rigid body modes
    #
    n_rigid = D * (D + 1) // 2
    u_rigid = np.zeros((DOF, n_rigid))

    for _i in range(D):
        u_rigid[_i::D, _i] = 1

    match D:
        case 1:
            # no rotiations
            pass
        case 2:
            # 1 rotation
            u_rigid[::D, -1] = -xy[:, 1]
            u_rigid[1::D, -1] = xy[:, 0]
        case 3:
            # 3 rotations, implemented using np.cross
            _e = np.identity(D)
            for _i in range(D):
                u_rigid[:, D + _i] = np.cross(_e[_i], xy).reshape((-1,))
        case _:
            # excercise: how to implement the general case?
            raise NotImplementedError(f"Rotations not implemented for {D}D.")

    # check computed rigid body modes are full rank
    assert np.linalg.matrix_rank(u_rigid) == min(DOF, n_rigid), (
        "Computed rigid body motions are not full rank"
    )
    return n_rigid, u_rigid


@app.cell
def _(D, NxElm, np):
    def kmat(inc, xy, k):
        """compute stiffness matrix, non vectorized"""

        assert D == 2, "Only 2D case implemented"
        assert np.shape(inc) == (NxElm,)
        assert np.isscalar(k)

        # stiffness matrix in local reference frame
        K11 = np.array([[k, 0.0], [0.0, 0]])
        K = np.block([[K11, -K11], [-K11, K11]])

        # directional cosines of bar
        c = xy[inc[1]] - xy[inc[0]]
        c /= np.linalg.norm(c)
        # rotation matrix, computed from directional cosines (D=2)
        rot = np.zeros((D, D))
        rot[0] = c
        rot[1] = [-c[1], c[0]]
        np.testing.assert_array_almost_equal_nulp(
            rot @ rot.mT, np.identity(D), nulp=5
        )

        # diagonal rotation matrix
        z = np.zeros((D, D))
        R = np.block([[rot, z], [z, rot]])

        # rotate reference frame
        K = R.mT @ K @ R

        return K

    return (kmat,)


@app.cell
def _(D, DOF, N, inc, k, kmat, np, xy):
    #
    # assemble global K matrix
    #
    K = np.zeros((DOF, DOF))
    _kblock = K.reshape((N, D, N, D)).transpose((0, 2, 1, 3))

    for _i in range(len(inc)):
        _ki = kmat(inc[_i], xy, k[_i])
        _i0, _i1 = inc[_i]
        _kblock[_i0, _i0] += _ki[0:2, 0:2]
        _kblock[_i0, _i1] += _ki[0:2, 2:4]
        _kblock[_i1, _i0] += _ki[2:4, 0:2]
        _kblock[_i1, _i1] += _ki[2:4, 2:4]
    return (K,)


@app.cell
def _(DOF, K, eps, n_rigid, np, u_rigid):
    #
    # check K properties
    #

    # data type
    K.dtype == np.double

    # symmetry
    np.testing.assert_allclose(
        K, K.mT, rtol=eps * DOF, err_msg="Symmetry check failed"
    )

    # rank
    assert np.linalg.matrix_rank(K, hermitian=True) <= DOF - n_rigid, (
        "Matrix rank check failed"
    )

    # rigid modes
    assert all(
        np.linalg.norm(K @ u_rigid, axis=0)
        <= 1e-15
        * np.linalg.matrix_norm(K, ord="fro")
        * np.linalg.norm(u_rigid, axis=0)
    ), "Rigid modes test failed"

    # positive semidefinite
    assert np.abs(
        np.linalg.eigvalsh(K)[0]
    ) < eps * DOF * np.linalg.matrix_norm(K, ord="fro"), "Pos. definiteness check failed"
    return


@app.cell(hide_code=True)
def _(K, mo, np):
    with np.printoptions(precision=3):
        _out = mo.md(rf"""
    Stiffness matrix $\mathbf{{K}}$:
    ```
    {K!s}
    ```
    """)
    _out
    return


if __name__ == "__main__":
    app.run()

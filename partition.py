def partition(array,nrows,ncols):
    """split matrix into submatrices"""

    r, c = array.shape
    A = array[:nrows,:ncols]
    B = array[:nrows,ncols:]
    C = array[nrows:,:ncols]
    D = array[nrows:,ncols:]
    return (A,B,C,D)


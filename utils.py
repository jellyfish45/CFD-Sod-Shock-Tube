def VanLeer(r):
    return (r + np.abs(r)) / (1.0 + np.abs(r) + 1e-8)
def Minmod(r):
    if r<=0:
        return 0
    elif 0<r<=1:
        return r
    else:
        return 1
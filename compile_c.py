from cffi import FFI
from geometry import get_hessian_parts_R
import numpy as np

ffi = FFI()

ffi.cdef("void get_hessian_parts_R_c(size_t length,int xp[][3],int yp[][3],int hdx_R[],int hdy_R[],int hnd_raw_R[]);")
f = open('geometry.h', 'w')
f.write("void get_hessian_parts_R_c(size_t length,int xp[][3],int yp[][3],int hdx_R[],int hdy_R[],int hnd_raw_R[]);")
f.close()

ffi.set_source("_geometry",  # name of the output C extension
    '#include "geometry.h"'
    ,
    sources=['geometry.c']
)
if __name__ == "__main__":
    ffi.compile(verbose=True)

def test_gethc(xp, yp):
    from _geometry.lib import get_hessian_parts_R_c
    xp_c = ffi.new(f"int[{len(xp)}][3]", xp.tolist())
    yp_c = ffi.new(f"int[{len(yp)}][3]", yp.tolist())
    hdx_c = ffi.new(f'int[{len(xp)}]', len(xp)*[0])
    hdy_c = ffi.new(f'int[{len(yp)}]', len(yp)*[0])
    hnd_raw_c = ffi.new(f'int[{len(xp)*len(yp)*9}]', (len(xp) * len(yp) * 9)*[0])
    hdx_R, hdy_R, hnd_raw_R = get_hessian_parts_R(xp, yp)
    get_hessian_parts_R_c(len(xp), xp_c, yp_c, hdx_c, hdy_c, hnd_raw_c)
    print(np.max(hdx_R - list(hdx_c)))
    print(np.max(hdy_R - list(hdy_c)))
    print(np.max(np.reshape(hnd_raw_R, 9 * len(xp) ** 2) - list(hnd_raw_c)))
    
b=9

xp = np.einsum('ik,jk->ijk', np.stack((np.arange(b), np.ones(
        (b)), (b//2+1)*np.ones((b))), axis = -1), np.stack((np.ones((b)), np.arange(b), np.ones((b))), axis = -1)) - b//2
xp = np.reshape(xp, (b * b, 3))
xp=np.array(xp,dtype=np.int)
yp = xp
test_gethc(xp,yp)
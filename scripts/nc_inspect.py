import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from paths import DATA, OUTB  # noqa
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h5lite import File
import numpy as np
for p in sys.argv[1:]:
    f=File(p); print('=====',p)
    ga=f.attrs('/')
    for k in ['title','source','subset:productId','subset:datasetId','subset:date','history','product_version']:
        if k in ga: print('  @',k,'=',str(ga[k])[:150])
    for n in f.names():
        inf=f.info(n)
        if inf is None: print('  (group)',n); continue
        a=f.attrs(n)
        keep={k:(str(v)[:60]) for k,v in a.items() if k in ('units','long_name','standard_name','scale_factor','add_offset','_FillValue','calendar')}
        print(f'  {n:22s} {inf[0]} {inf[1]} {keep}')
    f.close()

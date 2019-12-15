from numpy.linalg import norm as np_norm
import numpy as np


def c_(x):
    return np.array(x)
norm_1 = lambda x: np.sum(abs(x))


RK_METHODS = {
    'FE':            [c_([[0]]), c_([1]), c_([0]),  False], #Explicit Euler
    'BE':            [c_([[1]]), c_([1]), c_([1]), True], #Implicit Euler
    'IMP':           [c_([[.5]]), c_([1]), c_([.5]), True], #Implicit Midpoint
    'ETrapz':        [c_([[0,0],[1,0]]), c_([.5,.5]), c_([0,1]),   False], # Explicit Trapezoidal
    'ITrapz':        [c_([[0,0],[.5,.5]]), c_([.5,.5]), c_([0,1]),  True], # Implicit Trapezoidal
    'RK4':           [c_([[0,0,0,0],[.5,0,0,0],[0,.5,0,0],[0,0,1,0]]), c_([1./6.,1./3.,1./3.,1./6.]), c_([0,.5,.5,1]),False], #Classical RK 4
    'RK4(3)I':       [c_([[.5,0,0,0],[1./6.,.5,0,0],[-.5,.5,.5,0],[1.5,-1.5,0.5,0.5]]), c_([1.5,-1.5,0.5,0.5]), c_([.5,2./3.,.5,1]), True], # Implicit stable RK 4 with 3 order of approx
    '3/8':           [c_([[0,0,0,0],[1./3.,0,0,0],[-1./3.,1,0,0],[1,-1,1,0]]), c_([.125, .475, .475, .125]), c_([0,1./3.,2./3.,1]) ,False],   # Classical #/8 rule of R-K
    'SSPRK3':        [c_([[0,0,0],[1,0,0],[.25, .25, 0]]), c_([1./6., 1./6., 2./3.]), c_([0,1.,1./2.]), False] # Third-order Strong Stability Preserving Runge-Kutta
}

RK_NAMES = {
    'FE':            'Explicit Euler',
    'BE':            'Implicit Euler',
    'IMP':           'Implicit Midpoint',
    'ETrapz':        'Explicit Trapezoidal',
    'ITrapz':        'Implicit Trapezoidal',
    'RK4':           'Classical RK 4',
    'RK4(3)I':       'Implicit stable RK 4 with 3 order of approximation',
    '3/8':           'Classical #/8 rule of R-K',
    'SSPRK3':        'Third-order Strong Stability Preserving Runge-Kutta'
    

}

def fixedp(f,x0,tol=10e-7, max_iter=100, log=True, nrm=norm_1):
    """ 
     Fixed point algorithm 
     x0 and f(x0) must be scalars or numpy arrays-like
 
    """
    err=0
    x = 0
    for itr in range(max_iter):
        x = f(x0)
        err = nrm(x-x0) 
        if err < tol:
            return x
        x0 = x
    if log:
        print("warning! didn't converge, err: ",err)
    return x

def implicit_RK(f,a,b,c,x,N,s,h, y0, nrm=norm_1):
    number_of_systems = y0.shape[0]
    y = np.zeros((N,number_of_systems))
    y[0] = y0
    for n in range(0,N-1):
        def k_eq(k):
            _arr_2 = lambda i : y[n]+ h*np.sum([k[j]*a[i][j] for j in range(s)], axis=0)
            return np.array([f(x[n]+c[i]*h, _arr_2(i)) for i in range(s)])
        k = fixedp(k_eq, np.ones((s,number_of_systems)))
        y[n+1] = y[n] + h*np.sum([b[i]*k[i] for i in range(s)], axis = 0)
    return x,y
  
def explicit_RK(f,a,b,c,x,N,s,h, y0,):
    number_of_systems = y0.shape[0]
    y = np.zeros((N,number_of_systems))
    y[0] = y0
    k = np.zeros((s,number_of_systems))
    for n in range(0,N-1):
        y[n+1] = y[n] 
        for i in range(s):
            k[i] = f(x[n]+c[i]*h, y[n]+h*np.sum([a[i][j]*k[j] for j in range(i)] , axis=0) )
            y[n+1]+=h*b[i]*k[i]
    return x,y

def RK(f, y0, t0, T, N=1000, h=None, method = 'RK4', butcher  = None, implicit = None, nrm=norm_1):
    N=N+1
    if butcher is None:
        a,b,c,implicit = RK_METHODS[method]
    else:
        if implicit is None:
            implicit = True
        a,b,c, = butcher
    if h is None:
        h = (T-t0)/N
        t = np.linspace(t0,T,N)
    else:
        N = np.floor((T-t0)/h)
        t = np.arange(t0,t,h)
    s = a.shape[0]
    if implicit:
        return implicit_RK(f,a,b,c,t,N,s,h, y0, nrm=nrm)
    else:
        return explicit_RK(f,a,b,c,t,N,s,h, y0)


from functools import partial
class solver:
    def __init__(self,name, method = None,butcher = None, implicit = None):
        if butcher is None:
            self.method = partial(RK, method = method)
        else:
             self.method = partial(RK, butcher = butcher, implicit = implicit)
        self.name = name
    def __call__(self, f, y0, t0, T, N=1000, h=None):
        return self.method(f, y0, t0, T, N=1000, h=None)

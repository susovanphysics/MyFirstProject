"""============ INFO ==================================================
Code for Parameter Space with shock of Novikov-Thorne Model in the Kerr metric

Author: Susovan Maity
Date: Starting 20th December, 2019
Place: HRI, Allahabad

===================================================================="""

import numpy as np
from scipy import optimize
from scipy.integrate import quad
from scipy.integrate import solve_ivp
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import timeit
import matplotlib as mpl
mpl.style.use('default')
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

rmax = 100.0  # maximum radius of integration
g = 1.35  # adiabatic index
eng = 1.001
rmin = 1.0

#functions-----------------------------------------------------------------------------
def phi(r):
    a = -1/r
    return a

def phi1(r):
    b = 1.0/(r**2)
    return b

def phi2(r):
    b = -2.0/(r**3)
    return b

def u_c(r):
    a = np.sqrt(r*phi1(r)/2)
    return a


def r_crit(e):
    a = -e * (g - 3.0) / (2 * (g - 1.0))
    return a



def dudr_c(r):
    alpha = (g + 1.0)*u_c(r)
    beta = ((4*g - 6.0) *(u_c(r)**2))/r + phi1(r)
    gamma = (4* g * u_c(r)**3)/(r**2) + phi2(r)
    slope1 = (-beta - np.sqrt(beta**2.0 - 4.0 * alpha * gamma)) / (2.0 * alpha)
    slope2 = (-beta + np.sqrt(beta**2.0 - 4.0 * alpha * gamma)) / (2.0 * alpha)
    slope = [slope1, slope2]
    return slope


def dudr(r, u, c):
    x = u*((2 * c**2)/r - phi1(r))
    y = u**2.0 - c**2.0
    return x / y

def dcdr(r, u, c):
    x = -(g - 1.0)/2*c
    y = phi1(r) + u*dudr(r,u,c)
    return x*y

def dcdr_c(r):
    x = -(g - 1.0)/2*u_c(r)
    y_0 = phi1(r) + u_c(r) * dudr_c(r)[0]
    y_1 = phi1(r) + u_c(r) * dudr_c(r)[1]
    return x * y_0, x * y_1

#Defining starting points of integration-----------------------------

def next_to_crit(r, x, h):
    k = [0] * 4
    l = [0] * 4
    k[0] = h * dudr_c(r)[0]
    l[0] = h * dcdr_c(r)[0]
    k[1] = h * dudr(r + 0.5 * h, x[0] + k[0] * 0.5, x[1] + l[0] * 0.5)
    l[1] = h * dcdr(r + 0.5 * h, x[0] + k[0] * 0.5, x[1] + l[0] * 0.5)
    k[2] = h * dudr(r + 0.5 * h, x[0] + k[1] * 0.5, x[1] + l[1] * 0.5)
    l[2] = h * dcdr(r + 0.5 * h, x[0] + k[1] * 0.5, x[1] + l[1] * 0.5)
    k[3] = h * dudr(r + h, x[0] + k[2], x[1] + l[2])
    l[3] = h * dcdr(r + h, x[0] + k[2], x[1] + l[2])

    x[0] += (k[0] + 2.0 * k[1] + 2.0 * k[2] + k[3]) / 6.0
    x[1] += (l[0] + 2.0 * l[1] + 2.0 * l[2] + l[3]) / 6.0

    return x


def next_to_crit_wind(r, x, h):
    k = [0] * 4
    l = [0] * 4
    k[0] = h * dudr_c(r)[1]
    l[0] = h * dcdr_c(r)[1]
    k[1] = h * dudr(r + 0.5 * h, x[0] + k[0] * 0.5, x[1] + l[0] * 0.5)
    l[1] = h * dcdr(r + 0.5 * h, x[0] + k[0] * 0.5, x[1] + l[0] * 0.5)
    k[2] = h * dudr(r + 0.5 * h, x[0] + k[1] * 0.5, x[1] + l[1] * 0.5)
    l[2] = h * dcdr(r + 0.5 * h, x[0] + k[1] * 0.5, x[1] + l[1] * 0.5)
    k[3] = h * dudr(r + h, x[0] + k[2], x[1] + l[2])
    l[3] = h * dcdr(r + h, x[0] + k[2], x[1] + l[2])

    x[0] += (k[0] + 2.0 * k[1] + 2.0 * k[2] + k[3]) / 6.0
    # print k[0],k[1],k[2],k[3]
    x[1] += (l[0] + 2.0 * l[1] + 2.0 * l[2] + l[3]) / 6.0

    return x


def sys_eq(r, y):
    return dudr(r, y[0], y[1]), dcdr(r, y[0], y[1])


def phase_portrait(r_c):
    f1 = open("accretion.txt","w")
    for dr in [-0.0025, 0.006]:
        # Accretion trajectory through critical point----------------
        x0 = u_c(r_c)
        y0_crit = [x0, x0]
        y0 = next_to_crit(r_c, y0_crit, dr)
        print (y0)
        
        r0 = r_c + dr
        if dr < 0.0:
            rrange_outer = np.arange(r0, rmin, dr)
            r_span = [r0, rmin]
            sol = solve_ivp(sys_eq, r_span, y0, t_eval=rrange_outer)
            y = sol.y[0] /sol.y[1] 
            plt.plot(sol.t, y, 'C0')            
        if dr > 0.0:
            rrange = np.arange(r0, rmax, dr)
            r_span = [r0, rmax]
            soln = solve_ivp(sys_eq, r_span, y0, t_eval=rrange)
            y = soln.y[0] /soln.y[1] 
            plt.plot(soln.t, y, 'C1')               
        #  Wind trajectory through critical point------------------
        y0 = next_to_crit_wind(r_c, y0_crit, dr)
        r0 = r_c + dr
        if dr < 0.0:
            rrange = np.arange(r0, rmin, dr)
            r_span = [r0, rmin]
            solb = solve_ivp(sys_eq, r_span, y0, t_eval=rrange)   
            yb = solb.y[0] /solb.y[1] 
            plt.plot(solb.t, yb, 'C2')
        if dr > 0.0:
            rrange = np.arange(r0, rmax, dr)
            r_span = [r0, rmax]
            solv = solve_ivp(sys_eq, r_span, y0, t_eval=rrange)
            yv = solv.y[0] /solv.y[1] 
            plt.plot(solv.t, yv, 'C3')
    rval = np.append(sol.t, soln.t)
    uval = np.append(sol.y[0], soln.y[0])
    cval = np.append(sol.y[1], soln.y[1])
    np.save('accretion.npy', (rval, uval, cval))
    f1.write("{}    {}    {}\n".format("r", "$u_0$", "$cs_0$"))
    for x in zip(rval, uval, cval):
        f1.write("{}    {}    {}\n".format(x[0], x[1], x[2]))
    
    f1.close()
    plt.xlabel(r'$r$')
    plt.ylabel(r'$\mathcal{M} = u_0 / c_{s0}$')
    plt.title(
        r'$\mathcal{E} = %.3f,\gamma_0 = %.2f$' % (eng, g))
    plt.savefig('phase portrait_%.2f.pdf' % eng)
    plt.show()


r_cr = r_crit(eng)
print (r_cr)
phase_portrait(r_cr)

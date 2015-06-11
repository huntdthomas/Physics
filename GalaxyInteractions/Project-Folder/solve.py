
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.integrate import odeint
from IPython.html.widgets import interact, fixed

Rmin = 25

M = 1e1

Y0 = 20*np.sqrt(10)
e = 7 #Eccentricity

r_array = np.array([.2,.3,.4,.5,.6])*Rmin
N_array = np.array([12,18,24,30,36])

steps = 1e3
t = np.linspace(0,.4,steps) #Timescale of 1 billion years

atol=1e-6
rtol=1e-6
gamma = 4.49933e4 #Units ((kpc)^3)/((M_sun^10)(billion_years)^2)

def parabolic(Y0,M,S,gamma,Rmin):
    """Computes the initial conditions necessary for the disrupting galaxy to approach
    with a parabolic trajectory.
    
    Parameters
    ----------
    Y0 : Float
        The Initial Y Distance of S from M
    Rmin : Float
        The closet distance between S and M on this parabolic trajectory
        
    
    Returns
    -------
    ic,IC : ndarray
        The vector of initial conditions [X0,dX0,Y0,dY0].
    """
    
    X0 = Rmin - Y0**2/(4*Rmin) #X0 is calculated from a parabola equation
    dxdy = -Y0/(2*Rmin) #The slope is calculated at any point. Used to create a tangential velocity
    theta = np.arctan(1/abs(dxdy))
    v = np.sqrt(2*gamma*(M+S)/np.sqrt(X0**2+Y0**2))
    dX0 = v*np.cos(theta)
    dY0 = -v*np.sin(theta)
    
    return np.array([X0,dX0,Y0,dY0])

def hyperbolic(Y0,M,S,gamma,Rmin,e):
        
    """Computes the initial conditions necessary for the disrupting galaxy to approach
    with a hyperbolic trajectory.
    
    Parameters
    ----------
    Y0 : Float
        The Initial Y Distance of S from M
    Rmin : Float
        The closet distance between S and M on this hyperbolic trajectory
    e : Float
        The eccentricity of the hyperbolic orbit
        
    
    Returns
    -------
    ic,IC : ndarray
        The vector of initial conditions [X0,dX0,Y0,dY0].
    """
    
    C = (1+e)*Rmin
    Rmax = C/(1-e)
    a = (Rmin+Rmax)/2

    alpha = C/(e**2 - 1)
    beta = C/np.sqrt(e**2 - 1)
    delta = alpha*e
    
    X0 = -alpha*np.sqrt(1+((Y0**2)/beta**2)) + delta
    
    dydx = ((beta**2)/(alpha**2))*(X0 - delta)/Y0
    
    theta = np.arctan(dydx)
    
    v = np.sqrt(gamma*(M+S)*((2/np.sqrt(X0**2+Y0**2)) - 1/a))
    dX0 = v*np.cos(theta)
    dY0 = v*np.sin(theta)
    
    return np.array([X0,dX0,Y0,dY0])

#Creating the approach type based on settings entered at the start
def set_IC(Hyperbolic_Approach,M,S):
    """Creates IC. The initial conditions for S
    
    Parameters
    ----------
    Parabolic_Approach and Hyperbolic_Approach : Boolean
        The boolean dictates what type of initial conditions to create
    
    Returns
    -------
    IC : array
        The vector of initial conditions [Xo,dXo,Yo,dYo].
    """
    
    if Hyperbolic_Approach == True:
        IC = hyperbolic(Y0,M,S,gamma,Rmin,e)
    else:
        IC = parabolic(Y0,M,S,gamma,Rmin)
    return IC

def create_ic(theta,r,IC,gamma,Retrograde_Orbit):
    """Computes the initial conditions necessary for the stars to be in a circular orbit.
    Then pairs those initial conditions with the initial conditions of S to create an
    array that represents the initial conditions of the whole system. For one star.
    
    Parameters
    ----------
    theta : Float
        The angle [0,2pi] from the horizontal that the star will be placed at
    r : Float
        The radial distance from M that m will be located at.
    IC: Array
        The initial conditions for the disrupting galaxy. [Xo,dXo,Yo,dYo]
    
    Returns
    -------
    ic,IC : ndarray
        The vector of initial conditions [xo,dxo,yo,dyo,Xo,dXo,Yo,dYo].
    """
    x0 = r*np.cos(theta)
    y0 = r*np.sin(theta)

    v= np.sqrt(gamma*M/(np.sqrt(x0**2+y0**2)))
    
    if Retrograde_Orbit == True:
        dx0 = -v*np.sin(theta)
        dy0 = v*np.cos(theta)
    else:
        dx0 = v*np.sin(theta)
        dy0 = -v*np.cos(theta)

    ic = np.array([x0,dx0,y0,dy0])

    return np.append(ic,IC) 

def single_shell(N,r,IC,Retrograde_Orbit):
    """Creates an array of arrays that holds all of the initial conditions for a 
    shell of N stars at radius r.
    
    Parameters
    ----------
    N : Integer
        The number of orbiting stars.
    r : Float
        The radial distance from M that m will be located at.
    IC: Array
        The initial conditions for the disrupting galaxy. [Xo,dXo,Yo,dYo]
    
    Returns
    -------
    ic,IC : ndarray
        The vector of initial conditions [[xo,dxo,yo,dyo,Xo,dXo,Yo,dYo],.....,[xo,dxo,yo,dyo,Xo,dXo,Yo,dYo]].
    """
    thetas = np.arange(0,2*np.pi,2*np.pi/N)
    ic = np.array([create_ic(theta,r,IC,gamma,Retrograde_Orbit) for theta in thetas])
    return ic

def all_ic(N_array,r_array,IC,Retrograde_Orbit):
    """Creates an array of the initial conditions for every single star.
    
    Parameters
    ----------
    N_array : Array
        Array where entries correspond to the number of stars at each orbital shell.
    r_array : Array
        Entries correspond to the radius of each shell
    IC: Array
        The initial conditions for the disrupting galaxy. [Xo,dXo,Yo,dYo]
    
    Returns
    -------
    allic : ndarray
        The solution vector [[x,dx,y,dy,X,dX,Y,dY],.....,[x,dx,y,dy,X,dX,Y,dY]].
    """
    byshell = np.array([single_shell(N_array[i],r_array[i],IC,Retrograde_Orbit) for i in range(len(N_array))])
    allic = np.array([byshell[i][k] for i in range(len(byshell)) for k in range(N_array[i])])
    return allic

def derivs(rvec,t,M,S,gamma):
    """Compute the derivatives of our system.
    
    Parameters
    ----------
    rvec : ndarray
        The solution vector at the current time t[i]: [x[i],dx[i],y[i],dy[i],X[i],dX[i],Y[i],dY[i]].
        x,y corresponds to vector from M to m
        X,Y corresponds to vector from M to S
    t : float
        The current time t[i].
    M, S:
        Mass parameters in the differential equation.
    
    Returns
    -------
    drvec : ndarray
        The vector of derviatives at t[i]: [dx[i],d2x[i],dy[i],d2y[i],dX[i],d2X[i],dY[i],d2Y[i]].
    """
    
    "Compute x and y components of r vector"
    x = np.array(rvec[0])
    y = np.array(rvec[2])
    
    "Compute X and Y components of R vector"
    X = np.array(rvec[4])
    Y = np.array(rvec[6])
    
    
    "Compute x and y components of Rho vector"
    xrho = np.array(X-x)
    yrho = np.array(Y-y)
    
    "Compute Magnitudes of r,R, and Rho vectors"
    r = np.sqrt(x**2 + y**2)
    R = np.sqrt(X**2 + Y**2)
    rho = np.sqrt(xrho**2 + yrho**2)
    
    
    "Velocity Vector Components"
    dx = np.array(rvec[1])
    dy = np.array(rvec[3])
    
    dX = np.array(rvec[5])
    dY = np.array(rvec[7])
    
    "Acceleration Vectors"
    d2x = np.array(-gamma*((M/(r**3))*x - (S/(rho**3))*xrho + (S/(R**3))*X))
    d2y = np.array(-gamma*((M/(r**3))*y - (S/(rho**3))*yrho + (S/(R**3))*Y))
    
    d2X = np.array(-gamma*((M+S)/(R**3))*X)
    d2Y = np.array(-gamma*((M+S)/(R**3))*Y)
    
    "Vector Derivatives"
    drvec = np.array([dx,d2x,dy,d2y,dX,d2X,dY,d2Y])
    return drvec

def solution(t,allic,N_array,atol,rtol,gamma,Hyperbolic_Approach,M,S):
    """Creates the entire solution vector for the system.
    
    Parameters
    ----------
    allic : ndarray
        The initial conditions array for the entire system. Initial conditions
        for every single star.l
    t : linespace
        The points in time to solve the system.
    N_array:
        Array denotes the number of particles in each shell.
    
    Returns
    -------
    soln : ndarray
        An array that contains the solution array for N stars. 
        [[[xo1,dxo1,yo1,dyo1,Xo1,dXo1,Yo1,dYo1]],...,[xoN,dxoN,yoN,dyoN,XoN,dXoN,YoN,dYoN]]
    """
    steps = 1e3
    if Hyperbolic_Approach == True:
        t = np.linspace(0,.4,steps) #Timescale of 1 billion years
    else:
        t = np.linspace(0,.8,steps)
    soln = np.array([odeint(derivs,allic[k],t,args=(M,S,gamma),atol=atol, rtol=rtol) for k in range(len(allic))])
    return soln

def coordinate_solution(soln,steps):
    """Sweeps through soln(the output of solution()) to get the individual component solution vectors.
    Change k to determine which coordinate solution is returned.
    
    Parameters
    ----------
    soln : ndarray
        The output of solution().
    steps : Float
        The number of points in time the system is solved.
    Returns
    -------
    x,dx,y,dy,X,dX,Y,dY : 8 ndarrays
        An array that contains all coordinate for N stars. 
        [[[qo1]],...,[qoN]]
    """
    x = np.empty((len(soln),steps))
    dx = np.empty((len(soln),steps))
    y = np.empty((len(soln),steps))
    dy = np.empty((len(soln),steps))
    
    for i in range(len(soln)):
        x[i]  = np.array([e[0] for e in soln[i]])
        dx[i] = np.array([e[1] for e in soln[i]])
        y[i]  = np.array([e[2] for e in soln[i]])
        dy[i] = np.array([e[3] for e in soln[i]])
    X  = np.array([e[4] for e in soln[i]])
    dX = np.array([e[5] for e in soln[i]])
    Y  = np.array([e[6] for e in soln[i]])
    dY = np.array([e[7] for e in soln[i]])
    return x,dx,y,dy,X,dX,Y,dY

def frame(x,dx,y,dy,X,dX,Y,dY,CenterOfMass,M,S):
    """If CenterOfMass is true, this transforms the solution vectors
        into the new center of mass vectors

    Parameters
    ----------
    x,dx,y,dy,X,dX,Y,dY : array
        The outputs of coordinate_solution().
    CenterOfMass : Boolean
        If true this converts the vectors, if false
        x,dx,y,dy,X,dX,Y,dY are returned
    Returns
    -------
    Coordinate Solution Arrays : ndarrays
    """
    if CenterOfMass == False:
        return x,dx,y,dy,X,dX,Y,dY
    if CenterOfMass == True:
        #New m coordinates
        x = x -(S*X/(M+S))
        y = y - (S*Y/(M+S))

        #New M Coordinates
        X1 = X*(-S/(M+S))
        Y1 = Y*(-S/(M+S))

        #New S Coordinates
        X2 = X*(M/(M+S))
        Y2 = Y*(M/(M+S))
        
        return x,y,X1,Y1,X2,Y2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.integrate import odeint
from IPython.html.widgets import interact, fixed




def energy_calc(data,M,S):
    X = data['X']
    Y = data['Y']
    dX = data['dX']
    dY = data['dY']
    """Compute the energy of our system at each step in time.
    
    Parameters
    ----------
    X,Y,dX,dY : ndarray
        The coordinate solution vectors M/S system.

    M, S:
        Mass parameters in the differential equation.

    Returns
    -------
    TotalEnergy : ndarray
        The energy of the system at each point in time.
    """
    gamma = 4.49933e4
    
    
    U = -gamma*M*S/(np.sqrt(X**2 + Y**2))
    T = (1/2)*S*(dX**2 + dY**2)
    return np.array(U+T)

def plot_energy(t,Energy):
    plt.plot(t,Energy);
    plt.xlabel('Time (Billion Years)');
    plt.ylabel('Energy ($19*10^45$ Joules)');
    plt.title('Galaxy Energy vs Time');
    plt.show()
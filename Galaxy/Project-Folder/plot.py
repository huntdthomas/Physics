import matplotlib.pyplot as plt
import numpy as np
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy

def M_Animation(data,duration):
    """Compute the energy of our system at each step in time.

    Parameters
    ----------
    data : npz 
        Contains the coordinate solutions

    Duration: Integer
        Seconds that the animation will run.

    Returns
    -------
    Animation :
        Animation of the solution for time duration
    """
    
    x = data['x']
    y = data['y']
    X = data['X']
    Y = data['Y']
    
    fig_mpl, ax = plt.subplots(1,figsize=(6,6), facecolor='white');
    plt.axis('off')
    plt.sca(ax);
    plt.xlim(-45,45);
    plt.ylim(-45,45);
    scatterplot = ax.scatter(X, Y,color='red',s=40);
    starscatter = ax.scatter(x,y,color='green',s=5);

    ax.scatter(0,0,color='purple',s=30)

    def make_frame_mpl(t):
        # t is the current time between [0,duration]
        Xpoints = np.array(X[t*(1000/duration)])
        Ypoints = np.array(Y[t*(1000/duration)])

        starx = np.array([entry[t*(1000/duration)] for entry in x])
        stary = np.array([entry[t*(1000/duration)] for entry in y])
        # Just update the data on each frame
        # set_offset takes a Nx2 dimensional array of positions
        #First column is x, second is y.
        scatterplot.set_offsets(np.transpose(np.vstack([Xpoints, Ypoints])));
        starscatter.set_offsets(np.transpose(np.vstack([starx,stary])));
        # The mplfig_to_npimage convert the matplotlib figure to an image that
        # moviepy can work with:
        return mplfig_to_npimage(fig_mpl)
    return mpy.VideoClip(make_frame_mpl, duration=duration)

def CM_Animation(data,duration):
    """Compute the energy of our system at each step in time.
    
    Parameters
    ----------
    data : npz 
        Contains the coordinate solutions

    Duration: Integer
        Seconds that the animation will run.

    Returns
    -------
    Animation :
        Animation of the solution for time duration
    """
    x = data['x']
    y = data['y']
    X1 = data['X1']
    Y1 = data['Y1']
    X2 = data['X2']
    Y2 = data['Y2']
    
    fig_mpl, ax = plt.subplots(1,figsize=(6,6), facecolor='white');
    plt.axis('off')
    plt.sca(ax);
    plt.xlim(-65,65);
    plt.ylim(-65,65);
    scatterplot = ax.scatter(X1, Y1,color='red',s=40);
    starscatter = ax.scatter(x,y,color='green',s=5);
    scatterplot2 = ax.scatter(X2, Y2,color='red',s=40);

    ax.scatter(0,0,color='purple',s=1)

    def make_frame_mpl(t):
        # t is the current time between [0,duration]
        X1points = np.array(X1[t*(1000/duration)])
        Y1points = np.array(Y1[t*(1000/duration)])
        X2points = np.array(X2[t*(1000/duration)])
        Y2points = np.array(Y2[t*(1000/duration)])

        starx = np.array([entry[t*(1000/duration)] for entry in x])
        stary = np.array([entry[t*(1000/duration)] for entry in y])
        # Just update the data on each frame
        # set_offset takes a Nx2 dimensional array of positions
        #First column is x, second is y.
        scatterplot.set_offsets(np.transpose(np.vstack([X1points, Y1points])));
        starscatter.set_offsets(np.transpose(np.vstack([starx,stary])));
        scatterplot2.set_offsets(np.transpose(np.vstack([X2points, Y2points])));
        # The mplfig_to_npimage convert the matplotlib figure to an image that
        # moviepy can work with:
        return mplfig_to_npimage(fig_mpl)
    return mpy.VideoClip(make_frame_mpl, duration=duration)
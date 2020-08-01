# plot2d and plot3d
# ref: https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# Default configuration
contourParams = dict(
    zdir='z',
    alpha=0.5,
    zorder=1,
    antialiased=True,
    cmap=cm.PuRd_r
)

surfaceParams = dict(
    rstride=1,
    cstride=1,
    linewidth=0.1,
    edgecolors='k',
    alpha=0.5,
    antialiased=True,
    cmap=cm.PuRd_r
)


def plot2d(F, bounds, points=100, figure=None, figsize=(12, 8), contour=True, contour_levels=20,
           imshow_kwds=None, contour_kwds=None):
    if imshow_kwds is None:
        imshow_kwds = dict(cmap=cm.PuRd_r)
    if contour_kwds is None:
        contour_kwds = dict(cmap=cm.PuRd_r)
    xbounds, ybounds = bounds[0], bounds[1]
    x = np.linspace(min(xbounds), max(xbounds), points)
    y = np.linspace(min(xbounds), max(xbounds), points)
    X, Y = np.meshgrid(x, y)
    Z = F(np.asarray([X, Y]))
    if figure is None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = figure
    ax = fig.gca()
    if contour:
        ax.contourf(X, Y, Z, contour_levels, **contour_kwds)
    else:
        im = ax.imshow(Z, **imshow_kwds)
    if figure is None:
        plt.show()
    return fig, ax


def plot3d(F, bounds, points=100, contour_levels=20, ax3d=None, figsize=(12, 8),
           view_init=None, surface_kwds=None, contour_kwds=None):
    from mpl_toolkits.mplot3d import Axes3D
    contour_settings = dict(contourParams)
    surface_settings = dict(surfaceParams)
    if contour_kwds is not None:
        contour_settings.update(contour_kwds)
    if surface_kwds is not None:
        surface_settings.update(surface_kwds)
    xbounds, ybounds = bounds[0], bounds[1]
    x = np.linspace(min(xbounds), max(xbounds), points)
    y = np.linspace(min(ybounds), max(ybounds), points)
    X, Y = np.meshgrid(x, y)
    Z = F(np.asarray([X, Y]))
    if ax3d is None:
        fig = plt.figure(figsize=figsize)
        ax = Axes3D(fig)
        if view_init is not None:
            ax.view_init(*view_init)
    else:
        ax = ax3d
    # Make the background transparent
    ax.patch.set_alpha(0.0)
    # Make each axis pane transparent as well
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    surf = ax.plot_surface(X, Y, Z, **surface_settings)
    contour_settings['offset'] = np.min(Z)
    cont = ax.contourf(X, Y, Z, contour_levels, **contour_settings)
    if ax3d is None:
        plt.show()
    return ax


if __name__ == '__main__':
    plot2d(lambda x: sum(x**2), bounds=[(-100, 100)] * 2)
    plot3d(lambda x: sum(x**2), bounds=[(-100, 100)] * 2)

# makes a colormesh of a 2d array overlay

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap as Basemap
from matplotlib.patches import Polygon
import numpy as np

# data: a 2d array of *projected* datapoints.
# avg: the middle of the color bar
# rng: the min and max of the color bar
# title: title for the plot
# cmap: determines the colorbar type
def draw_states(data, avg, rng, title, min_x, max_x, min_y, max_y, fig_name=None, ax=None, cmap='coolwarm_r'):
    # set current figure to parent of ax
    if ax == None:
        plt.figure()
        ax=plt.axes()
    else:
        plt.sca(ax)
    if fig_name==None: fig_name = 'out/' + title + '.png'

    # Lambert Conformal map of lower 48 states.
    m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
                projection='lcc',lat_1=33,lat_2=45,lon_0=-95,ax=ax)

    xs = np.linspace(min_x, max_x, data.shape[1])
    ys = np.linspace(min_y, max_y, data.shape[0])
    xs, ys = np.meshgrid(xs, ys)


    m.drawmapboundary(fill_color=(0,0,0,0), zorder=20)
    shp_info = m.readshapefile('shp/nationp010g','borders', linewidth=0)
    for nshape,seg in enumerate(m.borders):
        if nshape == 1873: #This nshape denotes the large continental body of the USA, which we want
            poly = Polygon(seg,facecolor='none',edgecolor='none')
            ax.add_patch(poly)
            im = m.pcolormesh(xs, ys, data, latlon=True, cmap=cmap, vmin=avg-rng, vmax=avg+rng, zorder=1, clip_path=poly, clip_on=True)
            cb = m.colorbar(im, location='bottom', size='3%')
            break

    # draw state borders
    shp_info = m.readshapefile('shp/st99_d00','borders', color='none')
    for nshape,seg in enumerate(m.borders):
        color=[0.4]*4
        poly = Polygon(seg,facecolor='none', lw=0.6, edgecolor=color)
        ax.add_patch(poly)

    plt.title(title)
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    plt.close()

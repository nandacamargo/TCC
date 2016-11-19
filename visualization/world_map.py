import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import urllib, os


lats = []
lons = []
infile = 'all_states.json'

with open(infile, 'r') as f:
    for line in f:
        if ('coordinates' in line):
            lons.append(float(f.readline().strip().split(',')[0]))
            lats.append(float(f.readline().strip().split(',')[0]))

# draw map with Robinson Projection
m = Basemap(projection='robin',lon_0=0,resolution='c')

#This line transforms all the lat, lon coordinates into the appropriate projection.
x, y = m(lons,lats)

m.drawcoastlines()

#Draw parallels and meridians
m.drawparallels(np.arange(-90.,120.,30.))
m.drawmeridians(np.arange(0.,360.,60.))
m.drawmapboundary(fill_color='aqua')

#Draw countries and paint continents.
m.drawcountries()
m.fillcontinents(color='coral', lake_color='aqua', zorder=0)

#Plot
m.scatter(x,y,10,marker='o',color='k')
plt.title("Projeção de Robinson com pontos dos tweets com geolocalização")
plt.show()

from __future__ import division
import sys
from numpy import array, zeros, sqrt, exp, rot90, flipud, fliplr
from scipy import stats
import numpy as np
from draw_map import draw_states

# makes a map of lat-lon data from a beta distribution
# uses kernel density estimation to spread the data geographically
# uses bayesian methods to infer the posterior for each cell, which
#      makes the method robust against low population areas
# length_scale: the standard deviation of the kernel, controls how widely to spread each point (unit is degrees)
# width_constant: controls where we assume the kernel is insignificantly small. smaller values give faster running time, at the cost of inaccuracy
# colormap_width: controls how sensitive the colormap is. higher values produce a narrower range, and thus more dramatic plot
def make_map(input_file='data.input', title='fraction of census population that is male', out_file=None, grid_size=2000, length_scale=1, exponent=2, width_constant=10, colormap_width=1):
    print('reading data from file: %s' % input_file)
    data = read_from_file(input_file)

    min_x = min([e[0] for e in data])
    max_x = max([e[0] for e in data])
    min_y = min([e[1] for e in data])
    max_y = max([e[1] for e in data])
    wid_x = max_x-min_x
    wid_y = max_y-min_y

    # this is how many indices wide we compute the kernel
    width = int(width_constant*(length_scale/wid_x)*grid_size)

    # A *slice* allows us to add our data *only* to a local part of the array,
    # significantly speeding computation
    q_slice = zeros((2*width+1, 2*width+1))
    for i in range(-width, width + 1):
        for j in range(-width, width + 1):
            # compute distance in terms of degrees
            d = sqrt((i/(grid_size)*wid_x)**2 + (j/(grid_size)*wid_y)**2)
            wt = 0 if d>width_constant*length_scale else exp(-(d**exponent / length_scale**2))
            q_slice[i+width, j+width] = wt

    print('aggregating data via kernel density estimation')
    quest = zeros((grid_size+2*width+1, grid_size+2*width+1))
    total = zeros((grid_size+2*width+1, grid_size+2*width+1))
    for d in data:
        x_index, y_index = loc_to_index(d[0], d[1], min_x, max_x, min_y, max_y, width, grid_size)
        quest[x_index-width:x_index+width+1, y_index-width:y_index+width+1] += q_slice * d[2]
        total[x_index-width:x_index+width+1, y_index-width:y_index+width+1] += q_slice * d[3]

    # desregard the outer 'width' band around the matrix, because this was dead space
    quest = quest[width:grid_size+width+1, width:grid_size+width+1]
    total = total[width:grid_size+width+1, width:grid_size+width+1]

    # properly orient our matrices
    quest = fliplr(rot90(quest,3))
    total = fliplr(rot90(total,3))
    
    print('(bayesianly) smoothing data')
    quest, avg, rng = data_posterior(quest,total,grid_size,pct_ignore=.2)

    print('drawing map')
    draw_states(quest, avg, rng*colormap_width, title, min_x, max_x, min_y, max_y, fig_name=out_file)

    print('success!')

# calculate the beta posterior of a variable, given data and distribution parameters
def calc_beta(total, yes_count, avg, beta_scale):
    return (avg * beta_scale + yes_count) / (beta_scale + total)

# returns given some lon and lat, return the corresponding matrix index
def loc_to_index(x, y, min_x, max_x, min_y, max_y, index_buffer, bins):
    x_wid = max_x - min_x
    y_wid = max_y - min_y
    x_index = bins * (x-min_x) / float(x_wid)
    y_index = bins * (y-min_y) / float(y_wid)
    return index_buffer+int(x_index), index_buffer+int(y_index)

# reads data in the format: [longitude, latitude, yes answers, total answers]
def read_from_file(filename):
    lines = [line.split('\t') for line in open(filename).readlines()]
    d = [(float(lo), float(la), int(y), int(t)) for lo,la,y,t in lines]
    return d

# for data that is assumed to come from a beta distribution,
# compute compute the (average of the) posterior for each cell.
# this prevents variance in sparse regions from dominating the plot
#
# quest: a matrix containing total 'yes' values per cell
# total: a matrix containing total 'flips' per cell
# output: a matrix of data ready for plotting, the average, and the range of the data
def data_posterior(quest, total, grid_size, pct_ignore=0.0):
    # don't use the edges of the matrices for computation, as they are
    # off the map and can give division by 0 errors
    unbuff=int(pct_ignore*grid_size)
    questflat = quest[unbuff:grid_size-unbuff, unbuff:grid_size-unbuff].flatten()
    totalflat = total[unbuff:grid_size-unbuff, unbuff:grid_size-unbuff].flatten()

    # take the average
    pre_avg = np.mean(quest) / np.mean(total)
    ratios = np.array([x/y for (x,y) in zip(questflat, totalflat) if y>0 and x<y and x>0])

    # can optionally weight each ratio
    weights = np.array([1 for (x,y) in zip(questflat, totalflat) if y>0 and x<y and x>0])
    weight_sum = sum(weights)
    if weight_sum == 0:
        print('error: total weight is 0')
        exit()
    
    # for the prior we fit a beta distribution to the national histogram of ratios
    gmean = np.exp(sum(weights*np.log(ratios))/weight_sum)
    gmean_minus = np.exp(sum(weights*np.log(1-ratios))/weight_sum)
    alpha_hat = .5 + gmean / (2 * (1-gmean - gmean_minus))
    beta_hat = .5 + gmean_minus / (2 * (1-gmean - gmean_minus))

    # this is the standard deviation * 1.8
    rng = (alpha_hat*beta_hat / ((alpha_hat+beta_hat)**2 * (alpha_hat+beta_hat+1))) ** .5 * 1.8
    avg = alpha_hat / (alpha_hat + beta_hat)

    # normalize
    for i in range(grid_size):
      for j in range(grid_size):
        quest[i,j] = calc_beta(total[i,j], quest[i,j], avg, alpha_hat + beta_hat)
    return quest, avg, rng

if __name__=='__main__':
    if len(sys.argv) == 1:
        make_map()
    elif len(sys.argv) == 2:
        make_map(sys.argv[1])
    elif len(sys.argv) == 3:
        make_map(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        make_map(sys.argv[1], sys.argv[2], sys.argv[3])

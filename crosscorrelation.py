#================================================
#    Cross correlation for guide camera images
#================================================

from __future__ import division
import time
import glob
from tiptilt import image
from tiptilt import common
import numpy as np
from scipy import signal
from scipy import ndimage
from scipy import optimize
import numpy as np
import scipy
import sep
import re

def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))


"""
Source: http://scipy.github.io/old-wiki/pages/Cookbook/FittingData#Fitting_a_2D_gaussian
"""
def gaussian(height, center_x, center_y, width):
    """Returns a gaussian function with the given parameters"""
    width = float(width)
    return lambda x,y: height*np.exp(-(((center_x-x)/width)**2+((center_y-y)/width)**2)/2)


def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/abs(col.sum()))
    height = data.max()
    return height, x, y, width_x


def fit_2dgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    return p, success


def create_reference_image(directory, number_reference_images = 3):
    for dir in directories:
        files = sorted(glob.glob(dir+"*.pgm"))[0:number_reference_images]
        reference_image = np.zeros((read_pgm(files[0], byteorder='<').shape))
        for i, x in enumerate(files):
            print 'File used for reference image:', x.split('/')[-1]
            data = read_pgm(x, byteorder='<').copy()
            reference_image += data
        reference_image =  reference_image / number_reference_images
    reference_image, background =  image.subtract_background(reference_image)
    print ''
    return reference_image, background


def cross_correlation(title, l, object, image1, image2, window, M2_offset):
    pixel_size = 0.00586                # pixel size in mm of guide camera
    f = 330000                          # focal length in mm

    correlation_results = {}
    dxsum = 0
    dysum = 0
    saturation = 1
    error = 1
    status = []


    correlation_matrix = scipy.signal.correlate2d(image1[window], image2[window], mode = 'same', boundary = 'symm')
    params, cov = fit_2dgaussian(correlation_matrix)
    fit = gaussian(*params)

    x_cen = params[1]
    y_cen = params[2]
    '''
    fig = figure()
    axis = fig.add_subplot(111)
    axis.imshow(correlation_matrix, cmap = 'gray', interpolation = 'nearest', origin = 'lower')
    try:
        axis.contour(fit(*np.indices(correlation_matrix.shape)), 12, colors='w')
    except:
        pass
    fig.savefig('images/' + title.split('.')[0] + '_' + str(l) + '.png')
    plt.close(fig)
    '''
    xoffset = len(correlation_matrix)/2 - x_cen
    yoffset = len(correlation_matrix)/2 - y_cen

    xoffset_rad = (np.arctan( (xoffset * pixel_size ) / f))
    yoffset_rad = (np.arctan( (yoffset * pixel_size ) / f))

    '''
    fig = figure()
    axis = fig.add_subplot(111)
    axis.imshow(reference_image[window], cmap = 'gray', interpolation = 'nearest', origin = 'lower')
    #axis.scatter(x_cen,y_cen)
    #axis.scatter(spx0, spy0, color = 'r', s = 20)
    show()
    '''

    status, saturation, error, found, fitfactor = check_status(xoffset, yoffset, cov, image1, window)
    weight = object['flux'] * saturation * error * found * fitfactor

    correlation_results['x'] = xoffset_rad
    correlation_results['y'] = yoffset_rad
    correlation_results['weight'] = weight
    correlation_results['status'] = status

    return correlation_results


def check_status(dx, dy, cov, image0, window):
    status = []
    max_displacement = 5
    saturation_limit = 65536

    saturation = 1
    error = 1
    found = 1
    fitfactor = 1

    if abs(np.max(image0[window]) - np.mean(image0[window])) <= 5:
        found = 0
        status.append('No source found')

    if image0[window].max() >= saturation_limit:
        saturation = 0
        status.append('Saturation limit reached')

    if abs(dx) >= max_displacement or abs(dy) >= max_displacement:
        status.append('Exceptionally large displacement')
        fitfactor = 0

    if np.isinf(cov).any() == True:
        status.append('Convariance matrix contains infinities.')
        error = 0

    if (np.isinf(cov).any() == False and
    abs(dx) < max_displacement and
    abs(dy) < max_displacement and
    image0[window].max() != saturation_limit
    and abs(np.max(image0[window]) - np.mean(image0[window])) >= 5):
        status.append('Fit correct')

    return status, saturation, error, found, fitfactor


def main(title, reference_image, reference_background, file, M2_offset, window_size = 15, threshold_factor = 4, minimum_flux = 60):

    shift = []
    fitsfile = read_pgm(file, byteorder='<')
    original = fitsfile
    original = np.asarray(original, dtype = np.float)
    data, background =  image.subtract_background(original)
    sources = image.select_sources(data, background, threshold_factor = threshold_factor, window_size = window_size, minimum_flux = minimum_flux)
    sources = sources[np.where(sources['flux'] > 80)]
    print 'Number of sources detected:', len(sources)

    '''
    ratio = data.shape[0] * 1.0 / data.shape[1]
    fig = figure(figsize=(10,ratio *10))
    axis = fig.add_subplot(111)
    axis.imshow(data, cmap = 'gray', interpolation = 'nearest', origin = 'lower')
    for i,x in enumerate(sources):
        window = image.source_window(x, window_size)
        axis.add_patch(
            patches.Rectangle(
                (window[1].start, window[0].start),
                window[1].stop - window[1].start,
                window[0].stop - window[0].start,
                color = 'r',
                fill=False
            )
        )
    show()
    '''
    start = time.time()
    for i, x in enumerate(sources):
        window = image.source_window(x, window_size)
        shifts = cross_correlation(title,i,x, reference_image, data, window, M2_offset)

        #fig = figure()
        #axis = fig.add_subplot(111)
        #axis.imshow(reference_image[window], cmap = 'gray', interpolation = 'nearest', origin = 'lower')
        #axis.scatter(spx0, spy0, color = 'r', s = 20)
        #show()
        if shifts['status'] == ['Fit correct']:
            shift.append(shifts)

    print 'Number of correct sources detected:', len(shift)
    print shift
    xnumerator, ynumerator, denomiator = 0, 0, 0
    if len(shift) != 0:
        for j in xrange(0,len(shift)):
            xnumerator += shift[j]['x'] * shift[j]['weight']
            ynumerator += shift[j]['y'] * shift[j]['weight']
            denomiator += shift[j]['weight']
        dx = xnumerator / denomiator
        dy = ynumerator / denomiator
        print 'Average weighted shift x [rad], y [rad]: %s %s' %( dx,dy )
        print 'Calculation time', time.time() - start
    else:
        print 'No sources used to determine shift. '
        print 'Calculation time', time.time() - start
        dx, dy = 0, 0
    print ''

    return dx, dy


if __name__ == '__main__':
    # import required modules
    import scipy.optimize as opt
    import matplotlib
    from matplotlib.pyplot import figure, show
    import matplotlib.pyplot as plt
    from matplotlib import pyplot
    from matplotlib import patches
    from matplotlib import colors

    directory_prefix = "/media/data/"
    directories = [directory_prefix + "Surfdrive/MeerLICHT/2017_02_14_Guidecameras/field3/1Hz/"]

    reference_image, reference_background = create_reference_image(directories)

    #fig = figure()
    #axis = fig.add_subplot(111)
    #axis.imshow(reference_image, cmap = 'gray', interpolation = 'nearest', origin = 'lower')
    #fig.savefig('images/Reference_image.png')
    #plt.close(fig)

    window_size = 25
    number_reference_images = 3
    threshold_factor = 1.5
    minimum_flux = reference_background.globalrms * threshold_factor


    for dir in directories:
        files = sorted(glob.glob(dir+"*.pgm"))
        for i, x in enumerate(files[number_reference_images:190]):
            title = x.split('/')[-1]
            print title
            main(title, reference_image, reference_background, x, 100, window_size, threshold_factor, minimum_flux)

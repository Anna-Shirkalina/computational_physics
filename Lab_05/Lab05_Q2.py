import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as ft

# load the image into N by M np array
blured_img = np.loadtxt("blur.txt")
plt.figure(1)
plt.imshow(blured_img, cmap='gray')

# the number of columns and rows in the original image
N, M = blured_img.shape


def point_spread_gaussian(n, m, sigma):
    """
    Return the point spread function for an N X M image
    :param N: the number of rows in the image
    :param M: the number of columns in the image
    :param sigma: the standard deviation of the guassian
    :return: A N X M ndarray of the point spread function
    """
    x = np.arange(0, n)  # a one d array of the rows
    y = np.arange(0, m)  # a one d array of the columns

    f = np.zeros((n, m))  # initialize the point_spread array
    for i, row in enumerate(x):  # loop over the rows and columns
        if i > n // 2:  # to get the reflected guassian in each of the corners
            row = row - n  # reflect the x coordinate half-way through
        for j, column in enumerate(y):
            if j > m // 2:
                column = column - m  # reflect the x coordinate half-way through
            # calculate the point spread guassian
            f[i, j] = np.exp((row**2 + column**2) / (-2 * (sigma**2)))

    return f


point_spread = point_spread_gaussian(N, M, 25)

# display the point_spread guassian
plt.figure(2)
plt.imshow(point_spread, cmap="gray")
plt.colorbar(label='The Gaussian point map function', shrink=0.7,
             orientation='horizontal', pad=0.3)

# get the fourrier transform coefficients of the point_spread function
# and original image
ft_blurry_image = ft.rfft2(blured_img)
ft_noise = ft.rfft2(point_spread)

# compute the division for the new fourier coefficents of non-blurred image
ft_clear_img = np.divide(ft_blurry_image, ft_noise, out=np.zeros_like(ft_blurry_image), where= ft_noise > 1e-2) * (1 / (M * N))

# do the reverse fourier transform
clear_img = ft.irfft2(ft_clear_img)

# display the unblurred image
plt.figure(3)
plt.imshow(clear_img, cmap="gray")


plt.show()

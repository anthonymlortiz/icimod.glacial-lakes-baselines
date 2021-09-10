import numpy as np
from scipy.interpolate import RectBivariateSpline
from skimage.filters import sobel
from skimage import img_as_float
from scipy import ndimage as ndi
from itertools import cycle
from skimage.draw import polygon


def active_contour(image, snake, alpha=0.01, beta=0.1,
                   w_line=0, w_edge=1, gamma=0.01,
                   max_px_move=1.0,
                   max_num_iter=2500, convergence=0.1,
                   *,
                   boundary_condition='periodic'):
    """Active contour model.
    Active contours by fitting snakes to features of images. Supports single
    and multichannel 2D images. Snakes can be periodic (for segmentation) or
    have fixed and/or free ends.
    The output snake has the same length as the input boundary.
    As the number of points is constant, make sure that the initial snake
    has enough points to capture the details of the final contour.
    Parameters
    ----------
    image : (N, M) or (N, M, 3) ndarray
        Input image.
    snake : (N, 2) ndarray
        Initial snake coordinates. For periodic boundary conditions, endpoints
        must not be duplicated.
    alpha : float, optional
        Snake length shape parameter. Higher values makes snake contract
        faster.
    beta : float, optional
        Snake smoothness shape parameter. Higher values makes snake smoother.
    w_line : float, optional
        Controls attraction to brightness. Use negative values to attract
        toward dark regions.
    w_edge : float, optional
        Controls attraction to edges. Use negative values to repel snake from
        edges.
    gamma : float, optional
        Explicit time stepping parameter.
    max_px_move : float, optional
        Maximum pixel distance to move per iteration.
    max_num_iter : int, optional
        Maximum iterations to optimize snake shape.
    convergence : float, optional
        Convergence criteria.
    boundary_condition : string, optional
        Boundary conditions for the contour. Can be one of 'periodic',
        'free', 'fixed', 'free-fixed', or 'fixed-free'. 'periodic' attaches
        the two ends of the snake, 'fixed' holds the end-points in place,
        and 'free' allows free movement of the ends. 'fixed' and 'free' can
        be combined by parsing 'fixed-free', 'free-fixed'. Parsing
        'fixed-fixed' or 'free-free' yields same behaviour as 'fixed' and
        'free', respectively.
    Returns
    -------
    snake : (N, 2) ndarray
        Optimised snake, same shape as input parameter.
    References
    ----------
    .. [1]  Kass, M.; Witkin, A.; Terzopoulos, D. "Snakes: Active contour
            models". International Journal of Computer Vision 1 (4): 321
            (1988). :DOI:`10.1007/BF00133570`

    """
    max_num_iter = int(max_num_iter)
    if max_num_iter <= 0:
        raise ValueError("max_num_iter should be >0.")
    convergence_order = 10
    valid_bcs = ['periodic', 'free', 'fixed', 'free-fixed',
                 'fixed-free', 'fixed-fixed', 'free-free']
    if boundary_condition not in valid_bcs:
        raise ValueError("Invalid boundary condition.\n" +
                         "Should be one of: "+", ".join(valid_bcs)+'.')

    img = img_as_float(image)
    img = img.astype(float, copy=False)

    # Find edges using sobel:
    if w_edge != 0:
        if img.ndim == 1:
            edge = [sobel(img)]
        else:
            edge_new = [sobel(c) for c in np.moveaxis(img, -1, 0)]
            edge = np.stack(edge_new, axis=-1)
    else:
        edge = [0]

    # Superimpose intensity and edge images:
    if img.ndim > 1:
        img = w_line*np.sum(img, axis=2) \
            + w_edge*np.sum(edge)
    else:
        img = w_line*img + w_edge*edge[0]

    # Interpolate for smoothness:
    intp = RectBivariateSpline(np.arange(img.shape[1]),
                               np.arange(img.shape[0]),
                               img.T, kx=2, ky=2, s=0)

    snake_xy = snake[:, ::-1]
    x = snake_xy[:, 0].astype(float)
    y = snake_xy[:, 1].astype(float)
    n = len(x)
    xsave = np.empty((convergence_order, n), dtype=float)
    ysave = np.empty((convergence_order, n), dtype=float)

    # Build snake shape matrix for Euler equation in double precision
    eye_n = np.eye(n, dtype=float)
    a = (np.roll(eye_n, -1, axis=0)
         + np.roll(eye_n, -1, axis=1)
         - 2 * eye_n)  # second order derivative, central difference
    b = (np.roll(eye_n, -2, axis=0)
         + np.roll(eye_n, -2, axis=1)
         - 4 * np.roll(eye_n, -1, axis=0)
         - 4 * np.roll(eye_n, -1, axis=1)
         + 6 * eye_n)  # fourth order derivative, central difference
    A = -alpha * a + beta * b

    # Impose boundary conditions different from periodic:
    sfixed = False
    if boundary_condition.startswith('fixed'):
        A[0, :] = 0
        A[1, :] = 0
        A[1, :3] = [1, -2, 1]
        sfixed = True
    efixed = False
    if boundary_condition.endswith('fixed'):
        A[-1, :] = 0
        A[-2, :] = 0
        A[-2, -3:] = [1, -2, 1]
        efixed = True
    sfree = False
    if boundary_condition.startswith('free'):
        A[0, :] = 0
        A[0, :3] = [1, -2, 1]
        A[1, :] = 0
        A[1, :4] = [-1, 3, -3, 1]
        sfree = True
    efree = False
    if boundary_condition.endswith('free'):
        A[-1, :] = 0
        A[-1, -3:] = [1, -2, 1]
        A[-2, :] = 0
        A[-2, -4:] = [-1, 3, -3, 1]
        efree = True

    # Only one inversion is needed for implicit spline energy minimization:
    inv = np.linalg.inv(A + gamma * eye_n)
    # can use float_dtype once we have computed the inverse in double precision
    inv = inv.astype(float, copy=False)

    # Explicit time stepping for image energy minimization:
    for i in range(max_num_iter):
        # RectBivariateSpline always returns float64, so call astype here
        fx = intp(x, y, dx=1, grid=False).astype(float, copy=False)
        fy = intp(x, y, dy=1, grid=False).astype(float, copy=False)

        if sfixed:
            fx[0] = 0
            fy[0] = 0
        if efixed:
            fx[-1] = 0
            fy[-1] = 0
        if sfree:
            fx[0] *= 2
            fy[0] *= 2
        if efree:
            fx[-1] *= 2
            fy[-1] *= 2
        xn = inv @ (gamma*x + fx)
        yn = inv @ (gamma*y + fy)

        # Movements are capped to max_px_move per iteration:
        dx = max_px_move * np.tanh(xn - x)
        dy = max_px_move * np.tanh(yn - y)
        if sfixed:
            dx[0] = 0
            dy[0] = 0
        if efixed:
            dx[-1] = 0
            dy[-1] = 0
        x += dx
        y += dy

        # Convergence criteria needs to compare to a number of previous
        # configurations since oscillations can occur.
        j = i % (convergence_order + 1)
        if j < convergence_order:
            xsave[j, :] = x
            ysave[j, :] = y
        else:
            dist = np.min(np.max(np.abs(xsave - x[None, :])
                                 + np.abs(ysave - y[None, :]), 1))
            if dist < convergence:
                break

    return np.stack([y, x], axis=1)


class _fcycle(object):

    def __init__(self, iterable):
        """Call functions from the iterable each time it is called."""
        self.funcs = cycle(iterable)

    def __call__(self, *args, **kwargs):
        f = next(self.funcs)
        return f(*args, **kwargs)


# SI and IS operators for 2D and 3D.
_P2 = [np.eye(3),
       np.array([[0, 1, 0]] * 3),
       np.flipud(np.eye(3)),
       np.rot90([[0, 1, 0]] * 3)]
_P3 = [np.zeros((3, 3, 3)) for i in range(9)]

_P3[0][:, :, 1] = 1
_P3[1][:, 1, :] = 1
_P3[2][1, :, :] = 1
_P3[3][:, [0, 1, 2], [0, 1, 2]] = 1
_P3[4][:, [0, 1, 2], [2, 1, 0]] = 1
_P3[5][[0, 1, 2], :, [0, 1, 2]] = 1
_P3[6][[0, 1, 2], :, [2, 1, 0]] = 1
_P3[7][[0, 1, 2], [0, 1, 2], :] = 1
_P3[8][[0, 1, 2], [2, 1, 0], :] = 1


def _init_level_set(init_level_set, image_shape):
    """Auxiliary function for initializing level sets with a string.
    If `init_level_set` is not a string, it is returned as is.
    """
    if isinstance(init_level_set, str):
        if init_level_set == 'circle':
            res = circle_level_set(image_shape)
        elif init_level_set == 'ellipsoid':
            res = ellipsoid_level_set(image_shape)
        else:
            raise ValueError("`init_level_set` not in "
                             "['checkerboard', 'circle', 'ellipsoid']")
    else:
        res = init_level_set
    return res


def sup_inf(u):
    """SI operator."""

    if np.ndim(u) == 2:
        P = _P2
    elif np.ndim(u) == 3:
        P = _P3
    else:
        raise ValueError("u has an invalid number of dimensions "
                         "(should be 2 or 3)")

    erosions = []
    for P_i in P:
        erosions.append(ndi.binary_erosion(u, P_i))

    return np.array(erosions, dtype=np.int8).max(0)

def inf_sup(u):
    """IS operator."""

    if np.ndim(u) == 2:
        P = _P2
    elif np.ndim(u) == 3:
        P = _P3
    else:
        raise ValueError("u has an invalid number of dimensions "
                         "(should be 2 or 3)")

    dilations = []
    for P_i in P:
        dilations.append(ndi.binary_dilation(u, P_i))

    return np.array(dilations, dtype=np.int8).min(0)

_curvop = _fcycle([lambda u: sup_inf(inf_sup(u)),   # SIoIS
                   lambda u: inf_sup(sup_inf(u))])  # ISoSI

def circle_level_set(image_shape, center=None, radius=None):
    """Create a circle level set with binary values.
    Parameters
    ----------
    image_shape : tuple of positive integers
        Shape of the image
    center : tuple of positive integers, optional
        Coordinates of the center of the circle given in (row, column). If not
        given, it defaults to the center of the image.
    radius : float, optional
        Radius of the circle. If not given, it is set to the 75% of the
        smallest image dimension.
    Returns
    -------
    out : array with shape `image_shape`
        Binary level set of the circle with the given `radius` and `center`.
    See also
    --------
    ellipsoid_level_set
    checkerboard_level_set
    """

    if center is None:
        center = tuple(i // 2 for i in image_shape)

    if radius is None:
        radius = min(image_shape) * 3.0 / 8.0

    grid = np.mgrid[[slice(i) for i in image_shape]]
    grid = (grid.T - center).T
    phi = radius - np.sqrt(np.sum((grid)**2, 0))
    res = np.int8(phi > 0)
    return res


def level_set_from_polygon(shape, xy_polygon):
    img = np.zeros(shape, 'uint8')
    poly = xy_polygon
    rr, cc = polygon([a_tuple[1] for a_tuple in poly], [a_tuple[0] for a_tuple in poly], img.shape)
    img[rr,cc] = 1
    return img


def ellipsoid_level_set(image_shape, center=None, semi_axis=None):
    """Create a ellipsoid level set with binary values.
    Parameters
    ----------
    image_shape : tuple of positive integers
        Shape of the image
    center : tuple of integers, optional
        Coordinates of the center of the ellipsoid.
        If not given, it defaults to the center of the image.
    semi_axis : tuple of floats, optional
        Lengths of the semi-axis of the ellispoid.
        If not given, it defaults to the half of the image dimensions.
    Returns
    -------
    out : array with shape `image_shape`
        Binary level set of the ellipsoid with the given `center`
        and `semi_axis`.
    See also
    --------
    circle_level_set
    """

    if center is None:
        center = tuple(i // 2 for i in image_shape)

    if semi_axis is None:
        semi_axis = tuple(i / 2 for i in image_shape)

    if len(center) != len(image_shape):
        raise ValueError("`center` and `image_shape` must have the same length.")

    if len(semi_axis) != len(image_shape):
        raise ValueError("`semi_axis` and `image_shape` must have the same length.")

    if len(image_shape) == 2:
        xc, yc = center
        rx, ry = semi_axis
        phi = 1 - np.fromfunction(
            lambda x, y: ((x - xc) / rx) ** 2 +
                         ((y - yc) / ry) ** 2,
            image_shape, dtype=float)
    elif len(image_shape) == 3:
        xc, yc, zc = center
        rx, ry, rz = semi_axis
        phi = 1 - np.fromfunction(
            lambda x, y, z: ((x - xc) / rx) ** 2 +
                            ((y - yc) / ry) ** 2 +
                            ((z - zc) / rz) ** 2,
            image_shape, dtype=float)
    else:
        raise ValueError("`image_shape` must be a 2- or 3-tuple.")

    res = np.int8(phi > 0)
    return res


def _check_input(image, init_level_set):
    """Check that shapes of `image` and `init_level_set` match."""
    if image.ndim not in [2, 3]:
        raise ValueError("`image` must be a 2 or 3-dimensional array.")

    if len(image.shape) != len(init_level_set.shape):
        raise ValueError("The dimensions of the initial level set do not "
                         "match the dimensions of the image.")


def inverse_gaussian_gradient(x, alpha=100.0, sigma=5.0):
    """Inverse of gradient magnitude.
    Compute the gaussian of magnitude of the gradients in the image and then inverts the
    result in the range [0, 1]. Flat areas are assigned values close to 1,
    while areas close to borders are assigned values close to 0.
    """
    gradnorm = ndi.gaussian_gradient_magnitude(x, sigma, mode='nearest')
    return 1.0 / np.sqrt(1.0 + alpha * gradnorm)


def morphological_geodesic_active_contour(gimage, iterations,
                                          init_level_set='circle', smoothing=1,
                                          threshold='auto', balloon=0,
                                          iter_callback=lambda x: None):
    """Morphological Geodesic Active Contours (MorphGAC).
    Geodesic active contours implemented with morphological operators. It can
    be used to segment objects with visible but noisy, cluttered, broken
    borders.
    Parameters
    ----------
    gimage : (M, N) or (B, M, N) array
        Preprocessed image or volume to be segmented. This is very rarely the
        original image. Instead, this is usually a preprocessed version of the
        original image that enhances and highlights the borders (or other
        structures) of the object to segment.
        `morphological_geodesic_active_contour` will try to stop the contour
        evolution in areas where `gimage` is small. See
        `morphsnakes.inverse_gaussian_gradient` as an example function to
        perform this preprocessing. Note that the quality of
        `morphological_geodesic_active_contour` might greatly depend on this
        preprocessing.
    iterations : uint
        Number of iterations to run.
    init_level_set : str, (M, N) array, or (B, M, N) array
        Initial level set. If an array is given, it will be binarized and used
        as the initial level set. If a string is given, it defines the method
        to generate a reasonable initial level set with the shape of the
        `image`. Accepted values are 'checkerboard' and 'circle'. See the
        documentation of `checkerboard_level_set` and `circle_level_set`
        respectively for details about how these level sets are created.
    smoothing : uint, optional
        Number of times the smoothing operator is applied per iteration.
        Reasonable values are around 1-4. Larger values lead to smoother
        segmentations.
    threshold : float, optional
        Areas of the image with a value smaller than this threshold will be
        considered borders. The evolution of the contour will stop in this
        areas.
    balloon : float, optional
        Balloon force to guide the contour in non-informative areas of the
        image, i.e., areas where the gradient of the image is too small to push
        the contour towards a border. A negative value will shrink the contour,
        while a positive value will expand the contour in these areas. Setting
        this to zero will disable the balloon force.
    iter_callback : function, optional
        If given, this function is called once per iteration with the current
        level set as the only argument. This is useful for debugging or for
        plotting intermediate results during the evolution.
    Returns
    -------
    out : (M, N) or (B, M, N) array
        Final segmentation (i.e., the final level set)
    See also
    --------
    inverse_gaussian_gradient, circle_level_set
    Notes
    -----
    This is a version of the Geodesic Active Contours (GAC) algorithm that uses
    morphological operators instead of solving partial differential equations
    (PDEs) for the evolution of the contour. The set of morphological operators
    used in this algorithm are proved to be infinitesimally equivalent to the
    GAC PDEs (see [1]_). However, morphological operators are do not suffer
    from the numerical stability issues typically found in PDEs (e.g., it is
    not necessary to find the right time step for the evolution), and are
    computationally faster.
    The algorithm and its theoretical derivation are described in [1]_.
    References
    ----------
    .. [1] A Morphological Approach to Curvature-based Evolution of Curves and
           Surfaces, Pablo Márquez-Neila, Luis Baumela, Luis Álvarez. In IEEE
           Transactions on Pattern Analysis and Machine Intelligence (PAMI),
           2014, DOI 10.1109/TPAMI.2013.106
    """

    image = gimage
    init_level_set = _init_level_set(init_level_set, image.shape)

    _check_input(image, init_level_set)

    if threshold == 'auto':
        threshold = np.percentile(image, 50)

    structure = np.ones((3,) * len(image.shape), dtype=np.int8)
    dimage = np.gradient(image)
    # threshold_mask = image > threshold
    if balloon != 0:
        threshold_mask_balloon = image > threshold / np.abs(balloon)

    u = np.int8(init_level_set > 0)

    iter_callback(u)

    for _ in range(iterations):

        # Balloon
        if balloon > 0:
            aux = ndi.binary_dilation(u, structure)
        elif balloon < 0:
            aux = ndi.binary_erosion(u, structure)
        if balloon != 0:
            u[threshold_mask_balloon] = aux[threshold_mask_balloon]

        # Image attachment
        aux = np.zeros_like(image)
        du = np.gradient(u)
        for el1, el2 in zip(dimage, du):
            aux += el1 * el2
        u[aux > 0] = 1
        u[aux < 0] = 0

        # Smoothing
        for _ in range(smoothing):
            u = _curvop(u)

        iter_callback(u)

    return u


def rgb2gray(img):
    """Convert a RGB image to gray scale."""
    return 0.2989 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store


def snake_lakes_GAC(x, ls_center_point, iterations=300, ls_radious=10, igs_alpha=100, igs_sigma=2):
    # Load the image.
    imgcolor = x/256.
    img = rgb2gray(imgcolor)

    gimg = inverse_gaussian_gradient(img, alpha=igs_alpha, sigma=igs_sigma)

    # Initialization of the level-set with a circle. ls_center_point should be 
    # coordinate tuple (x, y) in the center of the known polygon
    # ls_radious radious of the circle level set
    init_ls = circle_level_set(img.shape, ls_center_point, ls_radious)

    # Callback for visual plotting
    evolution = []
    callback = store_evolution_in(evolution)

    # Morphological GAC
    return morphological_geodesic_active_contour(gimg, iterations=iterations,
                                      init_level_set=init_ls,
                               smoothing=2, threshold='auto',
                                balloon=1, iter_callback=callback
                              ).astype(np.uint8), evolution


def snake_lakes_GAC_from_polygon(x, polygon, iterations=300, igs_alpha=100, igs_sigma=2.25):
    # Load the image.
    imgcolor = x/256.
    img = rgb2gray(imgcolor)


    gimg = inverse_gaussian_gradient(img, alpha=igs_alpha, sigma=igs_sigma)
    

    # Initialization of the level-set.
    init_ls = level_set_from_polygon(img.shape, polygon)

    # Callback for visual plotting
    evolution = []
    callback = store_evolution_in(evolution)
    # Morphological GAC
    return morphological_geodesic_active_contour(gimg, iterations=iterations,
                                      init_level_set=init_ls,
                               smoothing=2, threshold='auto',
                                balloon=1, iter_callback=callback
                              ).astype(np.uint8), evolution
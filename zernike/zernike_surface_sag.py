"""Zernike surface creation and visualization

The Zernike polynomial geometry represents a surface defined by a Zernike
polynomial in two dimensions. The surface is defined as:

z(x,y) = z_{base}(x,y) + r^2 / (R * (1 + sqrt(1 - (1 + k) * r^2 / R^2))) +
    sum_i [c[i] * Z_i(rho, theta)]

where:
- r^2 = x^2 + y^2
- R is the radius of curvature
- k is the conic constant
- c[i] is the coefficient for the i-th Fringe Zernike polynomial
- Z_i(...) is the i-th Fringe Zernike polynomial in polar coordinates
- rho = sqrt(x^2 + y^2) / normalization, theta = atan2(y, x)

Zernike polynomials are a set of orthogonal functions defined over the unit disk, 
widely used in freeform optical surface design. 
They efficiently describe wavefront aberrations and complex surface deformations 
by decomposing them into radial and azimuthal components. 
Their orthogonality ensures minimal cross-coupling between terms, 
making them ideal for optimizing optical systems. 
In freeform optics, they enable precise control of surface shape, 
improving performance beyond traditional spherical and aspheric designs.

drpaprika, 2025
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12, 'font.family': 'STIXGeneral'})


def factorial(n):
    return np.prod(range(1, n+1))


class ZernikePolynomialGeometry:
    """
    Represents a Fringe Zernike polynomial geometry defined as:

    z(x,y) = z_{base}(x,y) + r^2 / (R * (1 + sqrt(1 - (1 + k) * r^2 / R^2))) +
        sum_i [c[i] * Z_i(rho, theta)]

    where:
    - r^2 = x^2 + y^2
    - R is the radius of curvature
    - k is the conic constant
    - c[i] is the coefficient for the i-th Fringe Zernike polynomial
    - Z_i(...) is the i-th Fringe Zernike polynomial in polar coordinates
    - rho = sqrt(x^2 + y^2) / normalization, theta = atan2(y, x)

    The coefficients are defined in a 1D array where coefficients[i] is the
    coefficient for Z_i.

    Args:
        radius (float): The radius of curvature of the geometry.
        conic (float, optional): The conic constant of the geometry.
            Defaults to 0.0.
        coefficients (list or np.ndarray, optional): The coefficients of the
            Zernike polynomial surface. Defaults to an empty list, indicating
            no Zernike polynomial coefficients are used.
        norm_radius (int, optional): The normalization radius.
            Defaults to 1.
    """

    def __init__(self,
                 radius: float,
                 conic: float = 0.0,
                 norm_radius: float = 1,
                 coefficients: np.ndarray = None,
        ):
        self.radius = radius
        self.conic = conic
        self.norm_radius = norm_radius
        self.coefficients = np.atleast_1d(coefficients)

    def __str__(self)->str:
        return 'Zernike Polynomial'

    def sag(self, x: np.ndarray, y: np.ndarray)->np.ndarray:
        """
        Calculate the sag of the Zernike polynomial surface at the given
        coordinates.

        Args:
            x (float, np.ndarray): The Cartesian x-coordinate(s).
            y (float, np.ndarray): The Cartesian y-coordinate(s).

        Returns:
            np.ndarray: The sag value at the given Cartesian coordinates.
        """
        self.norm_radius = max(np.max(x), np.max(y))
        x_norm = x / self.norm_radius
        y_norm = y / self.norm_radius

        self._validate_inputs(x_norm, y_norm)

        # Convert to local polar
        rho = np.sqrt(x_norm**2 + y_norm**2)
        theta = np.arctan2(y_norm, x_norm)

        # Base conic 
        z = (rho**2) / (self.radius * (1 + np.sqrt(1 - (1 + self.conic) * (rho**2) / self.radius**2)))

        # Add normalized Fringe Zernike contributions
        # Sum over all nonzero coefficients
        non_zero_indices = np.nonzero(self.coefficients)[0]
        for i in non_zero_indices:
            normalization_factor = np.sqrt(2*(i+1)/np.pi)
            z += normalization_factor * self.coefficients[i] * self._zernike(i+1, rho, theta)

        return z
        
    def _zernike(self, i: int, rho: np.ndarray, theta: np.ndarray)->np.ndarray:
        """
        Calculate the i-th Fringe Zernike polynomial at the given rho, theta.

        Args:
            i (int): The degree of the Zernike polynomial.
            rho (np.ndarray): Radial coordinate.
            theta (np.ndarray): Azimuthal coordinate.

        Returns:
            float or np.ndarray: Z_i(rho, theta).
        """
        n, m = self._fringezernike_order_to_zernike_order(i)
        Rnm = self._radial_poly(n, abs(m), rho)

        if m == 0:
            return Rnm
        elif m > 0:
            return Rnm * np.cos(m * theta)
        else: 
            return Rnm * np.sin(abs(m) * theta)

    def _zernike_derivative(self, i: int, rho: np.ndarray, theta: np.ndarray)->tuple[np.ndarray, np.ndarray]:
        """
        Return partial derivatives of Z_i w.r.t. rho and theta:
        (dZ/drho, dZ/dtheta).
        We'll use them in chain rule for partial derivatives w.r.t x,y.

        Args:
            i (int): Fringe Zernike index
            rho (float or np.ndarray): radial coordinate
            theta (float or np.ndarray): azimuthal coordinate

        Return:
            (dZdrho, dZdtheta) as floats or ndarrays
        """
        n, m = self._fringezernike_order_to_zernike_order(i)
        Rnm = self._radial_poly(n, abs(m), rho)
        dRnm = self._radial_poly_derivative(n, abs(m), rho)

        if m == 0:
            # Z_n^0(rho,theta) = R_n^0(rho), no theta dependence
            dZdrho = dRnm
            dZdtheta = 0.0
        elif m > 0:
            # Z_n^m = R_n^m(rho)*cos(m*theta)
            # d/d rho -> dR_n^m(rho)*cos(m*theta)
            dZdrho = dRnm * np.cos(m * theta)
            # d/d theta -> R_n^m(rho)*(-m sin(m theta))
            dZdtheta = -m * Rnm * np.sin(m * theta)
        else:
            # m < 0 => Z_n^m = R_n^|m|(rho)*sin(|m|*theta)
            dZdrho = dRnm * np.sin(abs(m) * theta)
            dZdtheta = abs(m) * Rnm * np.cos(abs(m) * theta)

        return dZdrho, dZdtheta

    def _radial_poly(self, n: int, m: int, rho: np.ndarray)->np.ndarray:
        """
        Compute the radial polynomial R_n^m(rho).

        R_n^m(rho) = sum_{k=0}^{(n - m)/2} (-1)^k * (n-k)! /
                     [k! ((n+m)/2 - k)! ((n-m)/2 - k)!] * rho^(n - 2k)

        Args:
            n (int): Zernike n
            m (int): Zernike m (assumed >= 0)
            rho (float or np.ndarray): Radial coordinate.

        Returns:
            float or np.ndarray: The radial polynomial evaluated at rho.
        """
        # sum_{k=0}^{(n - m)//2} (-1)^k * (n-k)! / [k!((n+m)/2 - k)!((n-m)/2 - k)!] * rho^(n-2k)
        val = 0.0
        upper_k = (n - m) // 2
        for k in range(upper_k + 1):
            sign = (-1)**k
            numerator = factorial(n - k)
            denominator = (
                factorial(k) *
                factorial((n + m)//2 - k) *
                factorial((n - m)//2 - k)
            )
            val += sign * (numerator / denominator) * (rho ** (n - 2*k))
        return val

    def _radial_poly_derivative(self, n: int, m: int, rho: np.ndarray):
        """
        Derivative of the radial polynomial R_n^m(rho) with respect to rho.

        d/d(rho) R_n^m(rho) = sum_{k=0} (...) (n-2k) * rho^(n-2k-1)

        Args:
            n (int): Zernike n.
            m (int): Zernike m (assumed >= 0).
            rho (float or np.ndarray): radial coordinate.

        Returns:
            float or np.ndarray: d/d rho of R_n^m(rho).
        """
        val = 0.0
        upper_k = (n - m) // 2
        for k in range(upper_k + 1):
            sign = (-1)**k
            numerator = factorial(n - k)
            denominator = (factorial(k) *
                           factorial((n + m)//2 - k) *
                           factorial((n - m)//2 - k))
            factor = (n - 2*k)
            if factor < 0:
                continue
            power_term = rho**(n - 2*k - 1) if (n - 2*k - 1) >= 0 else 0
            val += sign * (numerator / denominator) * factor * power_term
        return val
    
    def _surface_normal(self, x: np.ndarray, y: np.ndarray)->tuple[float, float, float]:
        """
        Calculate the surface normal of the full surface (conic + Zernike)
        in Cartesian coordinates at (x, y).

        Args:
            x (float or np.ndarray): x-coordinate(s).
            y (float or np.ndarray): y-coordinate(s).

        Returns:
            (nx, ny, nz): Normal vector components in Cartesian coords.
        """
        # Conic partial derivatives:
        r2 = x**2 + y**2
        denom = self.radius * np.sqrt(1.0 - (1.0 + self.k) * r2 / self.radius**2)

        # Protect against divide-by-zero for r=0
        # or handle small r if needed
        eps = 1e-14
        denom = np.where(np.abs(denom) < eps, eps, denom)

        dzdx = x / denom
        dzdy = y / denom

        # Now add partial derivatives from the Zernike expansions
        x_norm = x / self.norm_radius
        y_norm = y / self.norm_radius
        rho = np.sqrt(x_norm**2 + y_norm**2)
        theta = np.arctan2(y_norm, x_norm)

        # Chain rule:
        # dZ/dx = dZ/drho * d(rho)/dx + dZ/dtheta * d(theta)/dx
        # We'll define the partials of (rho,theta) wrt x:
        #   drho/dx    = x / (norm_x^2 * rho)
        #   dtheta/dx  = - y / (rho^2 * norm_y * norm_x)
        drho_dx = np.zeros_like(x) if np.all(rho == 0) else (x/(self.norm_radius**2)) / (rho + eps)
        drho_dy = np.zeros_like(y) if np.all(rho == 0) else (y/(self.norm_radius**2)) / (rho + eps)
        dtheta_dx = -(y_norm)/(rho**2 + eps) * (1.0/self.norm_radius)
        dtheta_dy = +(x_norm)/(rho**2 + eps) * (1.0/self.norm_radius)

        non_zero_indices = np.nonzero(self.coefficients)[0]
        for i in non_zero_indices:
            dZdrho, dZdtheta = self._zernike_derivative(i, rho, theta)
            # partial wrt x
            dzdx += self.coefficients[i] * (dZdrho * drho_dx + dZdtheta * dtheta_dx)
            # partial wrt y
            dzdy += self.coefficients[i] * (dZdrho * drho_dy + dZdtheta * dtheta_dy)

        # Surface normal vector in cartesian coordinates: (-dzdx, -dzdy, 1) normalized
        # Check sign conventions!   
        nx = +dzdx
        ny = +dzdy
        nz = -np.ones_like(x)
        norm = np.sqrt(nx*nx + ny*ny + nz*nz)
        norm = np.where(norm < eps, 1.0, norm)     # Avoid division by zero
        nx /= norm
        ny /= norm
        nz /= norm

        return (nx, ny, nz)

    @staticmethod
    def _fringezernike_order_to_zernike_order(k: int)->tuple[float, float]:
        """Convert Fringe Zernike index k to classical Zernike (n, m).
        https://wp.optics.arizona.edu/visualopticslab/wp-content/uploads/sites/52/2021/10/Zernike-Fit.pdf"""
        n = np.ceil((-3 + np.sqrt(9 + 8*k))/2)
        m = 2 * k - n * (n + 2)
        return (n.astype(int), m.astype(int))
    
    def _validate_inputs(self, x_norm: float, y_norm: float)->None:
        """
        Validate the input coordinates for the Zernike polynomial surface.

        Args:
            x_norm, y_norm (np.ndarray): The normalized x & y values.
        """
        if np.any(np.abs(x_norm) > 1) or np.any(np.abs(y_norm) > 1):
            raise ValueError('Zernike coordinates must be normalized '
                             'to [-1, 1]. Consider updating the normalization' 
                             'radius to 1.1x the surface aperture')


@dataclass
class SurfaceViewer:

    def view(self,
             x: np.ndarray,
             y: np.ndarray,
             sag: np.ndarray,
             semi_aperture: float = None,
             projection: str = '2d',
             title: str = None,
        ) -> None:
        """
        Visualize the surface.

        Args:
            x, y (ndarray): x & y coordinates.
            sag (ndarray): Sag surface data.
            semi_aperture (float): Semi-aperture (assuming circular aperture)
            projection (str): The type of projection to use for visualization.
                Can be '2d' or '3d'.
            title (str): Title.

        Raises:
            ValueError: If the projection is not '2d' or '3d'.
        """
        sag[np.sqrt(x**2+y**2) > semi_aperture] = np.nan

        if projection == '2d':
            self._plot_2d(x, y, sag, title=title)
        elif projection == '3d':
            self._plot_3d(x, y, sag, title=title)
        else:
            raise ValueError('Surface projection must be "2d" or "3d".')

    def _plot_2d(self, x, y, z, title):
        """
        Plot a 2D representation of the given data.

        Args:
            x (numpy.ndarray): Array of x-coordinates.
            y (numpy.ndarray): Array of y-coordinates.
            z (numpy.ndarray): Array of z-coordinates.
            title (str): Title.
        """
        _, ax = plt.subplots(figsize=(7, 5.5))
        extent = [np.min(x), np.max(x), np.min(y), np.max(y)]
        im = ax.imshow(np.flipud(z), extent=extent)

        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_title(title)
        cbar = plt.colorbar(im)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel("Deviation to plane [mm]", rotation=270)
        plt.grid(alpha=0.25)
        plt.show()

    def _plot_3d(self, x, y, z, title):
        """
        Plot a 3D surface plot of the given data.

        Args:
            x (numpy.ndarray): Array of x-coordinates.
            y (numpy.ndarray): Array of y-coordinates.
            z (numpy.ndarray): Array of z-coordinates.
            title (str): Title.
        """
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"},
                               figsize=(7, 5.5))

        surf = ax.plot_surface(x, y, z,
                               rstride=1, cstride=1,
                               cmap='viridis', linewidth=0,
                               antialiased=False)
        
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel("Deviation to plane [mm]")
        ax.set_title(title)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.15)
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    import random
    # Create a Zernike surface
    poly_order = 25
    zernike_geometry = ZernikePolynomialGeometry(
        radius=100,
        coefficients=[random.choice([-1, 1])*1e-1 for _ in range(poly_order)],
    )
    semi_aperture = 5
    x, y = np.meshgrid(np.linspace(-semi_aperture, semi_aperture, 256),
                       np.linspace(-semi_aperture, semi_aperture, 256))
    sag = zernike_geometry.sag(x, y)
    
    # Visualize the surface
    viewer = SurfaceViewer()
    viewer.view(x=x, y=y, sag=sag,
                semi_aperture=semi_aperture,
                title=f'Zernike freeform surface'
                      f'\n{poly_order} coefficients')

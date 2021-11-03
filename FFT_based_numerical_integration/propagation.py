import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from configparser import ConfigParser


class BeamPropagation:
    """
    Computes the transverse profile of a set of 2 Hermite-Gaussian modes of given orders
    (m1, n1) and (m2, n2) after numerical propagation over a given distance z in free space.
    """

    def __init__(self):

        # Read the arguments from the config file
        self.m1, self.n1 = config.getint('main', 'm1'), config.getint('main', 'n1')     # 1st mode orders
        self.m2, self.n2 = config.getint('main', 'm2'), config.getint('main', 'n2')     # 2nd mode orders
        self.w0  = config.getfloat('main', 'w0')           # Beams waist [µm]
        self.lam = config.getfloat('main', 'lam')          # Wavelength  [µm]
        self.z   = config.getfloat('main', 'z')            # Propagation distance [µm]
        self.N   = config.getint('main', 'N')              # Number of points in thz aperture square side
        self.aperture_side = config.getfloat('main', 'as') # Aperture square side size [µm]

        # Simpson's rule requires an odd N
        if self.N % 2 == 0:
            print(f"Simpson's rule requires an odd N : incrementing N by 1 ({self.N}→{self.N+1}).")
            self.N += 1

        # Print info
        self.rayleigh_range = np.pi/self.lam*self.w0**2.0
        delta_s = self.aperture_side/self.N
        print(f'Aperture size : {self.aperture_side}x{self.aperture_side} µm')
        print(f'Rayleigh range : {round(self.rayleigh_range,2)} µm')
        print(f'Propagation distance : {self.z} µm')
        print('Sampling number   in the aperture plane : {}'.format(self.N))
        print('Sampling interval in the aperture plane : {} µm = {}λ'.format(round(delta_s,2), round(delta_s/self.lam,2)))

        # Setup the sampling grids
        # Aperture plane grid
        self.s = np.linspace(-self.aperture_side/2, self.aperture_side/2, self.N)
        self.n = self.s
        self.ss, self.nn = np.meshgrid(self.s, self.n)
        # Observation plane grid
        self.x = np.linspace(-self.aperture_side/2, self.aperture_side/2, self.N)
        self.y = np.linspace(-self.aperture_side/2, self.aperture_side/2, self.N)
        self.xx, self.yy = np.meshgrid(self.x, self.y)


    def compute_U(self, mode:np.array):
        """
        Computes U from a mode array, using Simpson's rule and padding the result to 2N-1,
        using equations 17 & 11.
        :param mode: light field of the 2D H-G mode
        :type mode: np.array
        :return: U
        """

        # Simpson's rule
        B = (1/3)*np.array([1] + [4,2]*int(self.N/2-1) + [4, 1])
        W = np.dot(np.transpose(B), B)
        U = np.multiply(W, mode)

        # Zero-pad to 2N-1
        U = np.pad(U,
                   [(0, self.N-1),   # padding for first dimension
                    (0, self.N-1)],  # padding for second dimension
                   mode='constant',
                   constant_values=0)

        return U


    def compute_H(self, z:float):
        """
        Computes the matrix H, using the impulse response g of the medium (here, free space),
        using equations 12, 13 & 14.
        :param z: propagation distance
        :type z:  float
        :return:  H
        """

        X = np.zeros(2*self.N-1)
        Y = np.zeros(2*self.N-1)
        for j in range(2*self.N-1):
            if j < self.N:
                X[j] = self.x[0] - self.s[self.N-1-j]
                Y[j] = self.y[0] - self.n[self.N-1-j]
            else:
                X[j] = self.x[j-self.N-1] - self.s[0]
                Y[j] = self.y[j-self.N-1] - self.n[0]

        XX, YY = np.meshgrid(X, Y)
        H = self.impulse_response(XX, YY, z)

        return H


    def compute_S(self, mode:np.array, z:float):
        """
        Computes the light field after a given distance,
        using equation 10.
        :param mode: light field of the 2D H-G mode
        :type mode: np.array
        :param z: propagation distance
        :type z: float
        :return:
        """

        U = self.compute_U(mode)
        H = self.compute_H(z)

        # Sampling intervals in the aperture plane
        delta_s = self.aperture_side/self.N
        delta_n = self.aperture_side/self.N

        S = np.fft.ifft2(np.multiply(np.fft.fft2(U), np.fft.fft2(H))) * delta_s*delta_n

        return S[self.N-1:,self.N-1:]


    def plot(self, animate:bool):
        """
        Plots the result, animates it as an option.
        :param animate: animate the last subplot from 0 to self.z
        :type animate: bool
        :return:
        """

        if animate:
            # Necessary in PyCharm
            import matplotlib
            matplotlib.use("TkAgg")

        plt.style.use('dark_background')    # Dark theme
        fig, ax = plt.subplots(2, 3, figsize=(15,9))

        # Before propagation --------------------------------------------------

        # Compute the three modes before propagation
        mode1_before = self.two_dim_hg_mode(self.m1, self.n1, self.ss, self.nn)
        mode2_before = self.two_dim_hg_mode(self.m2, self.n2, self.ss, self.nn)
        mode3_before = mode1_before * mode2_before

        # First plot - First mode before propagation
        # plot1 = ax[0,0].pcolor(abs(mode1_before)**2)
        plot1 = ax[0,0].pcolor(self.s, self.n, abs(mode1_before)**2, shading ='auto')
        ax[0,0].set_title(f'First mode {self.m1, self.n1} before propagation')
        ax[0,0].set_xlabel('x [µm]')
        ax[0,0].set_ylabel('y [µm]')
        ax[0,0].axis('equal')
        plt.colorbar(plot1, ax=ax[0, 0])

        # Second plot - Second mode before propagation
        plot2 = ax[0,1].pcolor(self.s, self.n, abs(mode2_before)**2, shading ='auto')
        ax[0,1].set_title(f'Second mode {self.m2, self.n2} before propagation')
        ax[0,1].set_xlabel('x [µm]')
        ax[0,1].set_ylabel('y [µm]')
        ax[0,1].axis('equal')
        plt.colorbar(plot2, ax=ax[0, 1])

        # Third plot - Overlap before propagation
        plot3 = ax[0,2].pcolor(self.s, self.n, abs(mode3_before)**2, shading ='auto')
        ax[0,2].set_title('Overlap before propagation\nz = 0 µm')
        ax[0,2].set_xlabel('x [µm]')
        ax[0,2].set_ylabel('y [µm]')
        ax[0,2].axis('equal')
        plt.colorbar(plot3, ax=ax[0, 2])

        # After propagation --------------------------------------------------

        # Compute the three modes after propagation
        mode1_after = self.compute_S(mode=mode1_before, z=self.z)
        mode2_after = self.compute_S(mode=mode2_before, z=self.z)
        mode3_after = self.compute_S(mode=mode3_before, z=self.z)

        # Fourth plot - First mode after propagation
        plot4 = ax[1,0].pcolor(self.x, self.y, abs(mode1_after)**2, shading ='auto')
        ax[1,0].set_title(f'First mode {self.m1, self.n1} after propagation')
        ax[1,0].set_xlabel('x [µm]')
        ax[1,0].set_ylabel('y [µm]')
        ax[1,0].axis('equal')
        plt.colorbar(plot4, ax=ax[1, 0])

        # Fifth plot - Second mode after propagation
        plot5 = ax[1,1].pcolor(self.x, self.y, abs(mode2_after)**2, shading ='auto')
        ax[1,1].set_title(f'Second mode {self.m2, self.n2} after propagation')
        ax[1,1].set_xlabel('x [µm]')
        ax[1,1].set_ylabel('y [µm]')
        ax[1,1].axis('equal')
        plt.colorbar(plot5, ax=ax[1, 1])

        # Sixth plot - Overlap after propagation
        plot6 = ax[1,2].pcolor(self.x, self.y, abs(mode3_after)**2, shading ='auto')
        ax[1,2].set_title(f'Overlap after propagation\nz = {self.z} µm')
        ax[1,2].set_xlabel('x [µm]')
        ax[1,2].set_ylabel('y [µm]')
        ax[1,2].axis('equal')
        plt.colorbar(plot6, ax=ax[1, 2])

        # Animate the propagation plot ------------------------------------------------------------
        if animate:

            frames_num = 10
            z_increment = self.z // frames_num

            # First frame
            ax[1,2].pcolormesh(abs(self.compute_S(mode=mode3_before, z=1e-9))**2)
            ax[1,2].set_title(f'z = {0} µm')

            def update_plot(t):
                z = (t+1)*z_increment
                ax[1,2].cla()
                plot = ax[1,2].pcolormesh(abs(self.compute_S(mode=mode3_before, z=z)**2))
                ax[1,2].set_title(f'z = {z} µm')
                return plot

            anim = animation.FuncAnimation(fig, update_plot, frames=frames_num, interval=400, repeat_delay=1000)

        fig.subplots_adjust(hspace=0.3)
        plt.show()


    """ Tools --------------------------------------------------------------------------------------------------  """

    def hermite_poly(self, order:int, x:np.array):
        """
        Recursive method that returns a 1D Hermite polynomial of given order
        :param order: order of the 1D polynomial
        :type order: int
        :param x: values to compute the polynomial over
        :type x: np.array
        :return:
        """

        if order == 0:
            return 1
        elif order == 1:
            return 2*x
        else:
            return 2*x*self.hermite_poly(order-1, x) - 2*(order-1)*self.hermite_poly(order-2, x)


    def one_dim_hg_mode(self, order:int, x:np.array, z:float):
        """
        Compute the 1D electric field distribution at plane z
        SPIE - T. Sean Ross - Laser Beam Quality Metrics, page 24.
        :param order: order of the 1D polynomial
        :type order: int
        :param x: values to compute the polynomial over
        :type x: np.array
        :param z: distance
        :type x: float
        :return:
        """

        # Intermediate variables
        zR = self.rayleigh_range
        W  = self.w0*np.sqrt(1 + (z/zR)**2)                    # Gaussian beam radius
        R  = np.sqrt(z**2 + (np.pi*self.w0**2/self.lam)**2)    # Wavefront radius of curvature

        # Electric field amplitude
        return (2/np.pi)**(1/4) \
               * np.sqrt((2**order)*math.factorial(order)*W) * ((zR+1j*z)/R)**(order+1/2) \
               * self.hermite_poly(order, np.sqrt(2)*x/W) \
               * np.exp(-1j*(np.pi*(x**2)*z)/(2*self.lam*R**2) - x**2/(W**2))


    def two_dim_hg_mode(self, m:int, n:int, x:np.array, y:np.array):
        """
        Returns a 2D Hermite-Gauss polynomial at z=0
        :param m: order of the 1st 1D Hermite poly
        :type m: int
        :param n: order of the 2nd 1D Hermite poly
        :type n: int
        :param x:
        :param y:
        :return:
        """

        mode_amplitude = self.one_dim_hg_mode(m, x, 0) * \
                         self.one_dim_hg_mode(n, y, 0)
        # mode_intensity = abs(mode_amplitude)**2

        return mode_amplitude


    def impulse_response(self, x:np.array, y:np.array, z:float):
        """
        Computes the impulse response g of free-space at given parameters.
        :param x:
        :param y:
        :param z:
        :return:
        """

        r = np.sqrt(x**2 + y**2 + z**2)
        k = 2*np.pi/self.lam         # Wavenumber of light

        return 1/(2*np.pi) * np.exp(1j*k*r)/r * z/r * (1/r * 1j*k)



""" -------------------------------------------------------------------------------"""

# Read the arguments from the command line
parser = argparse.ArgumentParser()
# settings_cfg = ConfigParser(inline_comment_prefixes="#")
parser.add_argument("--animate-True",  default=False, action="store_true",  help="Animate the propagation plot")
parser.add_argument("--animate-False", default=False, action="store_false", help="Do not animate the propagation plot")
parser.add_argument("file_path", type=Path, help="Path to the config file")
args = parser.parse_args()

# Read the config filename (.ini) from the command line
config = ConfigParser(inline_comment_prefixes="#")
config.read(args.file_path)

# Run the script
propagator = BeamPropagation()
propagator.plot(animate=args.animate_True)
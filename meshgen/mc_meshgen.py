import os
import numpy as np
import pandas as pd 
from scipy.linalg import norm
from scipy.stats import uniform
from scipy.integrate import quad
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from celluloid import Camera # for animation

import cProfile # profiler, for test only


class ParticlePlacementSimulation:
    """
    A class to perform particle placement simulation in 3D space.
    """
    def __init__(self, sphere_radius,
                 ellipse_major_axis, ellipse_minor_axis, desired_area_per_point,
                 max_iteration = 2000, dt = 0.1, scale = 1.0,
                 output_dir = "output/", anim_dir = "anim/"):
        """
        Initialize the simulation parameters.

        Args:
            sphere_radius (float): Radius of the sphere.
            ellipse_major_axis (float): Major axis of the ellipse.
            ellipse_minor_axis (float): Minor axis of the ellipse.
            desired_area_per_point (float): Desired area per particle.
        """
        # 4.7 nm spacing is calculated from previous result (Y Qian)
        # Parameter for changing number of nodes (nnode)
        self.desiredlgag = 4.7 * scale # desired gag-gag bond length
        self.minlgag = 4.6 * scale # if lgag is smaller than this than add node
        self.maxlgag = 4.8 * scale # if lgag is bigger than this than remove node
        self.changeNodeInterval = 500 # minimum step interval between changing number of nodes
        self.changeNodeCountdown = -1 # count down for changing number of nodes;
                                      # this is initialized to -1, then reset to self.changeNodeInterval
                                      # every time a node is changed
                                      # If it is negative, the program allow for change of 
                                      # number of nodes
        self.nodeChangeRatio = 0.1 # every time a node change happens, nnode is multiplied
                                   # (1.0 +/- self.nodeChangeRatio), based on current lgag
        self.nodeChangeRatioStep = 0.6 # every time after a node change, self.nodeChangeRatio is
                                       # multiplied by this value
        self.nodeChanges = 0 # bookkeeping for times of node change
        self.maxNnodeChanges = 100 # if changing lgag too many times then stop changing nnode
        
        # Geometric Parameter
        self.R = sphere_radius  # radius of gag lattice sphere, usually 50 nm or 60 nm
        self.a = ellipse_major_axis # ellipse defined by x^2 / a^2 + y^2 / b^2 = 1
        self.b = ellipse_minor_axis # see ^
        self.A_ellipse = self.approximate_spherical_cap_area(R = self.R,
                                                             a = self.a,
                                                             b = self.b)
        print("Area approximated: " + str(np.pi * self.a * self.b)) # Area approximated by pi*a*b
        print("Area integraded: " + str(self.A_ellipse)) # Integrated surface area
        self.a0 = desired_area_per_point
        # ! This last multiplier makes sure the points are pushed apart initially
        self.sigma = np.sqrt(self.a0 / ( np.sqrt(3) /2 )) / 1.12246204831 * 1.8 * scale
        # See LaTeX doc
        self.num_nodes_start = int(162.0 / scale / scale) # if set to non zero, override starting num nodes
        if self.num_nodes_start != 0:
            self.N = self.num_nodes_start
        else:
            self.N = int(self.A_ellipse / self.a0) + int(np.pi*(self.a+self.b)/self.desiredlgag/2.0)
        #self.N = 70 # override for test only
        
        # save dir
        self.output_dir = output_dir
        self.anim_dir = anim_dir

        # Iteration control
        self.max_iteration = max_iteration # if exceed max iteration, opt out of the main loop
        self.dt_start = dt # initial step size
        self.dt = dt # current step size

    def approximate_spherical_cap_area(self, R, a, b):
        """
        Approximate the surface area of the enclosed area on a sphere cut by an
        elliptical cylindar.
        """
        def integrand(x, c, d):
            cos_x = np.cos(x)
            sin_x = np.sin(x)
            denominator = c**2 * cos_x**2 + d**2 * sin_x**2
            result = 1 - 1 / np.sqrt(1 + 1 / denominator)
            return result
        
        # Define the parameters c and d
        c = np.sqrt(R**2 / a**2 - 1)  # example value for c
        d = np.sqrt(R**2 / b**2 - 1)  # example value for d

        # Perform the integration
        result, error = quad(integrand, 0, 2 * np.pi, args=(c, d))
        
        # Yell at the user if error is too big
        if error > 1e-3:
            print("WARNING : large error in integrating [approximate_spherical_cap_area] with c = "\
                + str(c) + " and d = " + str(d))
        
        return R * R * result

    def theta_max(self, phi):
        """
        Calculate the maximum theta angle for a given phi.

        Args:
            phi (float): Phi angle.

        Returns:
            float: Maximum theta angle.
        """
        return np.arcsin((1.0 / self.R) / np.sqrt(np.power(np.cos(phi) / self.a, 2.0) \
                                                  + np.power(np.sin(phi) / self.b, 2.0)))

    def ljp(self, r1, r2, sigma6, eps):
        """
        Lennard-Jones potential calculation.

        Args:
            r1 (ndarray): Position vector of particle 1.
            r2 (ndarray): Position vector of particle 2.
            sigma6 (float): Sigma raised to the power of 6.
            eps (float): Epsilon value.

        Returns:
            float: Lennard-Jones potential.
        """
        quadrance = np.sum((r1 - r2) ** 2)
        if quadrance <= 1e-3:
            quadrance = 1e-3
        return 4 * eps * (sigma6 * sigma6 / (quadrance ** 6) - sigma6 / (quadrance ** 3))

    def sum_ljp(self, bound_points, sigma6=None, eps=0.05):
        """
        Sum of Lennard-Jones potentials.

        Args:
            bound_points (ndarray): Array of points.
            sigma6 (float): Sigma raised to the power of 6.
            eps (float): Epsilon value.

        Returns:
            float: Sum of Lennard-Jones potentials.
        """
        if sigma6 is None:
            sigma6 = self.sigma ** 6
        tree = KDTree(bound_points)
        summed_ljp = 0.0
        for point in bound_points:
            _, idx = tree.query(point, k=2)
            if len(idx) > 1:
                if idx[1] < len(bound_points):
                    nearest_point = bound_points[idx[1]]
                    summed_ljp += self.ljp(point, nearest_point, sigma6, eps)
        return summed_ljp
    
    def avg_dist(self, pts):
        """
        Average distance between each point and their nearest neighbor.
        
        Args:
            pts (ndarry): Array of points
            
        Returns:
            float: average distance
        """
        tree = KDTree(pts)
        sum_dist = 0.0
        for point in pts:
            _, idx = tree.query(point, k = 2)
            if len(idx) > 1:
                nearest_point = pts[idx[1]]
                sum_dist += np.linalg.norm(point - nearest_point)
        return sum_dist / len(pts)

    def force(self, r1, r2, mode="norm", sigma6=None, eps=0.05):
        """
        Calculate force between particles.

        Args:
            r1 (ndarray): Position vector of particle 1.
            r2 (ndarray): Position vector of particle 2.
            mode (str): Mode of force calculation.
            sigma6 (float): Sigma raised to the power of 6.
            eps (float): Epsilon value.

        Returns:
            ndarray: Force vector.
        """
        if mode == "norm":
            sigma_norm = 0.5
            quadrance = np.sum(np.square(r2 - r1))
            if quadrance <= 1e-3:
                quadrance = 1e-3
            return sigma_norm * (r2 - r1) / np.power(quadrance, 3.0)
        elif mode == "lj":
            return self.ljforce(r1, r2, sigma6, eps)
        return 0

    def ljforce(self, r1, r2, sigma6, eps):
        """
        Calculate Lennard-Jones force between particles.

        Args:
            r1 (ndarray): Position vector of particle 1.
            r2 (ndarray): Position vector of particle 2.
            sigma6 (float): Sigma raised to the power of 6.
            eps (float): Epsilon value.

        Returns:
            ndarray: Lennard-Jones force vector.
        """
        quadrance = np.sum((r1 - r2) ** 2)
        if quadrance <= 1e-3:
            quadrance = 1e-3
        return - 4 * eps * ((-12 * sigma6 * sigma6 / (quadrance ** 7)) +\
                            (6 * sigma6 / (quadrance ** 4))) * (r2 - r1)

    def xyz(self, thetas, phis):
        """
        Convert spherical coordinates to Cartesian coordinates.

        Args:
            thetas (ndarray): Array of theta angles.
            phis (ndarray): Array of phi angles.

        Returns:
            ndarray: Array of Cartesian coordinates.
        """
        points = np.zeros((len(phis), 3))
        points[:, 0] = self.R * np.sin(thetas) * np.cos(phis)
        points[:, 1] = self.R * np.sin(thetas) * np.sin(phis)
        points[:, 2] = self.R * np.cos(thetas)
        return points
    
    def plot_points(self, thetas, phis, ax):
        """
        Plot points in 3D space.

        Args:
            thetas (ndarray): Array of theta angles.
            phis (ndarray): Array of phi angles.
        """
        points = self.xyz(thetas, phis)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='k')
        return 0

    def project_theta_phi(self, cartesian):
        """
        Project Cartesian coordinates to spherical coordinates.

        Args:
            cartesian (ndarray): Array of Cartesian coordinates.

        Returns:
            ndarray: Array of theta angles.
            ndarray: Array of phi angles.
        """
        thetas = np.arctan(np.sqrt(np.power(cartesian[:, 0], 2.0) +\
                                   np.power(cartesian[:, 1], 2.0)) / cartesian[:, 2])
        phis = np.arctan2(cartesian[:, 1], cartesian[:, 0])
        return thetas, phis

    def constrain(self, thetas, phis):
        """
        Constrain theta angles within bounds.

        Args:
            thetas (ndarray): Array of theta angles.
            phis (ndarray): Array of phi angles.

        Returns:
            ndarray: Array indicating if points are on boundary.
        """
        isonboundary = np.zeros(len(phis))
        for i in range(len(thetas)):
            thetas[i] = min(self.theta_max(phis[i]), np.abs(thetas[i]))
            if self.theta_max(phis[i]) < np.abs(thetas[i]):
                isonboundary[i] = 1
        return isonboundary

    def monte_carlo_random_move(self, cartesian, max_displacement):
        """
        Apply Monte Carlo random move to particles.

        Args:
            cartesian (ndarray): Array of Cartesian coordinates.
            max_displacement (float): Maximum displacement.

        Returns:
            ndarray: Updated array of Cartesian coordinates.
        """
        N, _ = cartesian.shape
        idx = np.random.randint(0, N - 1)
        axis = np.random.randint(0, 2)
        cartesian[idx, axis] += np.random.uniform(-max_displacement, max_displacement)
        return cartesian

    def check_convergence(self, ljp_record, check_range=10, converge_threshold=1e-7):
        """
        Check convergence of the simulation.

        Args:
            ljp_record (list): List of Lennard-Jones potentials.
            check_range (int): Range to check for convergence.
            converge_threshold (float): Convergence threshold.

        Returns:
            bool: True if converged, False otherwise.
        """
        converge = True
        for i in range(2, check_range):
            if ljp_record[-1] > ljp_record[-i]:
                converge = False
            if ljp_record[-i] - ljp_record[-1] > converge_threshold:
                converge = False
        return converge

    def run_simulation(self):
        """
        Run the particle placement simulation.
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        cam = Camera(fig)

        # Generating initial points; initialize variable lists
        phivals = uniform.rvs(scale=2. * np.pi, size=self.N)
        thetavals = [uniform.rvs(scale=self.theta_max(phi)) for phi in phivals]
        thetavals = np.array(thetavals)
        isonboundary = np.zeros(len(phivals)) # 0 = not on boundary, 1 = on boundary

        # Plotting boundary
        bound_phis = np.linspace(0, 2. * np.pi, num=200)
        bound_thetas = self.theta_max(bound_phis)
        bound_points = self.xyz(bound_thetas, bound_phis)
        ax.plot(bound_points[:, 0], bound_points[:, 1], bound_points[:, 2], color='r')

        # plot of points
        self.plot_points(thetavals, phivals, ax)
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        cam.snap() # For drawing animation

        ljp_record = [] # Record of system energy over course of simulation
        tot_energy = 0.0 # Current system energy

        # Main loop
        for k in range(self.max_iteration):
            mode = "norm"
            f = np.zeros((self.N, 3))
            cartesian = self.xyz(thetavals, phivals)

            # KDTree
            tree = KDTree(cartesian)
            sum_dist = 0.0
            for (idx, point) in enumerate(cartesian):
                _, indices = tree.query(point, k = 7) # choose 6 closet neighbors, immitating trimesh
                if len(indices) > 1: # if there is at least two (first is itself) closeset neighbor
                    for j in range(1, len(indices)): # iterating starting from NOT itself to end
                        jdx = indices[j] # get the index of its neighbor
                        if jdx < len(cartesian):
                            if isonboundary[idx] == isonboundary[jdx]:
                                f[jdx] += 0.5 * self.force(cartesian[idx], cartesian[jdx], eps=0.05, mode=mode)
                                f[idx] -= 0.5 * self.force(cartesian[idx], cartesian[jdx], eps=0.05, mode=mode)
                            elif isonboundary[idx] == 1:
                                # Note: when a point reaches boundary, it bounce back
                                f[jdx] += 0.8 * self.force(cartesian[idx], cartesian[jdx], eps=0.05, mode=mode)
                                f[idx] -= 0.2 * self.force(cartesian[idx], cartesian[jdx], eps=0.05, mode=mode)
                            elif isonboundary[jdx] == 1:
                                f[jdx] += 0.2 * self.force(cartesian[idx], cartesian[jdx], eps=0.05, mode=mode)
                                f[idx] -= 0.8 * self.force(cartesian[idx], cartesian[jdx], eps=0.05, mode=mode)
            
            cartesian += f * self.dt
            thetavals, phivals = self.project_theta_phi(cartesian)
            isonboundary = self.constrain(thetavals, phivals)

            # Monte Carlo random move
            tot_energy = self.sum_ljp(cartesian)
            ljp_record.append(tot_energy)
            max_displacement = 4.0 # max displacement of Monte Carlo random move
            # Generate random move
            cartesian_new = self.monte_carlo_random_move(cartesian, max_displacement)
            # Calculate energy after random move
            tot_energy_new = self.sum_ljp(cartesian_new)
            # Apply move if new energy is lower
            if tot_energy_new < tot_energy:
                cartesian_new = cartesian
            # Still has a probability to apply move if new energy is higher
            elif np.random.uniform() < 0.1 * np.exp(tot_energy / tot_energy_new):
                cartesian_new = cartesian

            # Draw animation
            if k % 400 == 0: # Step interval of frames
                ax.plot(bound_points[:, 0], bound_points[:, 1], bound_points[:, 2], color='r')
                self.plot_points(thetavals, phivals, ax)
                cam.snap()

            # Adpative step size
            if k > 100 and k % 50 == 0:
                mode = "lj"
                if ljp_record[-1] < ljp_record[-2]: # increase the step size if energy is lower
                    self.dt *= 1.2
                elif ljp_record[-1] > ljp_record[-2]: # decrease the step size if energy is higher
                    self.dt *= 0.5
                
                # Check for convergence
                # check_convergence(self, ljp_record, check_range=10, converge_threshold=1e-6)
                if self.check_convergence(ljp_record, converge_threshold = 0.01) and k > 1000:
                    avg_dist = self.avg_dist(cartesian)
                    if avg_dist < self.minlgag and self.nodeChanges < self.maxNnodeChanges and\
                        self.changeNodeCountdown <= 0:
                        self.N = int(self.N * (1.0 - self.nodeChangeRatio))
                        print("Remove node: Current node num = " + str(self.N))
                        phivals = uniform.rvs(scale=2. * np.pi, size=self.N)
                        thetavals = [uniform.rvs(scale=self.theta_max(phi)) for phi in phivals]
                        thetavals = np.array(thetavals)
                        isonboundary = np.zeros(len(phivals)) # 0 = not on boundary, 1 = on boundary
                        self.dt = self.dt_start
                        self.nodeChanges += 1
                        self.nodeChangeRatio *= self.nodeChangeRatioStep
                        self.changeNodeCountdown = self.changeNodeInterval
                    elif avg_dist > self.maxlgag and self.nodeChanges < self.maxNnodeChanges and\
                        self.changeNodeCountdown <= 0:
                        self.N = int(self.N * (1.0 + self.nodeChangeRatio)) + 1
                        print("Add node: Current node num = " + str(self.N))
                        phivals = uniform.rvs(scale=2. * np.pi, size=self.N)
                        thetavals = [uniform.rvs(scale=self.theta_max(phi)) for phi in phivals]
                        thetavals = np.array(thetavals)
                        isonboundary = np.zeros(len(phivals)) # 0 = not on boundary, 1 = on boundary
                        self.dt = self.dt_start
                        self.nodeChanges += 1
                        self.nodeChangeRatio *= self.nodeChangeRatioStep
                        self.changeNodeCountdown = self.changeNodeInterval
                    elif self.check_convergence(ljp_record, converge_threshold = 1e-3) and\
                        self.changeNodeCountdown <= 0:
                        break
            self.changeNodeCountdown -= 1

            # Print Process
            if k % 100 == 0:
                print("Itr: " + str(k))
        # Plotting final points
        ax.plot(bound_points[:, 0], bound_points[:, 1], bound_points[:, 2], color='r')
        self.plot_points(thetavals, phivals, ax)
        cam.snap()
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        ax.set_zlim(-30, 30)

        # Creating and showing the animation
        anim = cam.animate()
        plt.show()

        # Check existence of path. If not, create one.
        if not os.path.exists(self.anim_dir):
            os.makedirs(self.anim_dir)
        anim.save(str(self.anim_dir) + "monte_carlo_mesh_a" + str(int(self.a)) + "_b" + str(int(self.b)) +\
            "d" + str(int(scale * 10)) + ".gif") # save animation to gif
        print("Converged LJP: ")
        print(ljp_record[-1])
        
        # Calculate average nearest distance
        cartesian = self.xyz(thetavals, phivals)
        print("Average distance: ")
        print(self.avg_dist(cartesian))
        
        # Write to csv
        # Check existence of path. If not, create one.
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        df = pd.DataFrame(cartesian)
        df.to_csv(str(self.output_dir) + "mc_mesh_a" + str(int(self.a)) + "_b" + str(int(self.b)) +\
            "d" + str(int(scale * 10)) + ".csv", header=False, index=False)

    

# Example usage
# 20.0 -
# 20.0, 15.0, 12.0, 10.0, 8.0
# 1.0, 0.75, 0.6, 0.5, 0.4 (major / minor axis ratio)
if __name__ == "__main__":
    # The scales are iterated for major = minor axis = 30.0
    # Scale range = 1.0, 1.2, 1.5, 2.0, 2.5
    scale = 0.7
    print(19.1305011695*scale*scale)
    sim = ParticlePlacementSimulation(sphere_radius=50.0, 
                                      ellipse_major_axis=20.0, 
                                      ellipse_minor_axis=20.0, 
                                      desired_area_per_point=19.1305011695*scale*scale, #desired_area_per_point=19.1305011695,
                                      scale = scale,
                                      max_iteration = 10,
                                      dt = 1.0)
    # for profiling
    
    #cProfile.run('sim.run_simulation()')
    sim.run_simulation()

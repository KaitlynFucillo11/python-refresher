import numpy as np
import math

g = 9.81  # gravity [m/s^2]
rho_water = 1000  # water density [kg/m^3]


# Problem 1
def calculate_buoyancy(V, density_fluid):
    """
    Calculate the buoyant force on a submerged object.

    Parameters:
        V (float): Volume of the object (m^3)
        density_fluid (float): Density of the fluid (kg/m^3)

    Returns:
        float: Buoyant force (N)
    """
    return density_fluid * V * g


# Problem 2
def will_it_float(V, mass):
    """
    Determine if an object will float in water.

    Parameters:
        V (float): Volume of the object (m^3)
        mass (float): Mass of the object (kg)

    Returns:
        bool: True if it floats, False otherwise
    """
    buoyant_force = calculate_buoyancy(V, rho_water)
    weight = mass * g
    return buoyant_force >= weight


# Problem 3
def calculate_pressure(depth):
    """
    Calculate the pressure at a given depth in water.

    Parameters:
        depth (float): Depth (m)

    Returns:
        float: Pressure (Pa)
    """
    return rho_water * g * depth


# Problem 4
def calculate_acceleration(F, m):
    """
    Calculate linear acceleration.

    Parameters:
        F (float): Force (N)
        m (float): Mass (kg)

    Returns:
        float: Acceleration (m/s^2)
    """
    return F / m


# Problem 5
def calculate_angular_acceleration(tau, I):
    """
    Calculate angular acceleration.

    Parameters:
        tau (float): Torque (Nm)
        I (float): Moment of inertia (kg·m^2)

    Returns:
        float: Angular acceleration (rad/s^2)
    """
    return tau / I


# Problem 6
def calculate_torque(F_magnitude, F_direction, r):
    """
    Calculate torque from force and distance.

    Parameters:
        F_magnitude (float): Force (N)
        F_direction (float): Direction in degrees
        r (float): Distance to axis (m)

    Returns:
        float: Torque (Nm)
    """
    angle_rad = math.radians(F_direction)
    return F_magnitude * r * math.sin(angle_rad)


# Problem 7
def calculate_moment_of_inertia(m, r):
    """
    Calculate moment of inertia.

    Parameters:
        m (float): Mass (kg)
        r (float): Distance from axis (m)

    Returns:
        float: Moment of inertia (kg·m^2)
    """
    return m * r**2


# Problem 8a
def calculate_auv_acceleration(
    F_magnitude, F_angle, mass=100, volume=0.1, thruster_distance=0.5
):
    """
    Calculate linear acceleration of the AUV.

    Parameters:
        F_magnitude (float): Thruster force (N)
        F_angle (float): Force direction (rad)
        mass (float): AUV mass (kg)
        volume (float): AUV volume (m^3)
        thruster_distance (float): Distance from center of mass (m)

    Returns:
        np.ndarray: Acceleration vector [ax, ay] (m/s^2)
    """
    ax = (F_magnitude * np.cos(F_angle)) / mass
    ay = (F_magnitude * np.sin(F_angle)) / mass
    return np.array([ax, ay])


# Problem 8b
def calculate_auv_angular_acceleration(
    F_magnitude, F_angle, inertia=1.0, thruster_distance=0.5
):
    """
    Calculate angular acceleration of the AUV.

    Parameters:
        F_magnitude (float): Force (N)
        F_angle (float): Force angle (rad)
        inertia (float): Moment of inertia (kg·m^2)
        thruster_distance (float): Distance to thruster (m)

    Returns:
        float: Angular acceleration (rad/s^2)
    """
    tau = thruster_distance * F_magnitude * np.sin(F_angle)
    return tau / inertia


# Problem 9a
def calculate_auv2_acceleration(T, alpha, theta, mass=100):
    """
    Calculate total linear acceleration from 4 thrusters.

    Parameters:
        T (np.ndarray): Thruster forces [T1, T2, T3, T4]
        alpha (float): Angle of thrusters (rad)
        theta (float): AUV orientation (rad)
        mass (float): Mass (kg)

    Returns:
        np.ndarray: Acceleration [ax, ay] (m/s^2)
    """
    ax = np.sum(T) * np.cos(alpha + theta) / mass
    ay = np.sum(T) * np.sin(alpha + theta) / mass
    return np.array([ax, ay])


# Problem 9b
def calculate_auv2_angular_acceleration(T, alpha, L, l, inertia=100):
    """
    Calculate angular acceleration from 4 thrusters.

    Parameters:
        T (np.ndarray): Thruster forces [T1, T2, T3, T4]
        alpha (float): Thruster angle (rad)
        L (float): Longitudinal distance (m)
        l (float): Lateral distance (m)
        inertia (float): Moment of inertia (kg·m^2)

    Returns:
        float: Angular acceleration (rad/s^2)
    """
    torque = (-T[0] + T[1] + T[2] - T[3]) * l * np.cos(alpha) + (
        T[0] + T[1] - T[2] - T[3]
    ) * L * np.sin(alpha)
    return torque / inertia


# Problem 10
def simulate_auv2_motion(
    T, alpha, L, l, mass=100, inertia=100, dt=0.1, t_final=10, x0=0, y0=0, theta0=0
):
    """
    Simulate AUV2 motion in 2D.

    Returns:
        t, x, y, theta, v, omega, a
    """
    steps = int(t_final / dt)
    t = np.linspace(0, t_final, steps)
    x = np.zeros(steps)
    y = np.zeros(steps)
    theta = np.zeros(steps)
    v = np.zeros(steps)
    omega = np.zeros(steps)
    a_list = np.zeros(steps)

    x[0], y[0], theta[0] = x0, y0, theta0

    for i in range(1, steps):
        acc = calculate_auv2_acceleration(T, alpha, theta[i - 1], mass)
        angular_acc = calculate_auv2_angular_acceleration(T, alpha, L, l, inertia)

        v[i] = v[i - 1] + dt * np.linalg.norm(acc)
        omega[i] = omega[i - 1] + dt * angular_acc
        theta[i] = theta[i - 1] + dt * omega[i]

        x[i] = x[i - 1] + dt * v[i] * np.cos(theta[i])
        y[i] = y[i - 1] + dt * v[i] * np.sin(theta[i])
        a_list[i] = np.linalg.norm(acc)

    return t, x, y, theta, v, omega, a_list


# Plotting
def plot_auv2_motion(t, x, y, theta, v, omega, a):
    """
    Plot the motion of AUV2 in the 2D plane.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="AUV Path")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("AUV2 Trajectory")
    plt.grid()
    plt.axis("equal")
    plt.legend()
    plt.show()

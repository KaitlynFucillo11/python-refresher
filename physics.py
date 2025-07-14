import numpy as np
import math
import matplotlib.pyplot as plt

G = 9.81  # gravity [m/s^2]
rho_water = 1000  # water density [kg/m^3]


# problem 1
def calculate_buoyancy(V, density_fluid):
    return density_fluid * V * G


# Problem 2
def will_it_float(V, mass):
    bouyant_force = calculate_buoyancy(V, rho_water)
    weight = mass * G
    if weight > bouyant_force:
        return "False, the item will sink"
    if weight < bouyant_force:
        return "True, the item will float"
    else:
        return "The item is neutrally buoyant!"


# Problem 3
def calculate_pressure(depth):
    depth = rho_water * G * depth
    return depth


# Problem 4
def calculate_acceleration(F, m):
    acceleration = F / m
    return acceleration


# Probelm 5
def calculate_angular_velocity(tau, I):
    angular_velocity = tau / I
    return angular_velocity


# Problem 6
def calculate_torque(F_magnitude, F_direction, r):
    F_x = F_magnitude * math.cos(math.radians(F_direction))
    F_y = F_magnitude * math.sin(math.radians(F_direction))
    torque = r * (F_x + F_y)
    return torque


# Problem 7
def calculate_moment_of_intertia(m, r):
    moment_of_inertia = m * r**2
    return moment_of_inertia


# Problem 8:


def calculate_auv_acceleration(F_magnitude, F_angle, mass=100, volume=0.1, r=0.5):
    ax = (F_magnitude * np.cos(F_angle)) / mass
    ay = (F_magnitude * np.sin(F_angle)) / mass
    return np.array([ax, ay])


def calculate_auv_angular_acceleration(F_magnitude, F_angle, I=1, r=0.5):
    return (r * F_magnitude * np.sin(F_angle)) / I


# Problem 9: AUV with 4 thrusters


def calculate_auv2_acceleration(T, alpha, theta, mass=100):
    T = np.asarray(T)

    ax_local = (
        np.cos(alpha) * T[0]
        + np.cos(-alpha) * T[1]
        + np.cos(np.pi + alpha) * T[2]
        + np.cos(np.pi - alpha) * T[3]
    ) / mass

    ay_local = (
        np.sin(alpha) * T[0]
        + np.sin(-alpha) * T[1]
        + np.sin(np.pi + alpha) * T[2]
        + np.sin(np.pi - alpha) * T[3]
    ) / mass

    acc_local = np.array([ax_local, ay_local])

    transformation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )

    acc_global = transformation_matrix @ acc_local

    return acc_global


def calculate_auv2_angular_acceleration(T, alpha, L, l, inertia=100):
    Fx = T * np.cos(alpha)
    Fy = T * np.sin(alpha)
    torque = np.sum(l * Fy - L * Fx)

    # Angular acceleration = torque / inertia
    angular_acceleration = torque / inertia

    return angular_acceleration


# problem 10
def simulate_auv2_motion(
    T, alpha, L, l, mass=100, inertia=100, dt=0.1, t_final=10, x0=0, y0=0, theta0=0
):

    steps = int(t_final / dt)

    t = np.linspace(0, t_final, steps)
    x = np.zeros(steps)
    y = np.zeros(steps)
    theta = np.zeros(steps)
    v = np.zeros((steps, 2))  # velocity vectors
    omega = np.zeros(steps)  # angular velocity
    a = np.zeros((steps, 2))  # acceleration vectors

    x[0], y[0], theta[0] = x0, y0, theta0

    for i in range(1, steps):
        # Compute linear acceleration in global frame
        acc = calculate_auv2_acceleration(T, alpha, theta[i - 1], mass)
        a[i] = acc

        # Integrate velocity
        v[i] = v[i - 1] + a[i] * dt

        # Integrate position
        x[i] = x[i - 1] + v[i][0] * dt
        y[i] = y[i - 1] + v[i][1] * dt

        # Compute angular acceleration
        ang_acc = calculate_auv2_angular_acceleration(T, alpha, L, l, inertia)

        # Integrate angular velocity and orientation
        omega[i] = omega[i - 1] + ang_acc * dt
        theta[i] = theta[i - 1] + omega[i] * dt

    # Return outputs
    v_magnitude = np.linalg.norm(v, axis=1)
    a_magnitude = np.linalg.norm(a, axis=1)
    return t, x, y, theta, v_magnitude, omega, a_magnitude

    import matplotlib.pyplot as plt


def plot_auv2_motion(t, x, y, theta, v, omega, a):
    plt.figure(figsize=(15, 10))

    # Subplot 1: Trajectory
    plt.subplot(2, 2, 1)
    plt.plot(x, y, label="Trajectory", color="dodgerblue")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("AUV Trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    # Subplot 2: Orientation over time
    plt.subplot(2, 2, 2)
    plt.plot(t, theta, label="Orientation (theta)", color="mediumseagreen")
    plt.xlabel("Time (s)")
    plt.ylabel("Orientation (rad)")
    plt.title("Orientation Over Time")
    plt.grid(True)
    plt.legend()

    # Subplot 3: Linear velocity and acceleration
    plt.subplot(2, 2, 3)
    plt.plot(t, v, label="Velocity (m/s)", color="orange")
    plt.plot(t, a, label="Acceleration (m/s²)", color="crimson")
    plt.xlabel("Time (s)")
    plt.ylabel("Magnitude")
    plt.title("Velocity and Acceleration")
    plt.grid(True)
    plt.legend()

    # Subplot 4: Angular velocity
    plt.subplot(2, 2, 4)
    plt.plot(t, omega, label="Angular Velocity (rad/s)", color="slateblue")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Velocity")
    plt.title("Angular Velocity Over Time")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

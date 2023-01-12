import numpy as np
from scipy.integrate import solve_ivp

class TwoJointArm(object):
    def __init__(self):
        self.state = np.asmatrix(np.zeros(4)).T

        # Length of segments
        self.l1 = 46.25 * .0254
        self.l2 = 41.80 * .0254
        
        # Mass of segments
        self.m1 = 9.34 * .4536
        self.m2 = 9.77 * .4536

        # Distance from pivot to CG for each segment
        self.r1 = 21.64 * .0254
        self.r2 = 26.70 * .0254

        # Moment of inertia about CG for each segment
        self.I1 = 2957.05 * .0254*.0254 * .4536
        self.I2 = 2824.70 * .0254*.0254 * .4536

        # Gearing of each segment
        self.G1 = 140.
        self.G2 = 90.

        # Number of motors in each gearbox
        self.N1 = 1
        self.N2 = 2

        # Gravity
        self.g = 9.81

        self.stall_torque = 3.36
        self.free_speed = 5880.0 * 2.0*np.pi/60.0
        self.stall_current = 166

        self.R = 12.0/self.stall_current

        self.Kv = self.free_speed / 12.0
        self.Kt = self.stall_torque / self.stall_current

        # K3*Voltage - K4*velocity = motor torque
        self.K3 = np.matrix([[self.N1*self.G1, 0], [0, self.N2*self.G2]])*self.Kt/self.R
        self.K4 = np.matrix([[self.G1*self.G1*self.N1, 0], [0, self.G2*self.G2*self.N2]])*self.Kt/self.Kv/self.R

        self.voltage_log = np.matrix([0,0])
        self.current_log = np.matrix([0,0])

        # Control law f(t, state) = voltage to each joint, as a column matrix. Defaults to 0 effort.
        # Must be able to pass in multiple (t, state) as [t1, t2, t3...] and [state1, state2, state3...] arrays
        #   if you want to use the voltage and current log functionality.
        self.control_law = lambda t, state: np.asmatrix(np.zeros((2, np.size(state,1))))
    
    # Returns angular position of joints
    def get_ang_pos(self):
        return self.state[:2]

    # Returns angular velocity of joints
    def get_ang_vel(self):
        return self.state[2:]

    # Returns linear position of end effector
    def get_lin_pos(self):
        (_, end_eff) = self.fwd_kinematics(self.get_ang_pos())
        return end_eff
    
    # Return linear position of each joint
    def get_lin_joint_pos(self, ang_pos = None):
        if ang_pos is None:
            ang_pos = self.get_ang_pos()
        return self.fwd_kinematics(ang_pos)

    # Returns linear velocity of end effector
    def get_lin_vel(self):
        (J, _, _) = self.jacobian(self.state)
        return J * self.get_ang_vel()

    # Forward kinematics for a target position pos (theta1, theta2)
    def fwd_kinematics(self, pos):
        [theta1, theta2] = pos.flat
        joint2 = np.matrix([self.l1*np.cos(theta1), self.l1*np.sin(theta1)]).T
        end_eff = joint2 + np.matrix([self.l2*np.cos(theta1 + theta2), self.l2*np.sin(theta1 + theta2)]).T
        return (joint2, end_eff)
    
    # Inverse kinematics for a target position pos (x,y). Invert controls elbow direction.
    def inv_kinematics(self, pos, invert = False):
        [x,y] = pos.flat
        theta2 = np.arccos((x*x + y*y - (self.l1*self.l1 + self.l2*self.l2)) / \
            (2*self.l1*self.l2))

        if invert:
            theta2 = -theta2
        
        theta1 = np.arctan2(y, x) - np.arctan2(self.l2*np.sin(theta2), self.l1 + self.l2*np.cos(theta2))
        return np.matrix([theta1, theta2]).T

    # Return some jacobian matrices of the arm.
    # J - Jacobian of end effector
    # Jcm1 - Jacobian of center of mass of link 1
    # Jcm2 - Jacobian of center of mass of link 2
    def jacobian(self, pos):
        [theta1, theta2] = pos.flat
        s1 = np.sin(theta1)
        c1 = np.cos(theta1)
        s12 = np.sin(theta1 + theta2)
        c12 = np.cos(theta1 + theta2)

        l1 = self.l1
        l2 = self.l2
        r1 = self.r1
        r2 = self.r2

        J = np.matrix([[-l1*s1-l2*s12, -l2*s12], [l1*c1+l2*c12, l2*c12], [0,0], [0,0], [0,0], [1,1]])
        Jcm1 = np.matrix([[-r1*s1, 0], [r1*c1, 0], [0,0], [0,0], [0,0], [1,0]])
        Jcm2 = np.matrix([[-l1*s1-r2*s12, -r2*s12], [l1*c1+r2*c12, r2*c12], [0,0], [0,0], [0,0], [1,1]])

        return (J, Jcm1, Jcm2)

    # Returns matrices D, C and vector G in D*alpha + C*omega + G = torque
    def dynamics_matrices(self, state):
        [theta1, theta2, omega1, omega2] = state.flat
        r1 = self.r1
        l1 = self.l1
        r2 = self.r2
        c2 = np.cos(theta2)

        hD = l1*r2*c2
        D = self.m1*np.matrix([[r1*r1, 0], [0, 0]]) + self.m2*np.matrix([[l1*l1 + r2*r2 + 2*hD, r2*r2 + hD], [r2*r2 + hD, r2*r2]]) + \
            self.I1*np.matrix([[1, 0], [0, 0]]) + self.I2*np.matrix([[1, 1], [1, 1]])

        hC = -self.m2*l1*r2*np.sin(theta2)
        C = np.matrix([[hC*omega2, hC*omega1 + hC*omega2], [-hC*omega1, 0]])

        G = self.g*np.cos(theta1) * np.matrix([self.m1*r1 + self.m2*l1, 0]).T + \
            self.g*np.cos(theta1+theta2) * np.matrix([self.m2*r2, self.m2*r2]).T

        return (D, C, G)
    
    # Perform Runge-Kutta 4 integration
    def RK4(self, f, x, dt):
        a = f(x)
        b = f(x + dt/2.0 * a)
        c = f(x + dt/2.0 * b)
        d = f(x + dt*c)
        return x + dt * (a + 2.0 * b + 2.0 * c + d) / 6.0
        #return x + dt*a

    # Uses the given state and applied torques to find the derivative of the state
    def dynamics_torque(self, state, torques):
        (D, C, G) = self.dynamics_matrices(state)
        omega_vec = state[2:]
        alpha_vec = D.I*(torques - C*omega_vec - G)
        return np.concatenate((omega_vec, alpha_vec))

    # Time step the arm with a set of given torques.
    def step_torque(self, torques, dt):
        self.state = self.RK4(lambda s: self.dynamics_torque(s, torques), self.state, dt)
        return self.state
    
    def feed_forward_torques(self, state, accels = np.matrix([0,0]).T):
        (D, C, G) = self.dynamics_matrices(state)
        torques = D*accels + C*state[2:] + G
        return torques
    
    # D*alpha + C*omega + G = torque = K3*u - K4*omega
    def dynamics(self, state, u):
        (D, C, G) = self.dynamics_matrices(state)
        omega_vec = state[2:]
        basic_torque = self.K3*u
        back_emf_loss = self.K4*omega_vec
        torque = basic_torque - back_emf_loss
        alpha_vec = D.I*(torque - C*omega_vec - G)
        return np.concatenate((omega_vec, alpha_vec))

    # Step the arm forward with a given input. Updates the internal state.
    # Much slower than using simulate().
    def step(self, u, dt):
        u = np.clip(u, -12, 12)
        self.state = self.RK4(lambda s: self.dynamics(s, u), self.state, dt)
        self.voltage_log = np.concatenate((self.voltage_log, u.T))
        current = self.stall_current * (u - np.multiply(self.state[2:], np.matrix([self.G1, self.G2]).T)/self.Kv) / 12.0
        self.current_log = np.concatenate((self.current_log, current.T))
        return self.state
    
    # Determine the feed forward to satisfy given state and accelerations.
    def feed_forward(self, state, alpha = np.matrix([0,0]).T):
        (D, C, G) = self.dynamics_matrices(state)
        omegas = state[2:]
        return self.K3.I * (D*alpha + C*omegas + G + self.K4*omegas)
    
    # Use the pre-assigned control law, and given state and time, to determine
    # the rate of change of the state 
    def get_dstate(self, t, state):
        state = np.asmatrix(state).T
        u = self.control_law(t, state)
        dstate = self.dynamics(state, u).A1
        return dstate
    
    # Perform the simulation using the arm's assigned control law.
    # Does not update or use the arm's internal state.
    def simulate(self, t_span, initial_state = None, t_eval = None):
        if initial_state is None:
            initial_state = self.state.A1
        return solve_ivp(self.get_dstate, t_span, initial_state, t_eval = t_eval)
    
    # Reconstruct the voltage history using the simulation results.
    def get_voltage_log(self, sim_solution):
        return self.control_law(sim_solution.t, sim_solution.y)

    # Get the current given a voltage and state.
    # If multiple voltages and states are given as matrices where each column is an entry,
    # will return all the currents corresponding to the pairs.
    def get_current(self, voltage, state):
        omegas = state[2:]
        omegas[0,:] = omegas[0,:] * self.G1
        omegas[1,:] = omegas[1,:] * self.G2
        stall_voltage = voltage - omegas/self.Kv
        current = stall_voltage/self.R
        return current

    # Reconstruct the current draw history using the simulation results.
    def get_current_log(self, sim_solution):
        voltage = self.get_voltage_log(sim_solution)
        return self.get_current(voltage, sim_solution.y)
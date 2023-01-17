import numpy as np
from scipy.integrate import solve_ivp
import controls_util as controls

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

        self.Rm = 12.0/self.stall_current

        self.Kv = self.free_speed / 12.0
        self.Kt = self.stall_torque / self.stall_current

        # K3*Voltage - K4*velocity = motor torque
        self.K3 = np.matrix([[self.N1*self.G1, 0], [0, self.N2*self.G2]])*self.Kt/self.Rm
        self.K4 = np.matrix([[self.G1*self.G1*self.N1, 0], [0, self.G2*self.G2*self.N2]])*self.Kt/self.Kv/self.Rm

        self.voltage_log = np.matrix([0,0])
        self.current_log = np.matrix([0,0])

        # Control law f(t, state) = voltage to each joint, as a column matrix. Defaults to 0 effort.
        # Must be able to pass in multiple (t, state) as [t1, t2, t3...] and [state1, state2, state3...] arrays
        #   if you want to use the voltage and current log functionality.
        self.control_law = lambda t, state: np.asmatrix(np.zeros((2, np.size(state,1))))

        pos_tol = .1
        vel_tol = .3

        # Bryson's rule - set Q[i,i] = 1/err_x_max[i]^2, where err_x_max[i] is the max acceptable error for x[i]
        #                 set R[i,i] = 1/12^2 for 12V
        self.Q = np.matrix(np.diag([1/pos_tol/pos_tol, 1/pos_tol/pos_tol, 1/vel_tol/vel_tol, 1/vel_tol/vel_tol]))
        self.R = np.matrix(np.diag([1/12.0/12.0, 1/12.0/12.0]))

        self.last_controller_time = -10
        self.loop_time = 1/50.0
        self.last_u = np.matrix([0,0]).T
    
    def get_ang_pos(self) -> np.matrix:
        """Returns angular position of joints"""
        return self.state[:2]

    def get_ang_vel(self) -> np.matrix:
        """Returns angular velocity of joints"""
        return self.state[2:4]

    def get_lin_pos(self) -> np.matrix:
        """Returns linear position of end effector"""
        (_, end_eff) = self.fwd_kinematics(self.get_ang_pos())
        return end_eff
    
    def get_lin_joint_pos(self, ang_pos: np.matrix = None) -> np.matrix:
        """Return linear position of each joint
        
        Arguments:
            ang_pos: 2x1 matrix containing joint angles, in radians. Defaults to internal values.
        """
        if ang_pos is None:
            ang_pos = self.get_ang_pos()
        return self.fwd_kinematics(ang_pos)

    def get_lin_vel(self) -> np.matrix:
        """Returns linear velocity of end effector."""
        (J, _, _) = self.jacobian(self.state)
        return J * self.get_ang_vel()

    def fwd_kinematics(self, pos: np.matrix) -> tuple[np.matrix, np.matrix]:
        """Forward kinematics for a target position pos (theta1, theta2)"""
        [theta1, theta2] = pos.flat
        joint2 = np.matrix([self.l1*np.cos(theta1), self.l1*np.sin(theta1)]).T
        end_eff = joint2 + np.matrix([self.l2*np.cos(theta1 + theta2), self.l2*np.sin(theta1 + theta2)]).T
        return (joint2, end_eff)
    
    def inv_kinematics(self, pos: np.matrix, invert: bool = False) -> np.matrix:
        """Inverse kinematics for a target position pos (x,y). Invert controls elbow direction."""
        [x,y] = pos.flat
        theta2 = np.arccos((x*x + y*y - (self.l1*self.l1 + self.l2*self.l2)) / \
            (2*self.l1*self.l2))

        if invert:
            theta2 = -theta2
        
        theta1 = np.arctan2(y, x) - np.arctan2(self.l2*np.sin(theta2), self.l1 + self.l2*np.cos(theta2))
        return np.matrix([theta1, theta2]).T

    def jacobian(self, state: np.matrix) -> tuple[np.matrix, np.matrix, np.matrix]:
        """Return some jacobian matrices of the arm.
        J - Jacobian of end effector
        Jcm1 - Jacobian of center of mass of link 1
        Jcm2 - Jacobian of center of mass of link 2
        """

        pos = state[:2]
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

    def dynamics_matrices(self, state):
        """Returns matrices M, C and vector G in M*alpha + C*omega + G = torque"""
        [theta1, theta2, omega1, omega2] = state[:4].flat
        r1 = self.r1
        l1 = self.l1
        r2 = self.r2
        c2 = np.cos(theta2)

        hM = l1*r2*c2
        M = self.m1*np.matrix([[r1*r1, 0], [0, 0]]) + self.m2*np.matrix([[l1*l1 + r2*r2 + 2*hM, r2*r2 + hM], [r2*r2 + hM, r2*r2]]) + \
            self.I1*np.matrix([[1, 0], [0, 0]]) + self.I2*np.matrix([[1, 1], [1, 1]])

        hC = -self.m2*l1*r2*np.sin(theta2)
        C = np.matrix([[hC*omega2, hC*omega1 + hC*omega2], [-hC*omega1, 0]])

        G = self.g*np.cos(theta1) * np.matrix([self.m1*r1 + self.m2*l1, 0]).T + \
            self.g*np.cos(theta1+theta2) * np.matrix([self.m2*r2, self.m2*r2]).T

        return (M, C, G)
    
    def RK4(self, f, x, dt):
        """Perform Runge-Kutta 4 integration"""
        a = f(x)
        b = f(x + dt/2.0 * a)
        c = f(x + dt/2.0 * b)
        d = f(x + dt*c)
        return x + dt * (a + 2.0 * b + 2.0 * c + d) / 6.0

    def dynamics_torque(self, state, torques):
        """Uses the given state and applied torques to find the derivative of the state."""
        (M, C, G) = self.dynamics_matrices(state)
        omega_vec = state[2:4]
        alpha_vec = M.I*(torques - C*omega_vec - G)
        return np.concatenate((omega_vec, alpha_vec))

    def step_torque(self, torques, dt):
        """Time step the arm with a set of given torques."""
        self.state = self.RK4(lambda s: self.dynamics_torque(s, torques), self.state, dt)
        return self.state
    
    def feed_forward_torques(self, state, accels = np.matrix([0,0]).T):
        (M, C, G) = self.dynamics_matrices(state)
        torques = M*accels + C*state[2:4] + G
        return torques
    
    def dynamics(self, state, u):
        """Finds the derivative of the state given an initial state and voltage input.
        M*alpha + C*omega + G = torque = K3*u - K4*omega
        """
        omega_vec = state[2:4]
        (M, C, G) = self.dynamics_matrices(state)
        basic_torque = self.K3*u
        back_emf_loss = self.K4*omega_vec
        disturbance_torque = np.matrix([0,0]).T

        # Try different disturbances:
        #basic_torque = basic_torque * .5
        #disturbance_torque = np.matrix([30, -20]).T
        #M = M * 1.5
        #G = G * 2

        torque = basic_torque - back_emf_loss + disturbance_torque
        alpha_vec = M.I*(torque - C*omega_vec - G)
        return np.matrix(np.concatenate((omega_vec, alpha_vec)))

    def unbounded_step(self, state, u, dt = .005):
        """Step the arm forward with a given input. Updates the internal state.
        Much slower than using simulate().
        """
        return self.RK4(lambda s: self.dynamics(s, u), state, dt)

    def step(self, u, dt):
        """Step the arm forward with a given input. Bounds the input if necessary.
        Updates the internal state. Much slower than using simulate().
        """
        u = np.clip(u, -12, 12)
        self.state = self.unbounded_step(self.state, u, dt)
        self.voltage_log = np.concatenate((self.voltage_log, u.T))
        current = self.stall_current * (u - np.multiply(self.state[2:4], np.matrix([self.G1, self.G2]).T)/self.Kv) / 12.0
        self.current_log = np.concatenate((self.current_log, current.T))
        return self.state
    
    def feed_forward(self, state, alpha = np.matrix([0,0]).T):
        """Determine the feed forward to satisfy given state and accelerations."""
        (D, C, G) = self.dynamics_matrices(state)
        omegas = state[2:4]
        return self.K3.I * (D*alpha + C*omegas + G + self.K4*omegas)
    
    def get_dstate(self, t, state):
        """Use the pre-assigned control law, and given state and time, to determine
        the rate of change of the state.
        """
        state = np.matrix(state).T
        u = self.control_law(t, state)
        dstate = self.dynamics(state, u).A1
        return dstate
    
    def simulate(self, t_span, initial_state = None, t_eval = None):
        """Perform the simulation using the arm's assigned control law.
        Does not update or use the arm's internal state.
        """
        if initial_state is None:
            initial_state = self.state.A1
        return solve_ivp(self.get_dstate, t_span, initial_state, t_eval = t_eval, max_step = self.loop_time)
    
    def get_voltage_log(self, sim_solution):
        """Reconstruct the voltage history using the simulation results."""
        return self.control_law(sim_solution.t, sim_solution.y)

    def get_current(self, voltage, state):
        """Get the current given a voltage and state.
        If multiple voltages and states are given as matrices where each column is an entry,
        will return all the currents corresponding to the pairs.
        """
        omegas = state[2:4]
        omegas[0,:] = omegas[0,:] * self.G1
        omegas[1,:] = omegas[1,:] * self.G2
        stall_voltage = voltage - omegas/self.Kv
        current = stall_voltage/self.Rm
        return current

    def get_current_log(self, sim_solution, voltage_log = None):
        """Reconstruct the current draw history using the simulation results."""
        if voltage_log is None:
            voltage_log = self.get_voltage_log(sim_solution)
        return self.get_current(voltage_log, sim_solution.y)
    
    def linearize(self, state = None, eps = 1e-4, dt = .005):
        """Get the arm model linearized at a stationary point at the current state.
        Generates the stationary point voltage using feed_forward() function.
        
        Arguments:
            state: state to linearize around. (Default: arm current state)
            eps: amount to +/- to x, u when taking the jacobian (Default: 1e-4)
            dt: time step for RK4 integration to estimate state-dot when taking the jacobian (Default: 0.005)
        """
        if state is None:
            state = self.state
        
        f = lambda x, u: self.unbounded_step(x, u, dt)

        (A,B) = controls.linearize(f, x = state, u = self.feed_forward(state), eps = eps)
        return (A,B)

    
    def get_K(self, state = None):
        if state is None:
            state = self.state
        
        (A,B) = self.linearize(state)
        return controls.lqr(A, B, self.Q, self.R)
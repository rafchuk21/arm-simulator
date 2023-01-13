import numpy as np
from TwoJointArm import TwoJointArm
from Trajectory import Trajectory
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

arm = TwoJointArm()

fps = 30
dt = 1/fps
t0 = 0
tf = 10

def get_arm_joints(state):
    (joint_pos, eff_pos) = arm.get_lin_joint_pos(state[:2])
    x = np.array([0, joint_pos[0,0], eff_pos[0,0]])
    y = np.array([0, joint_pos[1,0], eff_pos[1,0]])
    return (x,y)

pos_row = lambda t: np.matrix([1, t, t*t, t*t*t])
vel_row = lambda t: np.matrix([0, 1, 2*t, 3*t*t])

# state0 - initial state (theta1, theta2, omega1, omega2)
# statef - final state (theta1, theta2, omega1, omega2)
def cubic_interpolation(t0, tf, state0, statef):
    rhs = np.concatenate((state0.reshape((2,2)), statef.reshape(2,2)))
    lhs = np.concatenate((pos_row(t0), vel_row(t0), pos_row(tf), vel_row(tf)))

    coeffs = lhs.I*rhs
    return coeffs

start_state = np.concatenate((arm.inv_kinematics(np.matrix([1, -.2]).T, True), np.matrix([0,0]).T))
middle_state = np.concatenate((arm.inv_kinematics(np.matrix([-1.8, 1]).T), np.matrix([0,0]).T))
end_state = np.concatenate((arm.inv_kinematics(np.matrix([1.5, 1]).T, False), np.matrix([0,0]).T))

t0 = 0
t1 = 3
t2 = 4
t3 = 8

traj1 = Trajectory.from_coeffs(cubic_interpolation(t0, t1, start_state, middle_state), t0, t1)
traj2 = Trajectory.from_coeffs(cubic_interpolation(t1, t2, middle_state, middle_state), t1, t2)
traj3 = Trajectory.from_coeffs(cubic_interpolation(t2, t3, middle_state, end_state), t2, t3)

traj = traj1.append(traj2).append(traj3)

(xs, ys) = get_arm_joints(arm.state)
fig = plt.figure()
ax = fig.add_subplot(4,4,(1,14))
ax.axis('square')
ax.grid(True)
ax.set_xlim(-arm.l1-arm.l2, arm.l1+arm.l2)
ax.set_ylim(-arm.l1-arm.l2, arm.l1+arm.l2)
target_line, arm_line, = ax.plot(xs, ys, 'b--o', xs, ys, 'r-o')
ax.legend([arm_line, target_line], ["Current State", "Target State"], loc='lower left')

fig2, ax2 = plt.subplots()
ax2.axis('square')
ax2.grid(True)
ax2.set_xlim(-np.pi, np.pi)
ax2.set_ylim(-np.pi, np.pi)
state_line, = ax2.plot([], [])


ax_v = fig.add_subplot(4,4,4)
ax_c = fig.add_subplot(4,4,12)
ax_v.set_xlim((t0, tf))
ax_c.set_xlim((t0, tf))
ax_v.grid(True)
ax_c.grid(True)
ax_v.yaxis.set_label("Voltage (V)")
ax_c.yaxis.set_label("Current (A)")
v_line1, v_line2 = ax_v.plot([], [], 'r', [], [], 'b')
c_line1, c_line2 = ax_c.plot([], [], 'r', [], [], 'b')

# Control law for FF + kP control.
def control_law(t, state):
    kP1 = 20
    kD1 = 0
    kP2 = 20
    kD2 = 0

    PD_matrix = np.matrix([[kP1, 0, kD1, 0], [0, kP2, 0, kD2]])

    # If only one state is passed in, get the voltages
    if np.size(state, 1) == 1:
        target_state = traj.sample(t)[:4,:]
        err = target_state - state
        u = arm.feed_forward(target_state) + PD_matrix*err
    else:
        # If multiple states are passed in as a matrix, with each column representing a state, return
        # the voltages for each of those states. This is used for figuring out the voltage at each
        # time using the simulation results.
        u = []
        for i in np.arange(np.size(state, 1)):
            curr_t = t[i]
            curr_s = np.asmatrix(state[:,i]).T
            target_state = traj.sample(curr_t)[:4,:]
            err = target_state - curr_s
            new_u = arm.feed_forward(target_state) + PD_matrix*err
            if np.size(u) == 0:
                u = new_u
            else:
                u = np.concatenate((u, new_u), 1)
    return np.clip(u, -12, 12)

arm.control_law = control_law

sim_results = arm.simulate((t0, tf), initial_state = traj.sample(0)[:4,:].A1, t_eval = np.arange(t0, tf, dt))
time_vec = sim_results.t
voltage_log = arm.get_voltage_log(sim_results)
current_log = arm.get_current_log(sim_results)

ax_v.set_ylim((np.min(voltage_log), np.max(voltage_log)))
ax_c.set_ylim((np.min(current_log), np.max(current_log)))

ax_v.legend([v_line1, v_line2], ["Joint 1 Voltage", "Joint 2 Voltage"], loc='lower center', bbox_to_anchor = (0.5, -1))
ax_c.legend([c_line1, c_line2], ["Joint 1 Current", "Joint 2 Current"], loc='lower center', bbox_to_anchor = (0.5, -1))

def init():
    (xs, ys) = get_arm_joints(sim_results.y[:,0])
    arm_line.set_data(xs, ys)
    target_line.set_data(xs, ys)
    ax.set_xlim(-arm.l1-arm.l2, arm.l1+arm.l2)
    ax.set_ylim(-arm.l1-arm.l2, arm.l1+arm.l2)
    v_line1.set_data([], [])
    v_line2.set_data([], [])
    c_line1.set_data([], [])
    c_line2.set_data([], [])
    state_line.set_data([], [])
    return arm_line, target_line, v_line1, v_line2, c_line1, c_line2, state_line,

def animate(i):
    (xs, ys) = get_arm_joints(sim_results.y[:4,i])
    arm_line.set_data(xs, ys)
    (xs, ys) = get_arm_joints(traj.sample(sim_results.t[i])[:4,:])
    target_line.set_data(xs, ys)
    ax.set_xlim(-arm.l1-arm.l2, arm.l1+arm.l2)
    ax.set_ylim(-arm.l1-arm.l2, arm.l1+arm.l2)

    v_line1.set_data(time_vec[:i], voltage_log[0,:i])
    v_line2.set_data(time_vec[:i], voltage_log[1,:i])

    c_line1.set_data(time_vec[:i], current_log[0,:i])
    c_line2.set_data(time_vec[:i], current_log[1,:i])

    theta1 = sim_results.y[0, :(i+1)]
    theta2 = sim_results.y[1, :(i+1)]
    state_line.set_data(theta1, theta2)

    return arm_line, target_line, v_line1, v_line2, c_line1, c_line2, state_line,

nframes = len(sim_results.y.T)
anim = animation.FuncAnimation(fig, animate, init_func = init, frames = nframes, interval = int(dt*1000), blit=False, repeat=False)


plt.show()                                     # Uncomment this to show plot in window
#anim.save('sim.gif', writer='imagemagick')     # Uncomment this to save plot as gif
import numpy as np
from TwoJointArm import TwoJointArm
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

def eval_interpolation(coeffs, t):
    if t < 0:
        t = 0
    elif t > 3:
        t = 3
    return (np.concatenate((pos_row(t), vel_row(t))) * coeffs).reshape(4,1)

coeffs = cubic_interpolation(0, 3, np.matrix([0, 0, 0, 0]).T, np.matrix([np.pi/2, -np.pi/2, 0, 0]))

(xs, ys) = get_arm_joints(arm.state)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.axis('equal')
ax.grid(True)
ax.set_xlim((-arm.l1-arm.l2, arm.l1+arm.l2))
ax.set_ylim((-arm.l1-arm.l2, arm.l1+arm.l2))
target_line, arm_line, = ax.plot(xs, ys, 'r-o', xs, ys, 'b--o')
ax.legend([target_line, arm_line], ["Current State", "Target State"])

'''
ax_v = fig.add_subplot(2,2,2)
ax_c = fig.add_subplot(2,2,4)
ax_v.set_xlim((t0, tf))
ax_c.set_xlim((t0, tf))
v_line1, v_line2 = ax_v.plot([], [], 'r', [], [], 'b')
c_line1, c_line2 = ax_c.plot([], [], 'r', [], [], 'b')
t_vec = []
'''

def control_law(t, state):
    target_state = eval_interpolation(coeffs, t)
    err = target_state - state
    return arm.feed_forward(target_state) + 10*np.matrix([[1,0,0,0], [0,1,0,0]])*err

arm.control_law = control_law

sim_results = arm.simulate((t0, tf), t_eval = np.arange(t0, tf, dt))

def init():
    (xs, ys) = get_arm_joints(sim_results.y[:,0])
    arm_line.set_data(xs, ys)
    ax.set_xlim((-arm.l1-arm.l2, arm.l1+arm.l2))
    ax.set_ylim((-arm.l1-arm.l2, arm.l1+arm.l2))
    return arm_line, target_line

def animate(i):
    (xs, ys) = get_arm_joints(sim_results.y[:,i])
    arm_line.set_data(xs, ys)
    (xs, ys) = get_arm_joints(eval_interpolation(coeffs, sim_results.t[i]))
    target_line.set_data(xs, ys)
    ax.set_xlim((-arm.l1-arm.l2, arm.l1+arm.l2))
    ax.set_ylim((-arm.l1-arm.l2, arm.l1+arm.l2))
    return arm_line, target_line

nframes = len(sim_results.y.T)
anim = animation.FuncAnimation(fig, animate, init_func = init, frames = nframes, interval = int(dt*1000), blit=True)
plt.show()
anim.save('sim.gif', writer='imagemagick')
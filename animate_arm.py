import numpy as np
from TwoJointArm import TwoJointArm
import matplotlib.pyplot as plt
import time

arm = TwoJointArm()

dt = .01
t0 = 0
tf = 5

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
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(2,2,(1,3))
ax.set_xlim((-arm.l1-arm.l2, arm.l1+arm.l2))
ax.set_ylim((-arm.l1-arm.l2, arm.l1+arm.l2))
arm_line, target_line, = ax.plot(xs, ys, 'r', xs, ys, 'b')

ax_v = fig.add_subplot(2,2,2)
ax_c = fig.add_subplot(2,2,4)
ax_v.set_xlim((t0, tf))
ax_c.set_xlim((t0, tf))
v_line1, v_line2 = ax_v.plot([], [], 'r', [], [], 'b')
c_line1, c_line2 = ax_c.plot([], [], 'r', [], [], 'b')
t_vec = [-dt]

for t in np.arange(t0, tf, dt):
    t_vec = np.append(t_vec, t)
    target_state = eval_interpolation(coeffs, t)
    err = target_state - arm.state
    arm.step(arm.feed_forward(target_state) + 10*np.matrix([[1, 0, 0, 0], [0, 1, 0, 0]])*err, dt)
    (xs, ys) = get_arm_joints(arm.state)
    arm_line.set_xdata(xs)
    arm_line.set_ydata(ys)
    (xs, ys) = get_arm_joints(target_state)
    target_line.set_xdata(xs)
    target_line.set_ydata(ys)

    v_line1.set_xdata(t_vec)
    c_line1.set_xdata(t_vec)
    v_line1.set_ydata(arm.voltage_log[:,0].A1)
    c_line1.set_ydata(arm.current_log[:,0].A1)
    v_line2.set_xdata(t_vec)
    c_line2.set_xdata(t_vec)
    v_line2.set_ydata(arm.voltage_log[:,1].A1)
    c_line2.set_ydata(arm.current_log[:,1].A1)
    
    ax_v.set_ylim((np.min(arm.voltage_log), np.max(arm.voltage_log)))
    ax_c.set_ylim((np.min(arm.current_log), np.max(arm.current_log)))
    fig.canvas.draw()
    fig.canvas.flush_events()

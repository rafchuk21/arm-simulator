import numpy as np

class Trajectory(object):

    def __init__(self, times, states):
        self.times = times
        self.states: np.matrix = states
        self.start_time = times[0]
        self.end_time = times[-1]
    
    def clip_time(self, time):
        return np.clip(time, self.start_time, self.end_time)

    def sample(self, time):
        time = self.clip_time(time)
        prev_idx = np.where(self.times <= time)[0][-1]
        next_idx = np.where(self.times >= time)[0][0]

        if prev_idx == next_idx:
            return self.states[:, prev_idx]
        
        prev_val = self.states[:, prev_idx]
        next_val = self.states[:, next_idx]
        prev_time = self.times[prev_idx]
        next_time = self.times[next_idx]

        return (next_val - prev_val)/(next_time - prev_time)*(time-prev_time) + prev_val
    
    def append(self, other):
        combined = Trajectory(self.times, self.states)
        other.times = other.times + combined.end_time - other.start_time
        combined.times = np.concatenate((combined.times, other.times[1:]))
        combined.states = np.concatenate((combined.states, other.states[:,1:]), 1)
        combined.end_time = max(combined.times)
        return combined
    
    def from_coeffs(coeffs, t0, tf, n = 100):
        order = np.size(coeffs, 0) - 1
        t = np.linspace(t0, tf, n)
        pos_t_vec = np.power(np.array([t]).T, np.arange(order + 1))
        pos_vec = pos_t_vec * coeffs
        vel_t_vec = np.concatenate((np.zeros((n,1)), np.multiply(pos_t_vec[:, 0:-1], np.repeat(np.array([np.arange(order) + 1]), n, 0))), 1)
        vel_vec = vel_t_vec * coeffs
        acc_t_vec = np.concatenate((np.zeros((n,2)), np.multiply(vel_t_vec[:, 1:-1], np.repeat(np.array([np.arange(order - 1) + 2]), n, 0))), 1)
        acc_vec = acc_t_vec * coeffs

        states = np.asmatrix(np.concatenate((pos_vec, vel_vec, acc_vec), 1).T)
        return Trajectory(t, states)
    
    def to_table(self):
        return np.concatenate((np.array([self.times]).T, self.states.T), 1)
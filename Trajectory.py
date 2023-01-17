from __future__ import annotations
import numpy as np

class Trajectory(object):

    def __init__(self, times: np.matrix, states: np.matrix) -> Trajectory:
        """Initialize a Trajectory.
        
        Arguments:
            times: Column vector of trajectory timestamps.
            states: Matrix of corresponding states, where each state is a column
        """
        self.times: np.matrix = times
        self.states: np.matrix = states
        self.start_time = times[0]
        self.end_time = times[-1]
    
    def clip_time(self, time: float) -> np.matrix:
        """Limit Trajectory timestamp between start_time and end_time."""
        return np.matrix(np.clip(time, self.start_time, self.end_time))

    def sample(self, time: float) -> np.matrix:
        """ Sample the trajectory for the given time.
            Linearly interpolates between trajectory samples.
            If time is outside of trajectory, gives the start/end state.
        
        Arguments:
            time: time to sample
        """
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
    
    def append(self, other: Trajectory) -> Trajectory:
        """ Append another trajectory to this trajectory.
            Will adjust timestamps on the appended trajectory so it starts immediately after the
            current trajectory ends.
            Skips the first element of the other trajectory to avoid repeats.
            
        Arguments:
            other: The other trajectory to append to this one.
        """

        # Create new trajectory based off of this one
        combined = Trajectory(self.times, self.states)
        # Adjust timestamps on other trajectory
        other.times = other.times + combined.end_time - other.start_time
        # Combine the time and states
        combined.times = np.concatenate((combined.times, other.times[1:]))
        combined.states = np.concatenate((combined.states, other.states[:,1:]), 1)
        # Update the end time
        combined.end_time = max(combined.times)
        return combined
    
    def from_coeffs(coeffs: np.matrix, t0, tf, n = 100) -> Trajectory:
        """ Generate a trajectory from a polynomial coefficients matrix.
        
        Arguments:
            coeffs: Polynomial coefficients as columns in increasing order.
                    Can have arbitrarily many columns.
            t0: time to start the interpolation
            tf: time to end the interpolation
            n: number of interpolation samples (default 100)
        
        Returns:
            Trajectory following the interpolation. The states will be in the form:
            [pos1, pos2, ... posn, vel1, vel2, ... veln, accel1, ... acceln]
            Where n is the number of columns in coeffs
        """
        order = np.size(coeffs, 0) - 1
        t = np.matrix(np.linspace(t0, tf, n)).T
        pos_t_vec = np.power(t, np.arange(order + 1))
        pos_vec = pos_t_vec * coeffs
        vel_t_vec = np.concatenate((np.zeros((n,1)), np.multiply(pos_t_vec[:, 0:-1], np.repeat(np.array([np.arange(order) + 1]), n, 0))), 1)
        vel_vec = vel_t_vec * coeffs
        acc_t_vec = np.concatenate((np.zeros((n,2)), np.multiply(vel_t_vec[:, 1:-1], np.repeat(np.array([np.arange(order - 1) + 2]), n, 0))), 1)
        acc_vec = acc_t_vec * coeffs

        states = np.asmatrix(np.concatenate((pos_vec, vel_vec, acc_vec), 1).T)
        return Trajectory(t, states)
    
    def to_table(self) -> np.ndarray:
        return np.concatenate((self.times, self.states.T), 1)
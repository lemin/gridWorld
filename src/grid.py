""" grid world reinforcement learning sample program

                                    by lemin (Jake Lee)

    class GridWorld represent the grid world (similar to maze setting)
    For simplicity, learning algorithm is not implemented in a class.
"""
import numpy as np
import warnings
import collections
import sys
import matplotlib.pyplot as plt


debug = False
verbose = False


class GridWorld:
    """ Grid World environment
            there are four actions (left, right, up, and down) to move an agent
            In a grid, if it reaches a goal, it get 30 points of reward.
            If it falls in a hole or moves out of the grid world, it gets -5.
            Each step costs -1 point. 

        to test GridWorld, run the following sample codes:

            env = GridWorld('grid.txt')

            env.print_map()
            print [2,3], env.check_state([2,3])
            print [0,0], env.check_state([0,0])
            print [3,4], env.check_state([3,4])
            print [10,3], env.check_state([10,3])

            env.init([0,0])
            print env.next(1)  # right
            print env.next(3)  # down
            print env.next(0)  # left
            print env.next(2)  # up
            print env.next(2)  # up

        Parameters
        ==========
        _map        ndarray
                    string array read from a file input
        _size       1d array
                    the size of _map in ndarray
        goal_pos    tuple
                    the index for the goal location
        _actions    list
                    list of actions for 4 actions
        _s          1d array
                    current state
    """
    def __init__(self, fn):
        # read a map from a file
        self._map = self.read_map(fn)
        self._size = np.asarray(self._map.shape)
        self.goal_pos = np.where(self._map == 'G')

        # definition of actions (left, right, up, and down repectively)
        self._actions = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self._s = None

    def get_cur_state(self):
        return self._s

    def get_size(self):
        return self._size

    def read_map(self, fn):
        grid = []
        with open(fn) as f:
            for line in f:
               grid.append(list(line.strip()))
        return np.asarray(grid)

    def print_map(self):
        print self._map

    def check_state(self, s):
        if isinstance(s, collections.Iterable) and len(s) == 2:
            if s[0] < 0 or s[1] < 0 or\
               s[0] >= self._size[0] or s[1] >= self._size[1]:
               return 'N'
            return self._map[tuple(s)].upper()
        else:
            return 'F'  # wrong input

    def init(self, state=None):
        if state is None:
            s = [0, 0]
        else:
            s = state

        if self.check_state(s) == 'O':
            self._s = np.asarray(state)
        else:
            raise ValueError("Invalid state for init")

    def next(self, a):
        s1 = self._s + self._actions[a]
        # state transition
        #self._s = np.clip(s1, [0, 0], self._size-1)  # handle this later
        curr = self.check_state(s1)
        #curr = self._map[s1].upper()  

        if curr == 'H' or curr == 'N':
            return -5
        elif curr == 'F':
            warnings.warn("invalid state " + str(s1))
            return -5
        elif curr == 'G':
            self._s = s1
            return 30
        else:
            self._s = s1
            return -1
            
    def get_actions(self):
        return self._actions


def usage():
    print "Usage:"
    print "     python grid.py filename start_pos n_experiments n_episodes [plot_option]"
    print ""
    print "filename\tinput map file"
    print "start_pos\tstarting position"
    print "n_experiments\tthe number of experiments"
    print "n_episodes\tthe number of episodes per experiments"
    print "plot_option\toptional. 0 for turnning off plotting, -1 for the last episode"
    print "           \t          k for every k episode"


def epsilon_greedy(epsilon, s, na, Q):
    if np.random.uniform() < epsilon:
        a = np.random.randint(na)
        if debug:
            print "--rand action is ", a
    else:
        #a = np.argmax(Q[s[0], s[1], :])
        i_max = np.where(Q[s[0], s[1], :] == np.max(Q[s[0], s[1], :]))[0]
        a = int(np.random.choice(i_max))
        if debug:
            print "best candidates:", i_max
            print "--best action is ", a
    return a

# change the coordinate for plotting trace
def coord_convert(s, sz):
    return [s[1], sz[0]-s[0]-1]


if __name__ == '__main__':

    iplot = 0
    argc = len(sys.argv)
    if argc == 6:
        iplot = int(sys.argv[5])
    elif argc != 5:
        usage()
        sys.exit(-1)

    fn = sys.argv[1]
    st_pos = map(int, sys.argv[2].split(','))
    N = int(sys.argv[3])  # the number of experiments
    K = int(sys.argv[4])  # the number of episodes
    max_steps = 1000

    # create an GridWorld environment
    env = GridWorld(fn)

    # check start position input
    chk = env.check_state(st_pos)
    if chk != 'O':
        print "Invalid starting state position (", chk, ")"
        sys.exit(-1)

    # learning parameters 
    gamma = 0.99
    alpha = 0.1
    epsilon = 0.1
    sz = env.get_size()
    na = len(env.get_actions())

    print "========================================"
    print "  learning parameters:"
    print "\tgamma: ", gamma
    print "\talpha: ", alpha
    print "\tepsilon: ", epsilon
    print "========================================"

    if iplot:
        fig = plt.figure()
        plt.ion()

    if debug:
        np.set_printoptions(precision=3, suppress=True, linewidth=200)

    # online train
    for i in xrange(N):
        if verbose: print "experiment #", i
        # Q table including the surrounding border
        Q = np.zeros((sz[0], sz[1], na))
        Q[env._map == 'H'] = -np.inf

        # rewards and step trace
        rtrace = []
        steps = []
        for j in xrange(K):

            if verbose: print "\tepisode #", j, "   ",
            env.init(st_pos)
            s = env.get_cur_state()
            # selection an action
            a = epsilon_greedy(epsilon, s, na, Q)

            rewards = []
            trace = np.array(coord_convert(s, sz))
            for step in xrange(max_steps):
                if verbose: print "\tstep #", step, "   ",
                # move
                r = env.next(a)
                s1 = env.get_cur_state()
                a1 = epsilon_greedy(epsilon, s1, na, Q)
                
                rewards.append(r)
                trace = np.vstack((trace, coord_convert(s1, sz)))

                # update Q table (SARSA)
                Q[s[0], s[1], a] += alpha * (r + gamma * Q[s1[0], s1[1], a1] -\
                                             Q[s[0], s[1], a])

                if debug:
                    print
                    print "------------------------------------"
                    print "s :", s
                    print "a :", a
                    print "s1:", s1
                    print "a1:", a1
                    print "r1:", r
                    #print np.max(Q, axis=2)
                    for t in xrange(4):
                        print Q[...,t]
                    print "------------------------------------"

                if r == 30: # reached the goal
                    Q[s1[0], s1[1], a1] = 0
                    break

                s = s1
                a = a1


            if verbose: print "Done (", np.sum(rewards), ")", step

            rtrace.append(np.sum(rewards))
            steps.append(step+1)
            maxQ = np.max(Q, axis=2)
            if verbose: print np.max(Q, axis=2)

            last_plot = (j == K-1)
            bplot = False
            if iplot == -1:
                if last_plot:
                    bplot = True
            elif iplot > 0 and (j % iplot == 0 or last_plot):
                bplot = True

            if bplot:
                plt.clf()
                ax = fig.add_subplot(221)
                plt.plot(rtrace, "b-")
                plt.ylabel("sum of rewards")

                """
                ax1 = ax.twinx()
                ax1.plot(steps, "r-")
                ax1.set_ylabel("# steps", color='r')
                for tl in ax1.get_yticklabels():
                    tl.set_color('r')
                """

                ax1 = fig.add_subplot(222)
                plt.plot(steps)
                plt.ylabel("# steps")

                # contour plot for Q
                ax2 = fig.add_subplot(223)
                xs = range(sz[1])
                ys = range(sz[0])
                h_b = (maxQ==-np.inf)
                maxQ[h_b] = 0
                maxQ[h_b] = np.min(maxQ) - 100
                cs = plt.contourf(xs, ys[::-1], maxQ)
                plt.colorbar(cs)
                plt.text(env.goal_pos[1], sz[0]-env.goal_pos[0]-1, 'G')
                plt.text(st_pos[1], sz[0]-st_pos[0]-1, 'S')
                plt.ylabel("max Q")

                # plot traces
                ax3 = fig.add_subplot(224)
                plt.plot(trace[:, 0], trace[:, 1], "ko-")
                plt.text(env.goal_pos[1], sz[0]-env.goal_pos[0]-1, 'G')
                plt.text(st_pos[1], sz[0]-st_pos[0]-1, 'S')
                plt.xlim([0, sz[1]])
                plt.ylim([0, sz[0]])
                plt.title("trace of last episode")

                plt.suptitle(''.join(["Exp ",str(i)," Episode ",str(j)]))
                plt.draw()

        print "[", i , "]"
        print "\t Area under the reward curve:", np.sum(rtrace)
        print "\t the mean of the last 5 rewards:", np.mean(rtrace[-5:])
        print "\t the last number of steps:", steps[-1]
        print "\t the mean of the last 5 steps:", np.mean(steps[-5:])

    print "experiments are over. press enter to finish..."
    sys.stdin.readline()

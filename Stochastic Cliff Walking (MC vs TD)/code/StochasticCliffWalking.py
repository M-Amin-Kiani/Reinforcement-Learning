import sys
from contextlib import closing
from io import StringIO
import numpy as np

from gym import utils
from gym import Env, spaces
from gym.utils import seeding
from gym.utils.colorize import colorize

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

CLIFF_MAP = [
    "FFFFFFFFFFFF",
    "FFFFFFFFFFFF",
    "FFFFFFFFFFFF",
    "SCCCCCCCCCCG"
]


def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    np_random = np.random.rand()
    return (csprob_n > np_random).argmax()


class DiscreteEnv(Env):
    """
    Has the following members
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    - isd: initial state distribution (**)
    (*) dictionary of lists, where
      P[s][a] == [(probability, nextstate, reward, done), ...]
    (**) list or array of length nS
    """

    def __init__(self, nS, nA, P, isd, max_length=200, termination_penalty=-50):
        self.P = P
        self.isd = isd
        self.lastaction = None
        self.nS = nS
        self.nA = nA

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.seed()
        self.s = categorical_sample(self.isd, self.np_random)
        self.max_length = max_length
        self.termination_penalty = termination_penalty

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, **kwargs):
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        self.t = 0
        info = {}
        return int(self.s), info

    def step(self, action):
        transitions = self.P[self.s][action]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = action
        terminated = d
        truncated = False

        # Infinite loop check
        if self.t >= self.max_length and not terminated:
            truncated = True
            r = self.termination_penalty
        self.t += 1

        return (int(s), r, terminated, truncated, {"prob": p})


class StochasticCliffWalkingEnv(DiscreteEnv):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, slip_rate=0.1, gamma=0.99, max_length=200):
        self.desc = desc = np.asarray(CLIFF_MAP, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (-100, 0)
        self.gamma = gamma

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b"S").astype("float64").ravel()
        isd /= isd.sum()

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):
            newrow, newcol = inc(row, col, action)
            newletter = desc[newrow, newcol]

            if newletter == b"C":
                reward = -100
                newstate = to_s(3, 0)
                done = False
            elif newletter == b"G":
                reward = 0
                newstate = to_s(newrow, newcol)
                done = True
            else:
                reward = -1
                newstate = to_s(newrow, newcol)
                done = False

            return newstate, reward, done

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]

                    if letter == b"G":
                        # Absorbing State
                        li.append((1.0, s, 0, True))
                    else:
                        if slip_rate > 0:
                            for a_prime in range(4):
                                prob = (
                                    1.0 - slip_rate) if a_prime == a else 0.0
                                prob += slip_rate / 4.0

                                if prob > 0:
                                    next_s, reward, done = update_probability_matrix(
                                        row, col, a_prime)
                                    li.append((prob, next_s, reward, done))
                        else:
                            next_s, reward, done = update_probability_matrix(
                                row, col, a)
                            li.append((1.0, next_s, reward, done))

        super(StochasticCliffWalkingEnv, self).__init__(
            nS, nA, P, isd, max_length=max_length)

    def render(self, mode="human"):
        outfile = StringIO() if mode == "ansi" else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = list(self.desc)
        desc = [[c.decode("utf-8") for c in line] for line in desc]

        desc[row][col] = colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write(
                "  ({})\n".format(
                    ["Left", "Down", "Right", "Up"][self.lastaction])
            )
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        if mode == "ansi":
            with closing(outfile):
                if isinstance(outfile, StringIO):
                    return outfile.getvalue()

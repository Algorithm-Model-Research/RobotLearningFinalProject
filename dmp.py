import numpy as np
import math as m
import copy
from array import array
import matplotlib.pyplot as plot
from sklearn.linear_model import Ridge



class DMP(object):

    def __init__(self,w, pastor_mod = False):

####################################################
        #             Data Collection              #
        ############################################

        self.pastor_mod = pastor_mod
        #initial values
        self.x0 = 0                                              #initial position
        self.g = 20                                              #goal position
        self.step = 0.1                                          #The amount of steps in taken in a particular time frame

        #for x in range(len(step)):
        self.time = np.arange(self.x0,self.g,self.step)          #position of each step in the x axis in term of time

        #amplitude of the sine curve of a variable like time, that will be the value y; it is also collecting samples at te same time
        self.X = np.sin(self.time)                               #position
        self.k = 100
        self.d = 2.0 * np.sqrt(self.k)
        self.w = w

        #converges
        self.start = self.d / 3  # where it starts converges to 0 but won't equal 0

        self.l = 1000.0
        self.b = 20.0 / np.pi




#####################################################
        #     Implimentation DMP Learning          #
        ############################################

    def spring_damping(self):

        #Think of it as the slope of the function as it goes through

            return self.k * (self.g - self.X) - self.k * (self.g - self.x0) * self.s + self.k

    def converges(self):

        phases = np.exp(-self.start * (((np.linspace(0, 1, len(self.time))))))
        return phases #it displays the exponential converges

    def duplicate(self):

            #Vertically stack the array with y coordinates and x coordinates divided by the ammount of steps in secs
            original_matrix_1 = np.vstack((np.zeros([1, (self.g*10)], dtype = int), (self.X / self.step)))
            original_matrix_2 = np.vstack((np.zeros([1, self.g*10], dtype = int), original_matrix_1 / self.step))

            F = self.step * self.step * original_matrix_1 - self.d * (self.k * (original_matrix_1 ) - self.step * original_matrix_1)
            temp = np.zeros([200, (self.g*10)], dtype = int)

            temp[:F.shape[0],:F.shape[1]] = F
            design = np.array([self._features() for self.s in self.converges()])
            lr = Ridge(alpha=1.0, fit_intercept=False)
            lr.fit(design, temp)
            self.w = lr.coef_


            #Think of it as the x-asis of the duplicate
            return self.w

    def shape_path(self, scale=False):

        #creating a 2d vector base on the duplicate
        f = np.dot(self.w, self._features())

        return f

    def reproduction(self, o = None, shape = True, avoidance=False, verbose=0):

        if verbose <= 1:
            print("Trajectory with x0 = %s, g = %s, self.step=%.2f, step=%.3f" % (self.x0, self.g, self.step, self.step))

        #puts evething that was from X to x; from array to matrix
        x = copy.copy(self.X)
        temp_matrix_of_x1 = copy.copy(x)
        temp_matrix_of_x2 = copy.copy(x)
        X = [copy.copy(self.X)]
        original_matrix_1 = [copy.copy(temp_matrix_of_x1)]
        original_matrix_2 = [copy.copy(temp_matrix_of_x2)]

        #reproducing the x-asis
        t = 0.5 * self.step
        ti = 0

        S = self.converges()
        while t < self.step:
            t += self.step
            ti += 1
            self.s = S[ti]

            x += self.step * temp_matrix_of_x1
            temp_matrix_of_x1 += self.step * temp_matrix_of_x2

            sd = self.spring_damping()
            # the weighted shape base on the movement
            f = self.shape_path() if shape else 0.
            C = self.step.obstacle(o, x, temp_matrix_of_x1) if avoidance else 0.0
            temp_matrix_of_x2 = ()
            print(temp_matrix_of_x2)

            #Everything that you implemented in the  matrix that was temperary will initialize will be put into the none temperary matrix
            if ti % self.step == 0:
                self.X = np.append(copy.copy(x),copy.copy(self.X))
                original_matrix_1 = np.append(copy.copy(self.X),copy.copy(temp_matrix_of_x1))
                original_matrix_2 = np.append(copy.copy(self.X),copy.copy(temp_matrix_of_x2))

            #return the matrix as array when returning
            return np.array(self.X), np.array(x), np.array(temp_matrix_of_x2)


    def obstacle(self, o, original_matrix_1):

        if self.X.ndim == 1:
            self.X = self.X[np.newaxis, np.newaxis, :]
        if original_matrix_1.ndim == 1:
            original_matrix_1 = original_matrix_1[np.newaxis, np.newaxis, :]

        C = np.zeros_like(self.X)
        R = np.array([[np.cos(np.pi / 2.0), -np.sin(np.pi / 2.0)],
                        [np.sin(np.pi / 2.0),  np.cos(np.pi / 2.0)]])

        for i in xrange(self.X.shape[0]):
            for j in xrange(self.X.shape[1]):
                obstacle_diff = o - self.X[i, j]
                theta = (np.arccos(obstacle_diff.dot(original_matrix_1[i, j]) / (np.linalg.norm(obstacle_diff) * np.linalg.norm(original_matrix_1[i, j]) + 1e-10)))
                C[i, j] = (self.l * R.dot(original_matrix_1[i, j]) * theta * np.exp(-self.b * theta))

        return np.squeeze(C)

    def _features(self):

        #getting the y asis base on the x asis, since the amplitude just asolates between 1 and -1
        c = self.converges()

        #calculate the discrete difference along the y asis
        h= np.diff(c)
        h = np.hstack((h, [h[-1]]))
        phi = np.exp(-h * (self.s - c) ** 2)
        return self.s * phi / phi.sum()


def main():
#########################################################################################
    #title of the sine curve
    plot.title('Demonstration')

    #give x axis a label, it is the time
    plot.xlabel('Time represented as t')

    #give y asix a label, it is the amplitude
    plot.ylabel('Amplitude - sin(time)')

    plot.grid(True, which='both')
#########################################################################################

    w = [None]
    dmp = DMP(w,True)
    w = dmp.duplicate()
    dmp.w = w
    array1, array2, array3 = dmp.reproduction(dmp)

    array1_a = np.sin(array1)
    plot.plot(dmp.time,array1)
    plot.axhline(y=0, color='green')

    array1_b = np.sin(array2)
    plot.plot(dmp.time,array2)
    plot.axhline(y=0, color='red')

    #array1_c = np.sin(array3)
    #plot.plot(dmp.time,array3)
    plot.axhline(y=0, color='purple')
    plot.show()


if __name__ == "__main__":
    main()

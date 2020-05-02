import cv2
import numpy as np
import time
import copy
import sys

"""

todos:
try to identify interesting points: gradiants, laplace, derivatives harris
find matching corners and vpoint



"""


class partclass:

    def __init__(self, imagepart, pos):
        self.image = imagepart
        # position relative to bigframe
        self.pos = pos


class vpointclass:

    def __init__(self, f1, f2, searchradius, searchstepsize, partnumber):
        self.f1 = f1
        self.f2 = f2
        #cv2.imshow('', f1)
        # cv2.waitKey(1000)
        #cv2.imshow('', f2)
        # cv2.waitKey(1000)
        self.searchradius = searchradius
        self.searchstepsize = searchstepsize
        self.partnumber = partnumber
        self.fcut = f1[searchradius:-searchradius, searchradius:-searchradius]
        self.partmatrix = self.splitinparts()
        #self.scalarfield = [[[[1, 10]]]]
        # self.testplotpotential()
        self.scalarfield = self.scalarfieldcreator()
        self.vpoint = self.find()

    def splitinparts(self):
        # print('splitin...',n)
        shape = np.shape(self.fcut)[0:2]
        # matrix that contains normalized parts and the position of the upper left
        # corner as tuples
        return np.array([[partclass(self.normalise(self.fcut[shape[0] // self.partnumber * i: shape[0] // self.partnumber *
                                                             (i + 1) + shape[0] % self.partnumber, shape[1] // self.partnumber * j: shape[1] // self.partnumber *
                                                             (j + 1) + shape[1] % self.partnumber]), np.array([shape[0] // self.partnumber * i + self.searchradius, shape[1] // self.partnumber * j + self.searchradius])) for j in range(self.partnumber)] for i in range(self.partnumber)])

    def normalise(self, frame):
        # checks that var is not 0
        return (frame - np.mean(frame, (0, 1))) / ((np.var(frame, (0, 1)) == 0) + np.var(frame, (0, 1)))**0.5

    def scalarproduct(self, partindex, displacement):
        #print(partindex, displacement)
        bigframe = self.f2[displacement[0] + self.partmatrix[partindex].pos[0]:displacement[0] + self.partmatrix[partindex].pos[0] + np.shape(
            self.partmatrix[partindex].image)[0], displacement[1] + self.partmatrix[partindex].pos[1]:displacement[1] + self.partmatrix[partindex].pos[1] + np.shape(self.partmatrix[partindex].image)[1]]
        # select window
        bigframe = self.normalise(bigframe)
        return np.sum(bigframe * self.partmatrix[partindex].image)

    def scalarfieldcreator(self):
        scalarfield = np.zeros(
            (self.partnumber, self.partnumber, 2 * self.searchradius // self.searchstepsize, 2 * self.searchradius // self.searchstepsize))
        for index, part in np.ndenumerate(self.partmatrix):
            for i in range(-self.searchradius, self.searchradius, self.searchstepsize):
                for j in range(-self.searchradius, self.searchradius, self.searchstepsize):
                    scalarfield[index[0], index[1], (i + self.searchradius) // self.searchstepsize, (j + self.searchradius) // self.searchstepsize] =\
                        self.scalarproduct(index, (i, j))
                    # self.visualiseweights(index,i,j)

            scalarfield[index] = self.normalise(scalarfield[index])
            # set negative entries to zero(maybe dont, so that pos corner chancel)
        #scalarfield = scalarfield * (scalarfield > 0)
        return scalarfield

    def find(self):
        m = np.zeros((2, 2))
        qn = np.array([0., 0.])
        print(np.shape(self.f1))
        potential = np.zeros(np.shape(self.f1)[:2])
        nis = np.array([[np.array([-j, i]) / (((j**2 + i**2)**0.5 == 0) + (j**2 + i**2)**0.5) for i in range(-self.searchradius, self.searchradius, self.searchstepsize)]
                        for j in range(-self.searchradius, self.searchradius, self.searchstepsize)], dtype=float)

        for i1, vecmat in enumerate(self.scalarfield):
            for i2, weights in enumerate(vecmat):
                cis = np.array([[ni@self.partmatrix[i1, i2].pos for ni in vec]
                                for vec in nis])

                print(i1, i2)
                potential += self.addpotential(nis, cis, (i1, i2))

                for x, vec in enumerate(nis):
                    for y, ni in enumerate(vec):
                        m += weights[x, y] * np.tensordot(ni, ni, 0)
                        qn += weights[x, y] * ni * cis[x, y]
        self.plotpotential(potential)
        print(m)
        print(np.linalg.inv(m))
        print(qn)
        return np.linalg.inv(m)@qn

    def addpotential(self, nis, cis, partindex):
        potential = np.zeros(np.shape(self.f1)[:2])
        for index, weight in np.ndenumerate(self.scalarfield[partindex]):
            # print(index[:3])
            for pos, val in np.ndenumerate(potential):
                potential[pos] += weight / \
                    (1 + (nis[index]@pos - cis[index])**2)
        return potential

    def plotpotential(self, pot):
        #pot = np.log(pot - np.min(pot) + 1)
        print("shape of plot ", np.shape(pot))
        pot = 255 - ((pot - np.min(pot)) /
                     (np.max(pot) - np.min(pot)) * 255 + 0.5).astype(np.uint8, casting='unsafe')

        # print("middle: ", pot[36, 64],
        #      " lower right corner ", pot[71, 127])
        pot = cv2.resize(pot, (1280, 720))
        while True:
            cv2.imshow('', pot)
            cv2.waitKey(50000)

    def drawepipolarlines(self):
        pos = [(2 * p.pos + np.shape(p.image) + searchradius * 2) /
               2 for vec in self.partmatrix for p in vec]
        for p in pos:
            #print(p, vpoint)
            cv2.line(f1, (int(p[1] + 0.5), int(p[0] + 0.5)),
                     (int(self.vpoint[1] + 0.5), int(self.vpoint[0] + 0.5)), (255, 0, 0), 1)
            cv2.line(f2, (int(p[1] + 0.5), int(p[0] + 0.5)),
                     (int(self.vpoint[1] + 0.5), int(self.vpoint[0] + 0.5)), (255, 0, 0), 1)

    def visualiseweights(self, index, i, j):
        # not ready needs debugging
        shape = np.shape(self.f1)[:2]
        f0 = np.zeros(np.shape(f2), dtype=np.uint8)
        f0[searchradius + pos[0]:pos[0] + np.shape(part)[0] + searchradius, searchradius + pos[1]:searchradius + pos[1] + np.shape(part)[1]] = f1[shape[0] // n * i: shape[0] // n *
                                                                                                                                                  (i + 1) + shape[0] % n, shape[1] // n * j: shape[1] // n *
                                                                                                                                                  (j + 1) + shape[1] % n]
        print(f0)
        cv2.imshow(str(weight), f2)
        cv2.waitKey(200)
        cv2.imshow(str(weight), f0)
        cv2.waitKey(200)
        cv2.imshow(str(weight), f0 // 2 + f2 // 2)
        cv2.waitKey(200)
        cv2.destroyAllWindows()

    def testplotpotential(self):
        nis = np.array([[[1, 0], [0, 1]]])
        cis = np.array(
            [[0.5 * np.shape(self.f1)[0], 0.5 * np.shape(self.f1)[1]]])
        self.plotpotential(self.addpotential(nis, cis))


def showclipwithframenumber(cap):
    l = []
    ind = 0
    forward = True
    while True:
        if forward & (ind + 1 > len(l)):
            _, f1 = cap.read()
            l.append(f1)
            ind += 1
        elif forward:
            ind += 1
        else:
            ind -= 1
        cv2.imshow('asdf', l[ind - 1])
        print(ind)
        while True:
            if cv2.waitKey(1) & 0xFF == ord('w'):
                forward = True
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                forward = False
                break


cap = cv2.VideoCapture('flasche.mp4')
# showclipwithframenumber(cap)
searchradius = 8
searchstepsize = 2
#_, f1 = cap.read()
# print(np.shape(f1))
# for k in range(10):
for i in range(45):
    _, f1 = cap.read()
_, f2 = cap.read()
f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)

print(np.shape(f1))
print(tuple(np.array(np.shape(f1)[1::-1]) // 2))
f2 = cv2.resize(f1, tuple(np.array(np.shape(f1)[1::-1]) // 2))
f1 = np.minimum(f1 - cv2.resize(f2, np.shape(f1)
                                [1::-1]), cv2.resize(f2, np.shape(f1)[1::-1]) - f1)

_, f1 = cv2.threshold(f1, 10, 255, cv2.THRESH_BINARY)
print(f1, max([k for i in f1 for k in i]))
cv2.imshow('  ',  f1)
cv2.waitKey(0)

"""
h = cv2.cornerHarris(np.float32(
    cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)), 2, 3, 0.04)
f1[h > 0.01 * h.max()] = [0, 0, 255]
while True:
    cv2.imshow('', f1)
    cv2.waitKey(500000)
    cv2.imshow('', f2)
    cv2.waitKey(50)

#f1 = cv2.resize(f1, (128, 72))
#f2 = cv2.resize(f2, (128, 72))
# print(np.shape(f1))
cf = copy.deepcopy(f1)
vp = vpointclass(f1, f2, 80, 20, 3)

scalrfield, pos, smallframeshape = scalarfieldcreator(cf, f2, 9)
vpoint = find(scalrfield, pos, smallframeshape)
drawepipolarlines(pos, vpoint, f1, f2, smallframeshape)

while True:
    cv2.imshow('', f1)
    cv2.waitKey(500)
    cv2.imshow('', f2)
    cv2.waitKey(500)


#print(splitinparts(f1, 3))

f1 = f1[50:-50, 50:-50]
frames = splitinparts(f1, 3)
print(frames[0, 0])
print(np.shape(frames[0, 0]))
x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(x)
print(np.mean(frames, (2, 3)))
print(np.var(frames, (2, 3)))
#print((frames - np.mean(frames, (2, 3))) / np.var(frames, (2, 3)))
frames = np.array([[(i - np.mean(i, (0, 1))) / np.var(i, (0, 1))
                    ** 0.5 for i in vec] for vec in frames])
f2 = f2 - np.mean(f2)
print(frames)
print(np.shape(frames))

while True:

    cv2.imshow('', f2)
    cv2.waitKey(500)

print(scalarproduct(np.array([[1, 2], [0, 1]]),
                    np.array([[1, 1], [5, 1]]), [0, 0]), 'asfd')


"""

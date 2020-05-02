from copy import deepcopy
import numpy as np
import quaternion
from scipy.signal import convolve
import cProfile
#import torch
import fractions
import time
#import pyautogui
#import cv2
import sys
import re

reg1 = re.compile(r'<(?P<MailAdress>[^<>@]+@[^<>@]+)>')
reg = re.compile(
    r'Bestellung (?P<AuftragsNummer>\d+), Kundennummer (?P<KundenNummer>\d+), (?P<Name>[^\d]+)')
print(reg.search("Fwd: Bestellung 14490, Kundennummer 1017792, Nicole Bleckmann"))
print(reg.findall("Fwd: Bestellung 14490, Kundennummer 1017792, Nicole Bleckmann"))
print(reg.match("asfd"))
#reg = re.compile(r'<(?P<MailAdress>*?)>')
# print(reg.match("asdfjkbasdf<service@parfuemerie-noeth-shop.de>pjibsihupafewbu"))

#a = np.zeros(4)
# print(a)
#re.compile(r'\d+ \(UID (?P<uid>\d+)\)')

"""
a = np.array([1, 2, 3])
print(np.einsum('i,j->ij', a, a))
x = np.array([[1, 2], [3, 4]])
one = np.ones((2, 2))
xone = np.concatenate((x, one), axis=1)
onex = np.concatenate((one, x), axis=1)
netz = [6, 7, 8, 9]
print(np.einsum('ij,kj,j->ik', xone, onex, netz))
print(np.einsum('ij,ij->ij', x, x))

p = []
while len(p) < 2:
    a = input()
    p.append(pyautogui.position())
    print(p[-1])

img = pyautogui.screenshot(
    region=(p[0][0], p[0][1], p[1][0] - p[0][0], p[1][1] - p[0][1]))
print(pyautogui.locateOnScreen(img))

#cv2.imshow('asdf', np.array(img))
# cv2.waitKey(0)





pyautogui.moveTo(1800, 10, 2)
pyautogui.click()
pyautogui.click(1500, 300, button='right')
pyautogui.move(20, 20, 2)
pyautogui.move(0, 250, 2)
pyautogui.click()
pyautogui.move(-50, 0, 2)
pyautogui.move(0, 200, 2)
pyautogui.click()
for i in 'baobao':
    pyautogui.press(i)
    time.sleep(.5)
pyautogui.moveTo(1500, 333, 2)
pyautogui.click(clicks=2)
time.sleep(2)
for i in 'ich liebe dich':
    pyautogui.press(i)
    time.sleep(.5)
while True:
    print(pyautogui.position())
    time.sleep(1)
    
"""

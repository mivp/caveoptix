#/usr/bin/env python
import caveoptix

#from math import *
#from euclid import *
from omega import *
from cyclops import *
from omegaToolkit import *

cg = caveoptix.initialize()
cg.initOptix()

cam = getDefaultCamera()
cam.getController().setSpeed(10)
setNearFarZ(0.1, 1000)

mm = MenuManager.createAndInitialize()
menu = mm.getMainMenu()
mm.setMainMenu(menu)

#cmd = 'cam.setPosition(Vector3(' + str(campos[0]) + ',' + str(campos[1]) + ',' + str(campos[2]) + ')),' + \
#               'cam.setOrientation(Quaternion(' + str(camori[0]) + ',' + str(camori[1]) + ',' + str(camori[2]) + ',' + str(camori[3]) + '))'
menu.addButton("Go to camera 1", 'cam.setPosition(7.0, 9.2, -6.0)')

#cam.setPostion(7.0, 9.2, -6.0)
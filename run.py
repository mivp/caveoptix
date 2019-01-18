#/usr/bin/env python
import cavegvdb

#from math import *
#from euclid import *
from omega import *
from cyclops import *
from omegaToolkit import *

cg = cavegvdb.initialize()
cg.initGvdb('terraindata/aus/aus_config.ini')

cam = getDefaultCamera()
cam.getController().setSpeed(100)
setNearFarZ(1, 10000)

mm = MenuManager.createAndInitialize()
menu = mm.getMainMenu()
mm.setMainMenu(menu)

#cmd = 'cam.setPosition(Vector3(' + str(campos[0]) + ',' + str(campos[1]) + ',' + str(campos[2]) + ')),' + \
#		'cam.setOrientation(Quaternion(' + str(camori[0]) + ',' + str(camori[1]) + ',' + str(camori[2]) + ',' + str(camori[3]) + '))'
menu.addButton("Go to camera 1", 'cam.setPosition(0, 0, 700)')

cam.setPostion(0, 0, 700)
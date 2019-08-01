import gym
import pybullet as p 
import numpy as np
import pybullet_data
import os
urdfRoot=pybullet_data.getDataPath()
p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI,1)
p.resetSimulation()
p.loadSDF(os.path.join(urdfRoot,"stadium.sdf"))
sphereRadius = 0.05
mass = 1
visualShapeId = -1
colSphereId = p.createCollisionShape(p.GEOM_SPHERE,radius=sphereRadius)
sphereUid = p.createMultiBody(mass,colSphereId,visualShapeId,[0,0,0])





for  i in range(0,1000000):
	p.stepSimulation()
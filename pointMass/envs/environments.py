
import gym, gym.utils, gym.utils.seeding
import pybullet as p 
import numpy as np
import pybullet_data
import os
import time
from pybullet_utils import bullet_client
urdfRoot=pybullet_data.getDataPath()
import gym.spaces as spaces
import math

GUI = False
viewMatrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition = [0,0,0], distance = 6, yaw = 0, pitch = -90, roll = 0, upAxisIndex = 2)


projectionMatrix = p.computeProjectionMatrixFOV(fov = 50,aspect = 1,nearVal = 0.01,farVal = 10)


class pointMassEnv(gym.GoalEnv):


		metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second': 60
		}

		def __init__(self, render = False, use_object = False, sparse = True, TARG_LIMIT = 3):
			self.use_object = use_object

			action_dim = 2
			obs_dim = 4

			if use_object:
				obs_dim += 4 # pos and vel of the other pm that we are knocking around.
			goal_dim = 2
			high = np.ones([action_dim])
			self.action_space = spaces.Box(-high, high)
			high_obs = np.inf * np.ones([obs_dim])
			high_goal = np.inf * np.ones([goal_dim])
			self.observation_space = spaces.Dict(dict(
	            desired_goal=spaces.Box(-high_goal, high_goal),
	            achieved_goal=spaces.Box(-high_goal, high_goal),
	            observation=spaces.Box(-high_obs, high_obs),
        	))

			
			self.isRender = False
			self._p = p
			self.physics_client_active = 0
			self.movable_goal = False
			self.roving_goal = False
			self.TARG_LIMIT = TARG_LIMIT
			self.TARG_MIN = 0.1
			self._seed()
			self.global_step = 0
			self.opposite_goal = False
			self.show_goal = False

			if sparse:
				self.set_sparse_reward()
			

		def crop(self, num, lim):
				if num >= 0 and num < lim:
					num = lim
				elif num < 0 and num > -lim:
					num = -lim
				return num
			

		def reset_goal_pos(self, goal = None):
			

			if goal is None: 
				self.goal_x  = self.crop(self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), self.TARG_MIN)
				self.goal_y  = self.crop(self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), self.TARG_MIN)
			else:
				self.goal_x = goal[0]
				self.goal_y = goal[1]
			#self.goal_velocity = self.np_random.uniform(low=0, high=3)
			# try:
			# 	self._p.removeUserDebugItem(self.goal)
			# except:
			# 	pass

			# self.goal = self._p.addUserDebugText("o", [self.goal_x, self.goal_y, 0.1],
   #                 textColorRGB=[0, 0, 1],
   #                 textSize=2)

			if self.isRender:
				if self.show_goal:
					self._p.resetBasePositionAndOrientation(self.goal, [self.goal_x, self.goal_y,0.1], [0,0,0,1])
					self._p.changeConstraint(self.goal_cid,[self.goal_x, self.goal_y,0.1], maxForce = 100)

		def reset_object_pos(self, o = None, extra_info = None, curric = False):
			if o is None:
				current_pos = self._p.getBasePositionAndOrientation(self.mass)[0]
				if curric == True:

					vector_to_goal = np.array([self.goal_x-current_pos[0], self.goal_y-current_pos[1],0.6])

					pos = np.array(current_pos)+vector_to_goal/2+(np.random.rand(3)*1)-0.5

					# shift it a little if too close to the goal
				pos = np.random.rand(3)*4-2
				while self.calc_target_distance(pos[0:2], [self.goal_x, self.goal_y]) < 1:
					pos = pos + (np.random.rand(3)*1)-0.5
				while self.calc_target_distance(pos[0:2], [current_pos[0], current_pos[1]]) < 1:
					pos = pos + (np.random.rand(3)*1)-0.5
				pos[2] = 0.4
				ori = [0,0,0,1]
				obs_vel_x, obs_vel_y =0,0
			else:
				obs_x, obs_y = o[4], o[5]
				obs_z = 0.4 if extra_info is None else extra_info[0]
				pos = [obs_x, obs_y, obs_z]
				ori = [0,0,0,1] if extra_info is None else extra_info[1:5]
				obs_vel_x, obs_vel_y = o[6], o[7]

			self._p.resetBasePositionAndOrientation(self.object, pos,ori)
			self._p.resetBaseVelocity(self.object, [obs_vel_x, obs_vel_y, 0])


		def initialize_actor_pos(self,o):
			x, y, x_vel, y_vel = o[0], o[1], o[2], o[3]
			self._p.resetBasePositionAndOrientation(self.mass, [x, y, -0.1], [0, 0, 0, 1])
			self._p.changeConstraint(self.mass_cid, [x, y, -0.1], maxForce=100)
			self._p.resetBaseVelocity(self.mass, [x_vel, y_vel, 0])

		#TODO change the env initialise start pos to a more general form of the function

		def initialize_start_pos(self, o, extra_info = None):
			if type(o) is dict:
				o = o['observation']
			self.initialize_actor_pos(o)
			if self.use_object:
				self.reset_object_pos(o, extra_info)



			
		def calc_state(self):

			# state will be x,y pos, total velocity, x,y goal.
			current_pos = self._p.getBasePositionAndOrientation(self.mass)[0]
			x,y = current_pos[0], current_pos[1]
			velocity = self._p.getBaseVelocity(self.mass)[0]
			x_vel, y_vel = velocity[0], velocity[1]
			#print(x_vel, y_vel)
			velocity_mag = (np.sum(np.array(self._p.getBaseVelocity(self.mass)[0])[0:2])**2)**(1/2)
			obs = [x,y,x_vel, y_vel]

			goal = np.array([self.goal_x, self.goal_y])

			if self.use_object:
				obj_pose = self._p.getBasePositionAndOrientation(self.object)
				obs_x, obs_y = obj_pose[0][0], obj_pose[0][1]
				velocity_obs = self._p.getBaseVelocity(self.object)[0]
				x_vel_obj, y_vel_obj = velocity_obs[0], velocity_obs[1]

				obs += [obs_x, obs_y, x_vel_obj, y_vel_obj]
				achieved_goal = np.array([obs_x, obs_y])
				extra_info = np.squeeze(np.array(
					[list([obj_pose[0][2]] + list(obj_pose[1]))])).astype('float32')  # z pos of the object, ori quaternion of the object.

			else: # ag is the position of our controlled point mass
				achieved_goal = np.array([x,y])
				extra_info = None


			return_dict= {
	            'observation': np.array(obs).copy().astype('float32'),
	            'achieved_goal': achieved_goal.copy().astype('float32'),
	            'desired_goal':  goal.copy().astype('float32'),
	            'extra_info': extra_info
            }

			if self.isRender:
				img = p.getCameraImage(48, 48, viewMatrix, projectionMatrix, shadow=0,
									   flags=p.ER_NO_SEGMENTATION_MASK, renderer=p.ER_BULLET_HARDWARE_OPENGL)
				return_dict['image'] = img[2][:,:,:3]



			return return_dict
			

		#TODO replace with cleaner code
		def calc_target_distance(self, achieved_goal, desired_goal):
			

			x,y = achieved_goal[0], achieved_goal[1]
			goal_x, goal_y = desired_goal[0], desired_goal[1]
			distance = abs(goal_x-x) + abs(goal_y-y)

			return distance

		# def calc_velocity_distance(self):
		# 	velocity = (np.sum(np.array(self._p.getBaseVelocity(self.mass)[0])[0:2])**2)**(1/2)
			
		# 	return (velocity - self.goal_velocity)


		def activate_movable_goal(self):
			self.movable_goal = True

		def activate_roving_goal(self):
			self.roving_goal = True

		def compute_reward(self, achieved_goal, desired_goal, info = None):
			
			# reward given if new pos is closer than old
			current_distance = self.calc_target_distance(achieved_goal, desired_goal)

			position_reward = -1000*(current_distance - self.last_target_distance)
			self.last_target_distance = current_distance
			


			# velocity_diff = self.calc_velocity_distance()
			# velocity_reward = -100*(velocity_diff - self.last_velocity_distance)
			# self.last_velocity_distance = velocity_diff
			#velocity_reward = self.calc_velocity_distance()

			#print('Vreward', velocity_reward)

			return position_reward#+velocity_reward

		def set_sparse_reward(self):
			print('Environment set to sparse reward')
			self.compute_reward = self.compute_reward_sparse

		def compute_reward_sparse(self, achieved_goal, desired_goal, info = None):
			current_distance = self.calc_target_distance(achieved_goal, desired_goal)
			reward = -1
			if current_distance < 0.5:
				reward = 0

			return reward


		def step(self, action):

			action = action *0.1# put it to the correct scale
			x_shift, y_shift = action[0], action[1]
			current_pos = self._p.getBasePositionAndOrientation(self.mass)[0]
			x,y = current_pos[0], current_pos[1]

			new_x, new_y = np.clip(x+x_shift,-self.TARG_LIMIT*2, self.TARG_LIMIT*2), np.clip(y+y_shift,-self.TARG_LIMIT*2, self.TARG_LIMIT*2)
			self._p.changeConstraint(self.mass_cid,[new_x, new_y, -0.1], maxForce = 10)

			# force = action*10
			# print(force)
			# self._p.applyExternalForce(self.mass, -1, force, current_pos, flags=self._p.WORLD_FRAME)
			#
			for i in range(0,20):
				#self._p.applyExternalForce(self.mass, -1, force, current_pos, flags=self._p.WORLD_FRAME)
				self._p.stepSimulation()


			obs = self.calc_state()

			r = self.compute_reward(obs['achieved_goal'], obs['desired_goal'])

			current_distance = self.calc_target_distance(obs['achieved_goal'], [self.goal_x, self.goal_y])

			if self.movable_goal:

				if current_distance < 0.6:
					self.reset_goal_pos()

			if self.roving_goal:
				if self.global_step % 60 == 0:

					self.reset_goal_pos()

			self.global_step += 1



			return obs, r, False, {}

		def reset(self):
			#self._p.resetSimulation()
			self.global_step = 0
			
			if self.physics_client_active == 0:
				
				if self.isRender:
					self._p = bullet_client.BulletClient(connection_mode=p.GUI)
				else:
					self._p = bullet_client.BulletClient(connection_mode=p.DIRECT)

				self.physics_client_active = 1

				sphereRadius = 0.25
				mass = 1
				visualShapeId = 2
				colSphereId = self._p.createCollisionShape(p.GEOM_SPHERE,radius=sphereRadius)
				self.mass = self._p.createMultiBody(mass,colSphereId,visualShapeId,[0,0,0.4])
				#objects = self._p.loadMJCF("/Users/francisdouglas/bullet3/data/mjcf/sphere.xml")
				#self.mass = objects[0]
				# self.mass = [p.loadURDF((os.path.join(urdfRoot,"sphere2.urdf")), 0,0.0,1.0,1.00000,0.707107,0.000000,0.707107)]
				relativeChildPosition=[0,0,0]
				relativeChildOrientation=[0,0,0,1]
				self.mass_cid = self._p.createConstraint(self.mass,-1,-1,-1,self._p.JOINT_FIXED,[0,0,0],[0,0,0],relativeChildPosition,relativeChildOrientation)



				if self.isRender:
					if self.show_goal:
						self.goal = self._p.createMultiBody(mass,colSphereId,visualShapeId,[1,1,1.4])
						collisionFilterGroup = 0
						collisionFilterMask = 0
						self._p.setCollisionFilterGroupMask(self.goal, -1, collisionFilterGroup, collisionFilterMask)
						self.goal_cid = self._p.createConstraint(self.goal,-1,-1,-1,self._p.JOINT_FIXED,[1,1,1.4],[0,0,0],relativeChildPosition,relativeChildOrientation)
					#self._p.setRealTimeSimulation(1)
					
				if self.use_object:
					colcubeId = self._p.createCollisionShape(p.GEOM_BOX,halfExtents=[0.4,0.4,0.4])
					self.object = self._p.createMultiBody(0.1,colcubeId ,2,[0,0,1.5])
					#self.object = self._p.createMultiBody(mass,colSphereId,visualShapeId,[0.5,0.5,0.4])
					colwallId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 2.5, 0.5])
					wall = [p.createMultiBody(0, colwallId, 2, [self.TARG_LIMIT*2+0.2, 0, 0.0], p.getQuaternionFromEuler([0, 0, 0]))]
					wall = [p.createMultiBody(0, colwallId, 2, [-self.TARG_LIMIT*2-0.2, 0, 0.0], p.getQuaternionFromEuler([0, 0, 0]))]
					wall = [
						p.createMultiBody(0, colwallId, 2, [0, self.TARG_LIMIT*2+0.2, 0], p.getQuaternionFromEuler([0, 0, math.pi / 2]))]
					wall = [
						p.createMultiBody(0, colwallId, 2, [0, -self.TARG_LIMIT*2-0.2, 0], p.getQuaternionFromEuler([0, 0, math.pi / 2]))]

				if GUI:
					
					ACTION_LIMIT = 1
					self.x_shift = self._p.addUserDebugParameter("X", -ACTION_LIMIT, ACTION_LIMIT, 0.0)
					self.y_shift= self._p.addUserDebugParameter("Y", -ACTION_LIMIT, ACTION_LIMIT, 0.0)

				self._p.configureDebugVisualizer(p.COV_ENABLE_GUI,GUI)

				self._p.setGravity(0,0,-10)
				lookat = [0, 0, 0.1]
				distance = 7
				yaw = 0
				self._p.resetDebugVisualizerCamera(distance, yaw, -89, lookat)
				colcubeId = self._p.createCollisionShape(p.GEOM_BOX, halfExtents=[5, 5, 0.1])
				visplaneId = self._p.createVisualShape(p.GEOM_BOX, halfExtents=[5, 5, 0.1],rgbaColor=[1,1,1,1])
				plane = self._p.createMultiBody(0, colcubeId, visplaneId, [0, 0, -0.2])

				#self._p.loadSDF(os.path.join(urdfRoot, "plane_stadium.sdf"))

				
			


			self._p.resetBasePositionAndOrientation(self.mass, [0, 0,0.6], [0,0,0,1])
			self.reset_goal_pos()
			
			# reset mass location
			if self.opposite_goal:
				x=  -self.goal_x
				y = -self.goal_y
			else:
				x  = self.crop(self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT),self.TARG_MIN)
				y  = self.crop(self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT),self.TARG_MIN) 
			x_vel = 0#self.np_random.uniform(low=-1, high=1)
			y_vel = 0#self.np_random.uniform(low=-1, high=1)

			self.initialize_actor_pos([x,y,x_vel,y_vel])
			if self.use_object:
				self.reset_object_pos()

			obs = self.calc_state()
			self.last_target_distance = self.calc_target_distance(obs['achieved_goal'],obs['desired_goal'])
			#self.last_velocity_distance = self.calc_velocity_distance()




			#Instantiate some obstacles.
			#self.instaniate_obstacles()
			return obs

		def instaniate_obstacles(self):
			colcubeId = self._p.createCollisionShape(p.GEOM_BOX,halfExtents=[1.5,1.5,0.6])
			self.obstacle_center = self._p.createMultiBody(0,colcubeId ,2,[0,0,0.0])
			self.TARG_MIN = 1.5
			self.opposite_goal = True


		def render(self, mode):

			if (mode=="human"):
				self.isRender = True
				return np.array([])
			if mode == 'rgb_array':
				raise NotImplementedError


		def close(self):
			print('closing')
			self._p.disconnect()


		def _seed(self, seed=None):
			print('seeding')
			self.np_random, seed = gym.utils.seeding.np_random(seed)

			return [seed]





# a version where reward is determined by the obstacle. 
# need a sub class so we can load it as a openai gym env.
class pointMassEnvObject(pointMassEnv):
    def __init__(self,
                 render=False,
                 use_object = True, sparse = True, TARG_LIMIT = 1.3
                 ):
        super().__init__(render = render, use_object = use_object, sparse = sparse, TARG_LIMIT = TARG_LIMIT)

class pointMassEnvObjectDense(pointMassEnv):
    def __init__(self,
                 render=False,
                 use_object = True, sparse = False, TARG_LIMIT = 1.3
                 ):
        super().__init__(render = render, use_object = use_object, sparse = sparse, TARG_LIMIT = TARG_LIMIT)





def main(**kwargs):

	cameraDistance = 1
	cameraYaw = 35
	cameraPitch = -35

	env = pointMassEnvObject()
	env.render(mode = 'human')
	obs = env.reset()['observation']

	# objects = env._p.loadMJCF("/Users/francisdouglas/bullet3/data/mjcf/sphere.xml")
	# sphere = objects[0]
	# env._p.resetBasePositionAndOrientation(sphere, [0, 0, 1], [0, 0, 0, 1])
	# env._p.changeDynamics(sphere, -1, linearDamping=0.9)
	# env._p.changeVisualShape(sphere, -1, rgbaColor=[1, 0, 0, 1])
	forward = 0
	turn = 0

	forwardVec = [2, 0, 0]
	cameraDistance = 2
	cameraYaw = 35
	cameraPitch = -65
	steps = 0
	while steps<3000:

		spherePos, orn = env._p.getBasePositionAndOrientation(env.mass)

		cameraTargetPosition = spherePos
		env._p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition)
		camInfo = env._p.getDebugVisualizerCamera()
		camForward = camInfo[5]

		keys = env._p.getKeyboardEvents()
		for k, v in keys.items():

			if (k == env._p.B3G_RIGHT_ARROW and (v & env._p.KEY_WAS_TRIGGERED)):
				turn = -3
			if (k == env._p.B3G_RIGHT_ARROW and (v & env._p.KEY_WAS_RELEASED)):
				turn = 0
			if (k == env._p.B3G_LEFT_ARROW and (v & env._p.KEY_WAS_TRIGGERED)):
				turn = 3
			if (k == env._p.B3G_LEFT_ARROW and (v & env._p.KEY_WAS_RELEASED)):
				turn = 0

			if (k == env._p.B3G_UP_ARROW and (v & env._p.KEY_WAS_TRIGGERED)):
				forward = 0.6
			if (k == env._p.B3G_UP_ARROW and (v & env._p.KEY_WAS_RELEASED)):
				forward = 0
			if (k == env._p.B3G_DOWN_ARROW and (v & env._p.KEY_WAS_TRIGGERED)):
				forward = -0.6
			if (k == env._p.B3G_DOWN_ARROW and (v & env._p.KEY_WAS_RELEASED)):
				forward = 0


		force = [forward * camForward[0], forward * camForward[1], 0]
		cameraYaw = cameraYaw + turn

		# if (forward):
		# 	env._p.applyExternalForce(sphere, -1, force, spherePos, flags=env._p.WORLD_FRAME)
		#
		# env._p.stepSimulation()
		time.sleep(3. / 240.)

		env.step(np.array(force))
		steps += 1
		
		

if __name__ == "__main__":
    main()



# env = pointMassEnv()


# env.reset()

# for i in range(0,100):

# 	#env._p.stepSimulation()
# 	time.sleep(0.005)
# 	#action = [env._p.readUserDebugParameter(env.x_shift), env._p.readUserDebugParameter(env.y_shift)]
# 	o2, r, d, _ = env.step(np.ones(2))
# 	print(o2)

# test_env = pointMassEnv()

# test_env.render(mode = 'human')
# test_env.reset()

# for i in range(0,100000):

# 	#env._p.stepSimulation()
# 	time.sleep(0.005)
# 	action = np.array([test_env._p.readUserDebugParameter(test_env.x_shift), test_env._p.readUserDebugParameter(test_env.y_shift)])
# 	o2, r, d, _ = test_env.step(action)
# 	print(o2)




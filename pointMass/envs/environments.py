
import gym, gym.utils, gym.utils.seeding
import pybullet as p 
import numpy as np
import pybullet_data
import os
import time
from pybullet_utils import bullet_client
urdfRoot=pybullet_data.getDataPath()
import gym.spaces as spaces

GUI = False
class pointMassEnv(gym.GoalEnv):


		metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second': 60
		}

		def __init__(self, render = False):
			action_dim = 2
			obs_dim = 4
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
			self.TARG_LIMIT = 3
			self._seed()
			


			

		def reset_goal_pos(self, goal = None):
			if goal == None:
				self.goal_x  = self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT)
				self.goal_y  = self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT)
			else:
				self.goal_x = goal_x
				self.goal_y = goal_y
			#self.goal_velocity = self.np_random.uniform(low=0, high=3)
			
			self._p.resetBasePositionAndOrientation(self.goal, [self.goal_x, self.goal_y,0.4], [0,0,0,1])
			self._p.changeConstraint(self.goal_cid,[self.goal_x, self.goal_y,0.2], maxForce = 100)

		def initalize_start_pos(self, s_i, v_i):
			x,y,x_vel,y_vel = s_i[0],s_i[1],v_i[0],v_i[1] 
			self._p.resetBasePositionAndOrientation(self.mass, [x, y,-0.1], [0,0,0,1])
			self._p.changeConstraint(self.mass_cid,[x, y,-0.1], maxForce = 100)
			self._p.resetBaseVelocity(self.mass,[x_vel, y_vel, 0])




			
		def calc_state(self):

			# state will be x,y pos, total velocity, x,y goal.
			current_pos = self._p.getBasePositionAndOrientation(self.mass)[0]
			x,y = current_pos[0], current_pos[1]
			velocity = self._p.getBaseVelocity(self.mass)[0]
			x_vel, y_vel = velocity[0], velocity[1]
			#print(x_vel, y_vel)
			velocity_mag = (np.sum(np.array(self._p.getBaseVelocity(self.mass)[0])[0:2])**2)**(1/2)

			obs = np.array([x,y,x_vel, y_vel])
			achieved_goal = np.array([x,y])
			goal = np.array([self.goal_x,self.goal_y])
			return {
	            'observation': obs.copy(),
	            'achieved_goal': achieved_goal.copy(),
	            'desired_goal':  goal.copy(),
            }
			


		def calc_target_distance(self, achieved_goal, desired_goal):
			
			current_pos = self._p.getBasePositionAndOrientation(self.mass)[0]
			x,y = current_pos[0], current_pos[1]
			
			distance = abs(self.goal_x-x) + abs(self.goal_y-y)

			return distance

		# def calc_velocity_distance(self):
		# 	velocity = (np.sum(np.array(self._p.getBaseVelocity(self.mass)[0])[0:2])**2)**(1/2)
			
		# 	return (velocity - self.goal_velocity)


		def activate_movable_goal(self):
			self.movable_goal = True

		def compute_reward(self, achieved_goal, desired_goal, info = None):
			
			# reward given if new pos is closer than old
			current_distance = self.calc_target_distance(achieved_goal, desired_goal)

			position_reward = -1000*(current_distance - self.last_target_distance)
			self.last_target_distance = current_distance
			
			
			if self.movable_goal:

				if current_distance < 2:
					self.reset_goal_pos()

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
			reward = 0 
			if current_distance < 0.5:
				reward = 1

			if self.movable_goal:
				if current_distance < 2:
					self.reset_goal_pos()
			
			return reward


		def step(self, action):
			action = action * 0.13 # put it to the correct scale
			x_shift, y_shift = action[0], action[1]
			current_pos = self._p.getBasePositionAndOrientation(self.mass)[0]
			x,y = current_pos[0], current_pos[1]
			new_x, new_y = np.clip(x+x_shift,-self.TARG_LIMIT, self.TARG_LIMIT), np.clip(y+y_shift,-self.TARG_LIMIT, self.TARG_LIMIT)
			self._p.changeConstraint(self.mass_cid,[new_x, new_y, -0.1], maxForce = 4)
			#print(self._p.getBaseVelocity(self.mass)[0])
			#print(self._p.getBasePositionAndOrientation(self.mass)[0])
			
			for i in range(0,10):
				self._p.stepSimulation()

			obs = self.calc_state()
			r = self.compute_reward(obs['achieved_goal'], obs['desired_goal'])

			return obs, r, False, {}


		def reset(self):
			#self._p.resetSimulation()

			
			if self.physics_client_active == 0:
				
				if self.isRender:
					self._p = bullet_client.BulletClient(connection_mode=p.GUI)
				else:
					self._p = bullet_client.BulletClient(connection_mode=p.DIRECT)

				self.physics_client_active = 1

				sphereRadius = 0.1
				mass = 1
				visualShapeId = 2
				colSphereId = self._p.createCollisionShape(p.GEOM_SPHERE,radius=sphereRadius)
				self.mass = self._p.createMultiBody(mass,colSphereId,visualShapeId,[0,0,0.4])
				self.goal = self._p.createMultiBody(mass,colSphereId,visualShapeId,[1,1,1.4])
				# self.mass = [p.loadURDF((os.path.join(urdfRoot,"sphere2.urdf")), 0,0.0,1.0,1.00000,0.707107,0.000000,0.707107)]
				relativeChildPosition=[0,0,0]
				relativeChildOrientation=[0,0,0,1]
				self.mass_cid = self._p.createConstraint(self.mass,-1,-1,-1,self._p.JOINT_FIXED,[0,0,0],[0,0,0],relativeChildPosition,relativeChildOrientation)
				self.goal_cid = self._p.createConstraint(self.goal,-1,-1,-1,self._p.JOINT_FIXED,[1,1,1.4],[0,0,0],relativeChildPosition,relativeChildOrientation)

				if GUI:
					self._p.setRealTimeSimulation(1)
					ACTION_LIMIT = 1
					self.x_shift = self._p.addUserDebugParameter("X", -ACTION_LIMIT, ACTION_LIMIT, 0.0)
					self.y_shift= self._p.addUserDebugParameter("Y", -ACTION_LIMIT, ACTION_LIMIT, 0.0)

				self._p.configureDebugVisualizer(p.COV_ENABLE_GUI,GUI)
				self._p.loadSDF(os.path.join(urdfRoot,"plane_stadium.sdf"))
				self._p.setGravity(0,0,-10)

				
			


			self._p.resetBasePositionAndOrientation(self.mass, [0, 0,0.4], [0,0,0,1])
			self.reset_goal_pos()
			

			# reset mass location
			x  = self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT)
			y  = self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT)
			x_vel = 0#self.np_random.uniform(low=-1, high=1)
			y_vel = 0#self.np_random.uniform(low=-1, high=1)

			self.initalize_start_pos([x,y],[x_vel,y_vel])
			obs = self.calc_state()
			self.last_target_distance = self.calc_target_distance(obs['achieved_goal'],obs['desired_goal'])
			#self.last_velocity_distance = self.calc_velocity_distance()

			lookat = [0,0,0.1]
			distance = 7
			yaw = 0
			self._p.resetDebugVisualizerCamera(distance, yaw, -89, lookat)

			return obs

			


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




# pointMass
pointMass pybullet RL environment for simple experiments and algorithm verification.

It is a openAI gym goal based environment, and has modifiable difficulty. The easiest environment is a 2D self navigation task. Further levels of difficulty involve a modifiable number of objects which must be pushed to goal locations. 

The action space in all cases is 2D and is the positional changes to the mass in the x and y dimensions. 

In pointMassEnvObject-v0, the gold point mass must push the green block to it's goal location, the green sphere. The 'observation' is the xy positions and velocities of the mass and block respectively and the achieved and desired goal is in terms of the xy position of the block. 

<p align="center">
  <img src="https://github.com/sholtodouglas/pointMass/blob/master/images/object.gif?raw=true" alt="Object Manipulation?"/>
</p>

In pointMassEnv-v0, the gold point mass must move itself to it's goal location, the green sphere. The 'observation' is the xy positions and velocities of the mass and the achieved and desired goal is in terms of the xy position of the mass. 

<p align="center">
  <img src="https://github.com/sholtodouglas/pointMass/blob/master/images/self.gif?raw=true" alt="Object Manipulation?"/>
</p>


The env also supports my experiments into hierachial learning. Here are two examples of hierarchial sub-goal setting based on https://arxiv.org/abs/1910.11956 and https://arxiv.org/abs/1712.00948. 



<p align="center">
  <img src="https://github.com/sholtodouglas/pointMass/blob/master/images/relay.gif?raw=true" alt="Object Manipulation?"/>
</p>


<p align="center">
  <img src="https://github.com/sholtodouglas/pointMass/blob/master/images/relaylonger3.gif?raw=true" alt="Object Manipulation?"/>
</p>

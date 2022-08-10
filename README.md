# Udacity Deep Reinforcement Learning: Project 2, Continuous Control  

## Environment 
* Tennis Environment: There are two agents which control rackets to bounce a ball over a net 
* The goal of this problem is for each of the agent to keep the ball in play, and whe either of the agents hits a ball over the net they get a personal reward of +0.1, and -0.1 if it hits the floor or goes out of bounds 

* The goal of this problem is for the agent (a double jointed arm) to track a goal location, where it gives a reward of +0.1 for each timestep the hand is in the goal location 
* For the purposes of this project, the environment is considered solved when the max score of the agents' scores achieves an average (accumulated reward over an episode) of +0.5 over 100 episodes. 
* Effectively there is a single score for each episode 

## State Space: 
* 24-Dimensional continuous observation space corresponding to position and velocity of the ball and the racket (times three) because the agent gets three "frames" of this 8-tuple per observation 

## Action Space: 
* 2-Dimensional continuous space (horizontal movement -towards and away from net- and jump action), with each element bounded to be within (-1,1) 

## Installation and Usage 
* This notebook can be used by simply opening it in the provided Udacity Project Workspace 
* Alternatively, you can run this locally: 
    1. Clone this repository
    2. To get the python/conda environment setup, follow these instructions: https://github.com/udacity/deep-reinforcement-learning#dependencies
    3. Make sure you have jupyter properly installed, and you may have to get an archived version of pytorch 
    4. Download the environment for Linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip 
    5. In the repository directoy root `mv <path/to/Tennis_Linux.zip\> .`
    6. Then run `conda activate drlnd` 
    7. In your local directory, to see the trained agent in action: `python main.py "./Tennis_Linux/Tennis.x86_64"` 
    9. To train the agent yourself, use the ipynb notebook and use the Udacity Project Workspace as stated above so that you can take advantage of the GPU, hit "run all cells" 

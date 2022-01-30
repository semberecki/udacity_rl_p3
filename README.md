# udacity_rl_p3 - Tennis

## Project Details


The Unity mutli-agent Tennis environment solved with Multi-agent DDPG algorithm.

### State space and action space

```

INFO:unityagents:
'Academy' started successfully!
Unity Academy name: Academy
        Number of Brains: 1
        Number of External Brains : 1
        Lesson number : 0
        Reset Parameters :
		
Unity brain name: TennisBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 8
        Number of stacked Vector Observation: 3
        Vector Action space type: continuous
        Vector Action space size (per agent): 2
        Vector Action descriptions: , 
        
 
```
The state space contains 2 x 24 states. 
The environment is considered to be solved if one of the agent gets at least 0.5 points as an average score over 100 consecutive episodes.

```
Number of agents: 2
Size of each action: 2
There are 2 agents. Each observes a state with length: 24
The state for the first agent looks like: [ 0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.         -6.65278625 -1.5
 -0.          0.          6.83172083  6.         -0.          0.        ]
```


The agent has 2 actions that can have values from 0 to 1.


The solution is described in ```report.md``` file

## Getting Started

Install Python 3.6 and then
```
cd python && pip install -e . 
pip install protobuf==3.5.2 tensorboard==1.7 tensorboardX==1.2
unzip Tennis_Linux.zip
python -m ipykernel install --user --name drlnd --display-name "drlnd"
jupyter notebook
```


## Project sructure

1. ```maddpg_agent.py``` - contains MADDPG implementation. 
2. ```maddpg_loop.py``` - defines workflow of algorithm
3. ```main.py``` - headless run
4. ```model.py``` - defines used networks
5. ```Tennis-solution-training.ipynb``` - contains example for training procedure
6. ```Tennis-solution-training.html``` -  contains example for training procedure - html
7. ```Tennis-solution-evaluation.ipynb``` - contains highly-skilled trained agent
8. ```Tennis-solution-evaluation.html``` - contains highly-skilled trained agent - html
9. ```Tennis_Linux.zip``` - binaries version of environment
10. ```report.md``` directory with saved checkpoints.
11. ```images``` contains graphics.
12. ``` python ``` contains Unity Agents and other required packages
13. ```checkpoints_jup``` - checkpoints dir for jupyter with trained agent with 0.5 average score
14. ```checkpoints``` - - checkpoints dir for headless version with trained agent with 0.8 and 1.0 average score

![agent](images/trained-long.gif)
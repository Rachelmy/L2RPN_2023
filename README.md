# Enhancing enumeration and optimization strategies for power network efficiency

**Authors**: Shourya Bose, Qiuling Yang, Yu Zhang


We present a candidate entry for the [Delft L2RPN 2023 competition](https://codalab.lisn.upsaclay.fr/competitions/12420). The competition is built on top of the electric grid simulator [Grid2Op](https://github.com/rte-france/Grid2Op), which allows for temporal simulation of an electric grid with continuous and discrete actions, in an episodic fashion. The goal is to prevent a blackout, and while doing so, reduce the economic costs of running the grid.

Our solution is inspired by [this](https://github.com/AlibabaResearch/l2rpn-wcci-2022) and [this](https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution) solution for previous iterations of L2RPN. The main idea is as follows.

1. For each episode, simulate in open-loop (i.e. no action) for some timesteps, then purposefully disconnect a line (simulating an overload, which may lead to a blackout unless rectified soon). Then, check the list of all possible unitary discrete actions to see which is the most helpful.
2. Make records of such unitary actions and their fequencies across different episodes.
3. Select top n most frequent unitary actions to be tested during evaluation.
4. During evaluation, after finding the best unitary action from the above list, use an optimizer ([cvxpy](https://github.com/cvxpy/cvxpy)) to set continuous values by solving the DC Optimal Power Flow model.

**P.S.** By `unitary actions' we mean a small indivial action like opening/closing breakers or changing a busbar. Complex actions like configuring a line to carry majority of the power flow from a power plant to a city are made up of multiple unitary actions.

Here are some challenges.

1. The brute-force method can take a significant amount of time. Here's how the compute compounds. The total compute will have:
	* 25-100 realizations of random fast-forwarding. Let us assume 25 realizations.
	* For each random fast-forwarding, there are 1662 chronics.
	* For each chronic, there are 186 lines to disconnect.
	* For each disconnected line, there are 73141 unitary actions to enumerate.
	* Enumerating 70000 unitary action takes approx. 200 seconds.
	* So, the total time will be: 25 x 1662 x 186 x 200 = 1545660000s = **49 years**!
2. Previous solutions don't consider that some unitary actions could only be helpful for some lines, but not all.

To that end, here's our contributions:

1. We reduce the 200s part to 1-2s by considering a 'neighbor of neighbors approach'. If a line is overloaded, we only consider on substations upto a degree of separation d. For example, if line 0 has 'to' and 'from' nodes a and b, then d=2 corresponds to considering discrete actions only on substations a,b,neighbor(a),neighbor(b). 
	
	We use d=4 for our simulation.
2. We do not pool all unitary actions into a single lookup table. Rather, we save 186 pools (corresponding to 186 lines) so that we can know which lines' disconnection was a given unitary action helpful for.
3. We support MPI protocol for accelerating across line disconnections. Happy parallelizing!

After applying our solution, the train time reduces to ~**23 hours** on a machine with 2x AMD Milan CPU with 64 cores each. We use 186 workers for MPI.

## Requirements

There are many articles on the internet about how to get ```Grid2Op``` running. We simply created a virtual ```conda``` environment, followed by ```pip install lightsim2grid``` and ```pip install chronics2grid``` and it worked! 

After installing, here's what to do. The competition environment is called ```l2rpn_wcci_2022```. Run the following lines of code.

	from grid2op import make
	from lightsim2grid import LightSim2Grid
	env = make('l2rpn_wcci_2022',backend=LightSim2Grid())
	
Running the above will download about 1.7GB of chronics, and the environment is good to go! The backend is not necessary but highly recommended, since we haven't tried running our codes without it.

We use a modified version of [OptimCVXPY](https://github.com/rte-france/l2rpn-baselines/tree/master/l2rpn_baselines/OptimCVXPY) agent from  [L2RPN Baselines](https://github.com/rte-france/l2rpn-baselines) in our final submission. However, the required files are present in the submission, so installation of `l2rpn_baselines` is not needed.

## File layout

This repository contains a lot of files corresponding to the different attempts we made. But what your looking for is ```Simulations.py``` for the simulations, and the folder ```final_submission``` for our final submission. These files trace their heritage from [this](https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution) solution for L2RPN NeurIPS 2020, and basically reduce the total number of discrete unitary actions to test over during evaluation. Our turbo-charged version implements the aforementioned `neighbor of neighbors' approach and per-line recording.

You can just run it with python.
	
	python Simulations.py
	
It supports the following arguments

	--samplestep # how many steps to skip in an ep.
	--supress_int_logs # set to nonzero to make it less verbose
	--topn # limit to number of unitary actions per line (i.e. top n)
	--filename # file for each line saved as line_x_[filename], where x is line number
	--save_every # save records every .. episodes
	--deg_of_sep # how many degrees of separation you want in the `neighbor of neighbors' approach?
	--n_episodes # number of episodes
	--use_parallel # set to nonzero to use parallelism
	
Here's how to use parallelism with OpenMPI. Suppose you want to use 20 workers and have created a job which has 20 worker cores available. Run as

	mpiexec -np 20 --map-by core:PE=1 python Simulations.py --use_parallel 1
	
Once execution has completed (either in serial or parallel), you will have ```n_line``` files with format ```.npz``` (186 for ```l2rpn-wcci-2022```). They are then copied to the submission folder (```submission5```) in our case, where ```my_agent.py``` and ```optimCVXPY.py``` use them to instantiate the agent.

## Example

As a baseline, we use an agent which does not do anything when all line flows are below thermal limits, and otherwise uses an agent trained with [Soft Actor-Critic](https://arxiv.org/pdf/1812.05905.pdf) in PyTorch.

**Baseline:**
![proposed](gif/baseline.gif)
**Proposed:**
![proposed](gif/proposed.gif)

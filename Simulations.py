"""
Original author: chen binbin
mail: cbb@cbb1996.com
link: https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution/blob/master/Teacher/Teacher1.py

Modified:
Shourya Bose | Qiuling Yang | Yu Zhang
mail: shbose@ucsc.edu

How-to: Once all .npz files are generated after running this code, put the files in the same folder as the agent
and use the agent. Ensure that the number of .npz files are the same as the number of lines in the environment.
"""
import os
import time
import random
import grid2op
import numpy as np
from grid2op import make
from tqdm import tqdm, trange
import argparse
from lightsim2grid import LightSimBackend

## parser
parser = argparse.ArgumentParser(description='MODDED_TEACHER')

parser.add_argument('--samplestep',type=int,default=72,help="Offset of random fast forwarding.")
parser.add_argument('--supress_int_logs',type=int,default=1,help="Supress verbose logs.")
parser.add_argument('--topn',type=int,default=350,help="n for saving top-n unitary actions for each line.")
parser.add_argument('--filename',type=str,default='act.npz',help="Suffix of filename of lookup table for each lines unitary actions.")
parser.add_argument('--save_every',type=int,default=3,help="Number of episodes or fast forward realizations at which a save will be done.")
parser.add_argument('--deg_of_sep',type=int,default=4,help="Degree of separation from overloaded line for querying unitary actions.")
parser.add_argument('--n_episodes',type=int,default=25,help="Total number of realizations of fast forwards.")
parser.add_argument('--use_parallel',type=int,default=0,help="Indicator to use parallelization or not.")

args = parser.parse_args()

# use parallel?
if args.use_parallel:
    from mpi4py import MPI

## global vars
envname = "l2rpn_wcci_2022"
action_buffer = {} # dict to store action buffer
count_buffer = {} # dict to store counts
line_buffer = {} # dict to store endangered lines

def find_neighbors(line_id,env,deg=args.deg_of_sep):
    if deg == 0:
        return []
    nb = np.array([env.line_or_to_subid[line_id],env.line_ex_to_subid[line_id]])
    line_idx = np.arange(env.n_line)
    for _ in range(deg-1):
        collector = []
        for sub in nb:
            # check lines whose or (ex) is sub and record their ex (or)
            exlines = env.line_ex_to_subid[line_idx[env.line_or_to_subid==sub]].tolist()
            if exlines !=[]:
                collector += exlines
            orlines = env.line_or_to_subid[line_idx[env.line_ex_to_subid==sub]].tolist()
            if orlines != []:
                collector += orlines
        if collector != []:
            nb = np.concatenate((nb,np.array(collector)),axis=None)
            nb = np.unique(nb)
    return nb

def topology_search(env,line_id):
    print('Starting topology search!',flush=True)
    obs = env.get_obs()
    min_rho, overflow_id = obs.rho.max(), obs.rho.argmax()
    if not args.supress_int_logs:
        print("step-%s, line-%s(from bus-%d to bus-%d) overflows, max rho is %.5f" %
            (dst_step, overflow_id, env.line_or_to_subid[overflow_id],
            env.line_ex_to_subid[overflow_id], obs.rho.max()))
    nb_buses = find_neighbors(line_id,env)
    all_actions = []
    for buses in nb_buses:
        all_actions += env.action_space.get_all_unitary_topologies_change(env.action_space,sub_id=buses)
    action_chosen = env.action_space({})
    tick = time.time()
    for action in all_actions:
        if not env._game_rules(action, env):
            continue
        obs_, _, done, _ = obs.simulate(action)
        if (not done) and (obs_.rho.max() < min_rho):
            min_rho = obs_.rho.max()
            action_chosen = action
    print("find a greedy action and max rho decreases to %.5f, search duration: %.2f" %
          (min_rho, time.time() - tick))
    return action_chosen

def hash_action(act):
    return hash(act.to_vect().astype(int).data.tobytes())

def save_to_buffer(env,act,line):
    if not act == env.action_space({}):
        hash_act = hash_action(act)
        try:
            count_buffer[hash_act] += 1
            if line not in line_buffer[hash_act]:
                line_buffer[hash_act].append(line)
            seen = True
        except KeyError:
            action_buffer[hash_act] = act.to_vect().astype(int)
            count_buffer[hash_act] = 1
            line_buffer[hash_act] = [line]
            seen = False
        print('\n\nSaved to buffer! Action was %s.\n\n'%('seen' if seen else 'not seen'),flush=True)
            
def save_buffer_to_disk(line_id):
    if not count_buffer.keys() == []:
        sorted_keys = [k for k,v in sorted(count_buffer.items(),key=lambda x:x[1],reverse=True)]
        if len(sorted_keys) >= args.topn:
            top_acts = np.array([action_buffer[key] for key in sorted_keys[:args.topn]])
            top_counts = np.array([count_buffer[key] for key in sorted_keys[:args.topn]])
        else:
            top_acts = np.array([action_buffer[key] for key in sorted_keys])
            top_counts = np.array([count_buffer[key] for key in sorted_keys])
        if not os.path.isdir(os.getcwd()+'/save_acts'):
            os.mkdir(os.getcwd()+'/save_acts')
        np.savez(os.getcwd()+'/save_acts/line_%d_%s'%(line_id,args.filename),action_space=top_acts,counts=top_counts)
        print('\n\nSaved to disk!\n\n',flush=True)

if __name__ == "__main__":
    
    # comm for parallel ops
    if args.use_parallel:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        rank = 0
    
    # init the environment
    env = grid2op.make(envname, backend=LightSimBackend())
    
    # check workers if parallel
    if args.use_parallel:
        print('Rank %d of total %d.'%(rank,size),flush=True)
        
    # attack all lines
    LINES2ATTACK = np.arange(env.n_line)

    for line_to_disconnect in LINES2ATTACK:
        # traverse all attacks of all parallel lines
        # either its a parallel worker's turn, or parallelization is disabled altogether
        if ((line_to_disconnect+1)%(rank+1) == 0) or (not args.use_parallel): 
            
            for episode in trange(args.n_episodes):
                
                if (episode+1) % args.save_every == 0:
                    save_buffer_to_disk(line_to_disconnect)
                # shuffle up the chronics
                _ = env.chronics_handler.shuffle(shuffler=lambda x: x[np.random.choice(len(x), size=len(x), replace=False)])
                
                # traverse all chronics
                for chronic in range(env.chronics_handler._real_data.chronics_used.size): # loop across all chronics
                    
                    env.reset()
                    dst_step = episode * args.samplestep + random.randint(0, args.samplestep)  # a random sampling every 6 hours
                    
                    if not args.supress_int_logs:
                        print('\n\n' + '*' * 50 + '\nScenario[%s]: at step[%d], disconnect line-%d(from bus-%d to bus-%d]' % (
                            env.chronics_handler.get_name(), dst_step, line_to_disconnect,
                            env.line_or_to_subid[line_to_disconnect], env.line_ex_to_subid[line_to_disconnect]),flush=True)
                        
                    # to the destination time-step
                    env.fast_forward_chronics(dst_step - 1)
                    obs, reward, done, _ = env.step(env.action_space({}))
                    if done:
                        break
                    
                    # disconnect the targeted line
                    new_line_status_array = np.zeros(obs.rho.shape).astype(int)
                    new_line_status_array[line_to_disconnect] = -1
                    action = env.action_space({"set_line_status": new_line_status_array})
                    obs, reward, done, _ = env.step(action)
                    
                    if obs.rho.max() < 1:
                        # not necessary to do a dispatch
                        continue
                    else:
                        # search a greedy action
                        action = topology_search(env,line_to_disconnect)
                        obs_, reward, done, _ = env.step(action)
                        save_to_buffer(env,action,line_to_disconnect)
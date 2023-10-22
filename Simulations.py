"""
Original author: chen binbin
mail: cbb@cbb1996.com
link: https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution/blob/master/Teacher/Teacher1.py

Modified:
Shourya Bose | Qiuling Yang | Yu Zhang
mail: shbose@ucsc.edu
"""
import os
import time
import random
import grid2op
import numpy as np
from grid2op import make
from tqdm import tqdm, trange
import argparse
from hashlib import sha1
from lightsim2grid import LightSimBackend

## parser
parser = argparse.ArgumentParser(description='MODDED_TEACHER')

parser.add_argument('--envname',type=str,default="l2rpn_wcci_2022")
parser.add_argument('--samplestep',type=int,default=72)
parser.add_argument('--cap_fast_forward_at',type=int,default=1e+4)
parser.add_argument('--supress_int_logs',type=int,default=1)
parser.add_argument('--topn',type=int,default=350)
parser.add_argument('--filename',type=str,default='act.npz')
parser.add_argument('--save_every',type=int,default=2)
parser.add_argument('--deg_of_sep',type=int,default=6) 
parser.add_argument('--use_parallel',type=int,default=0)
parser.add_argument('--data_path',type=str,default=os.getcwd())
parser.add_argument('--prefix',type=str,default=os.getcwd())

args = parser.parse_args()

# use parallel?
if args.use_parallel:
    from mpi4py import MPI

## global vars
envname = args.envname
action_buffer = [] # dict to store action buffer
count_buffer = [] # dict to store counts

def find_neighbors(line_id,env,deg):
    if deg == 0:
        return []
    nb = np.array([env.line_or_to_subid[line_id],env.line_ex_to_subid[line_id]]).tolist()
    line_idx = np.arange(env.n_line)
    for _ in range(deg-1):
        nbprime = nb
        for sub in nbprime:
            # check lines whose or (ex) is sub and record their ex (or)
            exlines = env.line_ex_to_subid[line_idx[env.line_or_to_subid==sub]].tolist()
            if exlines !=[]:
                nb += exlines
            orlines = env.line_or_to_subid[line_idx[env.line_ex_to_subid==sub]].tolist()
            if orlines != []:
                nb += orlines
            nb = np.unique(nb).tolist()
    return nb

def get_all_unitary_line_topologies_change(action_space,sub_id=None):
    res = []
    if sub_id is not None:
        or_lines = np.arange(action_space.n_line)[action_space.line_or_to_subid==sub_id]
        ex_lines = np.arange(action_space.n_line)[action_space.line_ex_to_subid==sub_id]
        for line in or_lines:
            status = action_space.get_change_line_status_vect()
            status[line] = True
            res.append(action_space({"change_line_status": status}))
        for line in ex_lines:
            status = action_space.get_change_line_status_vect()
            status[line] = True
            res.append(action_space({"change_line_status": status}))
    return res

def get_all_unitary_load_topologies_change(action_space,sub_id=None):
    res = []
    if sub_id is not None:
        load_bus = np.arange(env.n_load)[env.load_to_subid==sub_id]
        for lbus in load_bus:
            res.append(action_space({"change_bus":{"loads_id":[lbus]}}))
    return res

def get_all_unitary_gen_topologies_change(action_space,sub_id=None):
    res = []
    if sub_id is not None:
        gen_bus = np.arange(env.n_gen)[env.gen_to_subid==sub_id]
        for gbus in gen_bus:
            res.append(action_space({"change_bus":{"generators_id":[gbus]}}))
    return res

def get_all_unitary_storage_topologies_change(action_space,sub_id=None):
    res = []
    if sub_id is not None:
        storage_bus = np.arange(env.n_storage)[env.storage_to_subid==sub_id]
        for sbus in storage_bus:
            res.append(action_space({"change_bus":{"storages_id":[sbus]}}))
    return res

def get_unitary_actions(env,line_id,deg):
    all_actions = []
    nb_buses = find_neighbors(line_id,env,deg)
    for buses in nb_buses:
        all_actions += get_all_unitary_line_topologies_change(env.action_space,sub_id=buses)
        all_actions += get_all_unitary_load_topologies_change(env.action_space,sub_id=buses)
        all_actions += get_all_unitary_gen_topologies_change(env.action_space,sub_id=buses)
        all_actions += get_all_unitary_storage_topologies_change(env.action_space,sub_id=buses)
    return all_actions

def topology_search(env,line_id,ep,max_deg=args.deg_of_sep):
    #print('Starting topology search!',flush=True)
    obs = env.get_obs()
    min_rho, overflow_id = obs.rho.max(), obs.rho.argmax()
    if not args.supress_int_logs:
        print("step-%s, line-%s(from bus-%d to bus-%d) overflows, max rho is %.5f" %
            (dst_step, overflow_id, env.line_or_to_subid[overflow_id],
            env.line_ex_to_subid[overflow_id], obs.rho.max()))
    all_actions = get_unitary_actions(env,line_id,max_deg)
    action_chosen = env.action_space({})
    tick = time.time()
    for action in all_actions:
        if not env._game_rules(action, env):
            continue
        obs_, _, done, _ = obs.simulate(action)
        if (not done) and (obs_.rho.max() < min_rho):
            min_rho = obs_.rho.max()
            action_chosen = action
    if not args.supress_int_logs:
        print("On episode %d and line %d,max rho decreases to %.5f, search duration: %.2f." %
            (ep,line_id,min_rho, time.time() - tick))
    return action_chosen

def hash_action(act):
    return sha1(act.to_vect().astype(int).data.tobytes()).hexdigest()

def save_to_buffer(env,act,line,ep,ch,ff):
    if not act == env.action_space({}):
        hash_act = hash_action(act)
        try:
            count_buffer[line][hash_act] += 1
            seen = True
        except KeyError:
            action_buffer[line][hash_act] = act.to_vect().astype(int)
            count_buffer[line][hash_act] = 1
            seen = False
        print('Episode %d, Line %d, Chronic %d, Fastforward %d: Action with hash %s %s buffer! Action was %s.'
              %(ep+1,line+1,ch+1,ff,str(hash_act),'saved to' if not seen else 'updated in',
                'seen' if seen else 'not seen'),flush=True)
            
def save_buffer_to_disk(env,line,ep,prefix,top):
    if not os.path.isdir(prefix+'/save_acts'):
        os.makedirs(prefix+'/save_acts')
    if action_buffer[line].keys() != []:
        sorted_keys = [k for k,_ in sorted(count_buffer[line].items(),key=lambda x:x[1],reverse=True)]
        if len(sorted_keys) >= top:
            top_acts = np.array([action_buffer[line][key] for key in sorted_keys[:top]])
            top_counts = np.array([count_buffer[line][key] for key in sorted_keys[:top]])
        else:
            top_acts = np.array([action_buffer[line][key] for key in sorted_keys])
            top_counts = np.array([count_buffer[line][key] for key in sorted_keys])
        np.savez(prefix+'/save_acts/line_%d_%s'%(line,args.filename),action_space=top_acts,counts=top_counts,episode=ep)
        print('\n\nSaved to disk!\n\n',flush=True)
    else:
        print('XXXXX-----XXXXX---WARNING: Found no legal options for line %d on episode %d in buffer.'%(line,ep))
        print('Saving empty action!')
        top_acts = np.array([env.action({}).to_vect().astype(int)])
        top_counts = np.array([0])
        np.savez(prefix+'/save_acts/ep%d/line_%d_%s'%(ep,line,args.filename),action_space=top_acts,counts=top_counts)
        
def split_given_size(a, size):
    return np.split(a, np.arange(size,len(a),size))

if __name__ == "__main__":
    
    # comm for parallel ops
    if args.use_parallel:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        rank = 0
        size = 1
    
    # check workers if parallel
    if args.use_parallel:
        print('Rank %d of total %d.'%(rank,size),flush=True)
        
    # init the environment and ensure path of downloaded chronics
    if grid2op.MakeEnv.get_current_local_dir() != args.data_path:
        grid2op.MakeEnv.change_local_dir(args.data_path)
    env = grid2op.make(envname, backend=LightSimBackend())
    
    # attack all lines
    LINES2ATTACK = np.arange(env.n_line)
    
    # make space for all lines in dict
    action_buffer = [{} for _ in range(env.n_line)]
    count_buffer = [{} for _ in range(env.n_line)]
    
    # number of episodes
    n_episodes = int(np.ceil(int(np.min([env.max_episode_duration(),args.cap_fast_forward_at]))/args.samplestep))
    if rank == 0:
        print('Depending on values of --samplestep and --cap_fast_forward_at, running %d episodes.'%n_episodes,flush=True)
    
    # split episodes and lines for parallelization
    line_splits = np.array_split(LINES2ATTACK,size)
    
    for episode in range(n_episodes):
            
        for idx_line_split,line_split in enumerate(line_splits):
            
            if rank == idx_line_split or not args.use_parallel:

                for line_to_disconnect in line_split:
                    
                    # shuffle up the chronics
                    _ = env.chronics_handler.shuffle(shuffler=lambda x: x[np.arange(len(x))])
                    
                    # traverse all chronics
                    for chronic in range(env.chronics_handler._real_data.chronics_used.size): # loop across all chronics
                        
                        dst_step = np.min([episode * args.samplestep + random.randint(0, args.samplestep), env.max_episode_duration()-1]).item()  # a random sampling every 6 hours
                        
                        if not args.supress_int_logs:
                            print('\n\n' + '*' * 50 + '\nScenario[%s]: at step[%d], disconnect line-%d(from bus-%d to bus-%d]' % (
                                env.chronics_handler.get_name(), dst_step, line_to_disconnect,
                                env.line_or_to_subid[line_to_disconnect], env.line_ex_to_subid[line_to_disconnect]),flush=True)
                            
                        # to the destination time-step
                        _ = env.reset()
                        env.fast_forward_chronics(dst_step - 1)
                        if not env.done:
                            # episode survived fast forwarding; now disconnect line
                            obs = env.get_obs()
                            new_line_status_array = np.zeros(obs.rho.shape).astype(int)
                            new_line_status_array[line_to_disconnect] = -1
                            action = env.action_space({"set_line_status": new_line_status_array})
                            obs, reward, done, _ = env.step(action)
                            if done:
                                # disconnecting the line killed the episode
                                continue
                        else:
                            # episode died during fast forwarding
                            continue
                        
                        if obs.rho.max() < 1:
                            # not necessary to do a dispatch
                            continue
                        else:
                            # search a greedy action
                            action = topology_search(env,line_to_disconnect,episode)
                            obs_, reward, done, _ = env.step(action)
                            retval = save_to_buffer(env,action,line_to_disconnect,episode,chronic,dst_step)
                        
                    # save to disk    
                    if (episode+1) % args.save_every == 0:
                        # truncated (used for agents)
                        save_buffer_to_disk(env,line_to_disconnect,episode,args.prefix,args.topn)
                        # full (for your records) - 1e+6 is a large number
                        save_buffer_to_disk(env,line_to_disconnect,episode,args.prefix+'/full',1e+6) 
                            
                            
        # # synchronization point for parallel
        # if args.use_parallel:
        #     comm.Barrier()
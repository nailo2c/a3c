import torch
import torch.nn.functional as F
from torch.autograd import Variable

from envs import create_atari_env
from a3c import ActorCritic

import time

from gym import wrappers


# 從shared_model拉參數下來，看看目前model學得如何
def monitor(rank, args, shared_model):
    
    env = create_atari_env(args.env_name)
    env = wrappers.Monitor(env, './video/pong-a3c', video_callable=lambda count: count % 30 == 0, force=True)
    
    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    
    # eval mode
    model.eval()
    
    # init
    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    episode_length = 0
    done = True
    start_time = time.time()
    
    while True:
        env.render()
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = Variable(torch.zeros(1, 256), volatile=True) # lstm's param
            hx = Variable(torch.zeros(1, 256), volatile=True) # lstm's param
        else:
            cx = Variable(cx.data, volatile=True)
            hx = Variable(hx.data, volatile=True)
            
        # unsqueeze(0)後tensor的size會從1x42x42 -> 1x1x42x42
        value, logit, (hx, cx) = model((Variable(state.unsqueeze(0), volatile=True), (hx, cx)))
        prob = F.softmax(logit)
        # 直接選機率最大的動作
        action = prob.max(1)[1].data.numpy()
        
        state, reward, done, _ = env.step(action[0][0])
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward
        
        if done:
            print("Time {}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                reward_sum, episode_length))
            # reset
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()
            time.sleep(60)
            
        state = torch.from_numpy(state)

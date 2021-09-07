#!/usr/bin/env python3

import gym
from griddly import GymWrapper, gd
from griddly.RenderTools import VideoRecorder
import matplotlib.pyplot as plt


def test_griddly():
    env = gym.make('GDY-Labyrinth-v0')
    env.reset()

    for s in range(10):
        obs, reward, done, info = env.step(env.action_space.sample())

        if done:
            env.reset()

def test_custom_env():
    env = GymWrapper(yaml_file='simple_zelda.yaml', level=0,
        global_observer_type=gd.ObserverType.SPRITE_2D, player_observer_type=gd.ObserverType.SPRITE_2D)
    env.reset()

    for s in range(1000):
        #obs, reward, done, info = env.step(env.action_space.sample())
        env.render()
        # if done:
        #     env.reset()

def test_zelda_env():
    env = GymWrapper(yaml_file='simple_zelda.yaml')
    env.reset()
    total_reward = 0
    for s in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward = total_reward + reward
        if done:
            env.reset()
    print(total_reward)
    return total_reward

def test_video():
    video_recorder = VideoRecorder()

    env = GymWrapper(yaml_file='simple_zelda.yaml')
    env.reset()
    obs = env.render()
    video_recorder.start("video/random_agent.mp4", obs.shape)
    total_reward = 0
    
    for s in range(500):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward = reward + total_reward
        image = env.render()
        video_recorder.add_frame(image)

        if done:
            env.reset()

    print(total_reward)
    video_recorder.close()


#test_custom_env()
#test_video()

# results = []
# for i in range(200):
#     results.append(test_zelda_env())
# print(sum(results)/200)
# plt.plot(results)
# plt.show()


test_custom_env()
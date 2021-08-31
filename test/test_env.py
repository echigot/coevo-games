#!/usr/bin/env python3

import gym
from griddly import GymWrapper, gd
from griddly.RenderTools import VideoRecorder


def test_griddly():
    env = gym.make('GDY-Labyrinth-v0')
    env.reset()

    for s in range(10):
        obs, reward, done, info = env.step(env.action_space.sample())

        if done:
            env.reset()

def test_custom_env():
    env = GymWrapper(yaml_file='simple_zelda.yaml', level=1,
        global_observer_type=gd.ObserverType.SPRITE_2D, player_observer_type=gd.ObserverType.SPRITE_2D)
    env.reset()

    for s in range(10):
        obs, reward, done, info = env.step(env.action_space.sample())
        if done:
            env.reset()

def test_zelda_env():
    env = GymWrapper(yaml_file='simple_zelda.yaml')
    env.reset()
    
    for s in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            env.reset()


def test_video():
    video_recorder = VideoRecorder()

    env = GymWrapper(yaml_file='simple_zelda.yaml')
    env.reset()
    obs = env.render()
    video_recorder.start("video_test.mp4", obs.shape)
    
    for s in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        image = env.render()
        video_recorder.add_frame(image)

        if done:
            env.reset()

    
    video_recorder.close()
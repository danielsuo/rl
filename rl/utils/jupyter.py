import base64
import glob
import io
import os

from gym import wrappers
import IPython
import pyvirtualdisplay
import matplotlib.pyplot as plt

curr_env = None
img = None
vdisplay = pyvirtualdisplay.Display(visible=0, size=(1400, 900))
vdisplay.start()


def render(env, width=9, height=9):
    global curr_env
    global img

    data = env.render(mode="rgb_array")
    if curr_env != env:
        plt.figure(figsize=(width, height))
        img = plt.imshow(data)
        curr_env = env
    else:
        img.set_data(data)
        IPython.display.display(plt.gcf())
        IPython.display.clear_output(wait=True)


class DifferentiableMonitor(wrappers.Monitor):
    def step(self, action, **kwargs):
        self._before_step(action)
        observation, reward, done, info = self.env.step(action, **kwargs)
        done = self._after_step(observation, reward, done, info)

        return observation, reward, done, info


def monitor(env, path):
    return DifferentiableMonitor(env, path, force=True)


def video(env, path, width=360, height="auto"):
    if os.path.isdir(path):
        paths = glob.glob(os.path.join(path, "*.mp4"))
        maxsize = 0
        for video in paths:
            if os.path.getsize(video) > maxsize:
                path = video

    data = io.open(path, "r+b").read()
    data = base64.b64encode(data).decode("ascii")

    return IPython.display.HTML(
        data=f"""
    <video width="{width}" height="{height}" alt="test" controls><source src="data:video/mp4;base64,{data}" type="video/mp4" /></video>"""
    )

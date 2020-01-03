
# coding: utf-8

# # 项目：指导四轴飞行器学会飞行
# 
# 设计一个能够使四轴飞行器飞行的智能体，然后使用你选择的强化学习算法训练它！
# 
# 请尝试运用你在这一单元中学到的知识，看看哪个方法效果最好，当然你也可以自己想出创新型方法并测试它们。
# ## 说明
# 
# 请查看目录下的文件，以更好地了解项目结构。 
# 
# - `task.py`：在本文件中定义你的任务（环境）。
# - `agents/`：本文件夹中包含强化学习智能体。
#     - `policy_search.py`：我们为你提供了一个智能体模板。
#     - `agent.py`：在本文件中开发你的智能体。
# - `physics_sim.py`：本文件中包含四轴飞行器模拟器。**请勿修改本文件**。
# 
# 在本项目中，你需要在 `task.py` 中定义你的任务。尽管我们为你提供了一个任务示例，来帮助你开始项目，但你也可以随意更改这个文件。在这个 notebook 中，你还将学习更多有关修改这个文件的知识。
# 
# 你还需要在 `agent.py` 中设计一个强化学习智能体，来完成你选择的任务。
# 
# 我们也鼓励你创建其他文件，来帮助你整理代码。比如，你也许可以通过定义一个 `model.py` 文件来定义其他你需要的神经网络结构。
# 
# ## 控制四轴飞行器
# 
# 在下方的代码中，我们提供了一个智能体示例，来示范如何使用模拟器来控制四轴飞行器。这个智能体比你在 notebook 中需要测试的智能体（在 `agents/policy_search.py` 中）更加简单！
# 
# 这个智能体通过设置飞行器四个轴上的转速来控制飞行器。`Basic_Agent` 类中提供的智能体将会随机为四个轴指定动作。这四个速度将通过 `act` 方法以四个浮点数列表的形式返回。
# 
# 在本项目中，你将在 `agents/agent.py` 中实现的智能体会以更加智能的方法进行指定的动作。

# In[1]:


import random

class Basic_Agent():
    def __init__(self, task):
        self.task = task
    
    def act(self):
        new_thrust = random.gauss(450., 25.)
        return [new_thrust + random.gauss(0., 1.) for x in range(4)]


# 运行下方代码，让智能体指定动作来控制四轴飞行器。
# 
# 请随意更改我们提供的 `runtime`，`init_pose`，`init_velocities` 和 `init_angle_velocities` 值来更改四轴飞行器的初始条件。
# 
# 下方的 `labels` 列表为模拟数据的注释。所有的信息都储存在 `data.txt` 文档中，并保存在 `results` 目录下。

# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import csv
import numpy as np
from task import Task

# Modify the values below to give the quadcopter a different starting position.
runtime = 5.                                     # time limit of the episode
init_pose = np.array([0., 0., 10., 0., 0., 0.])  # initial pose
init_velocities = np.array([0., 0., 0.])         # initial velocities
init_angle_velocities = np.array([0., 0., 0.])   # initial angle velocities
file_output = 'data.txt'                         # file name for saved results

# Setup
task = Task(init_pose, init_velocities, init_angle_velocities, runtime)
agent = Basic_Agent(task)
done = False
labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
          'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
          'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4']
results = {x : [] for x in labels}

# Run the simulation, and save the results.
with open(file_output, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(labels)
    while True:
        rotor_speeds = agent.act()
        _, _, done = task.step(rotor_speeds)
        to_write = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(rotor_speeds)
        for ii in range(len(labels)):
            results[labels[ii]].append(to_write[ii])
        writer.writerow(to_write)
        if done:
            break


# 在进行模拟时，请运行下方的代码来查看四轴飞行器的位置变化。

# In[3]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(results['time'], results['x'], label='x')
plt.plot(results['time'], results['y'], label='y')
plt.plot(results['time'], results['z'], label='z')
plt.legend()
_ = plt.ylim()


# In[4]:


import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure(figsize=(10, 10))
ax = fig.gca(projection='3d')

# Prepare arrays x, y, z
z = results['z']
x = results['x']
y = results['y']

# Draw trajectory
ax.plot(x, y, z, label='Trajectory')
# Add start position
ax.scatter(x[0], y[0], z[0], c='g', label='Start')
ax.text(x[0], y[0], z[0], '(%.1f, %.1f, %.1f)' % (x[0], y[0], z[0]))
# Add final position
ax.scatter(x[-1],y[-1],z[-1], marker='x', c='r', label='Final')
ax.text(x[-1], y[-1], z[-1], '(%.1f, %.1f, %.1f)' % (x[-1], y[-1], z[-1]))

ax.legend()

plt.show()


# 下方代码输出了四轴飞行器的速度。

# In[5]:


plt.plot(results['time'], results['x_velocity'], label='x_hat')
plt.plot(results['time'], results['y_velocity'], label='y_hat')
plt.plot(results['time'], results['z_velocity'], label='z_hat')
plt.legend()
_ = plt.ylim()


# 接下来，你可以绘制欧拉角 (Euler angles)（四轴飞行器围绕 $x$-，$y$- 和 $z$-轴的旋转）的图表，

# In[6]:


plt.plot(results['time'], results['phi'], label='phi')
plt.plot(results['time'], results['theta'], label='theta')
plt.plot(results['time'], results['psi'], label='psi')
plt.legend()
_ = plt.ylim()


# 在绘制每个欧拉角的速度（每秒的弧度）图之前。

# In[7]:


plt.plot(results['time'], results['phi_velocity'], label='phi_velocity')
plt.plot(results['time'], results['theta_velocity'], label='theta_velocity')
plt.plot(results['time'], results['psi_velocity'], label='psi_velocity')
plt.legend()
_ = plt.ylim()


# 最后，你可以使用下方代码来输出智能体选择的动作。

# In[8]:


plt.plot(results['time'], results['rotor_speed1'], label='Rotor 1 revolutions / second')
plt.plot(results['time'], results['rotor_speed2'], label='Rotor 2 revolutions / second')
plt.plot(results['time'], results['rotor_speed3'], label='Rotor 3 revolutions / second')
plt.plot(results['time'], results['rotor_speed4'], label='Rotor 4 revolutions / second')
plt.legend()
_ = plt.ylim()


# 在指定任务之前，你需要在模拟器中衍生环境状态。运行下方代码来在模拟结束时输出以下变量值：
# 
# - `task.sim.pose`：四周飞行器在 ($x,y,z$) 坐标系中的位置和欧拉角。
# - `task.sim.v`：四轴飞行器在 ($x,y,z$) 坐标系中的速度。
# - `task.sim.angular_v`：三个欧拉角的弧度/每秒。

# In[9]:


# the pose, velocity, and angular velocity of the quadcopter at the end of the episode
print(task.sim.pose)
print(task.sim.v)
print(task.sim.angular_v)


# 在 `task.py` 中的任务示例中，我们使用了四轴飞行器六个维度的动作来构建每个时间步的环境状态。然而，你也可以按照自己的意愿更改任务，你可以添加速度信息来扩大状态向量，也可以使用任何动作、速度和角速度的组合，并构造适用于你的任务的环境状态。
# 
# ## 任务
# 
# 在 `task.py` 中，我们为你提供了一个任务示例。请在新窗口中打开这个文件。
# 
# 使用 `__init__()` 方法来初始化指定本任务所需的几个变量。
# 
# - 模拟器作为 `PhysicsSim` 类（来自 `physics_sim.py` 文件）的示例进行初始化。
# - 受到 DDPG 论文中研究方法的启发，我们使用了重复调用动作的方法。对于智能体的每一个时间步，我们将利用 `action_repeats` 时间步来进行模拟。如果你并不熟悉这种方法，可以阅读 [DDPG 论文](https://arxiv.org/abs/1509.02971)的结论部分。
# - 我们设置了状态向量中每个分量的数值。在任务示例中，我们只设置了六个维度的动作信息。为了设定向量大小（`state_size`），我们必须考虑重复的动作。
# - 任务环境通常是一个四维动作空间，每个轴有一个输入（`action_size=4`）。你可以设置每个输入的最小值（`action_low`）和最大值（`action_high`）。
# - 我们在文件中提供的任务示例将使智能体达到目标位置。我们将目标位置设置为一个变量。
# 
# `reset()` 方法将重置模拟器。每当阶段结束时，智能体都将调用此方法。你可以查看下方代码中的例子。
# 
# `step()` 方法是最重要的一个方法。它将接收智能体选择的动作 `rotor_speeds`，并准备好下一个状态，同时返回给智能体。接着，你将通过 `get_reward()` 计算奖励值。当超过规定时间，或是四轴飞行器到达模拟器边缘时，这一阶段将视作结束。
# 
# 接下来，你将学习如何测试这个任务中智能体的性能。
# 
# ## 智能体
# 
# `agents/policy_search.py` 文件中提供的智能体示例使用了非常简单的线性策略，将动作向量视作状态向量和矩阵权重的点积直接进行计算。接着，它通过添加一些高斯噪声来随机干扰参数，以产生不同的策略。根据每个阶段获得的平均奖励值（`score`），它将记录迄今为止发现的最佳参数集以及分数的变化状态，并据此调整比例因子来扩大或减少噪音。
# 
# 请运行下方代码来查看任务示例中智能体的性能。

# In[10]:


import sys
import pandas as pd
from policy_search import PolicySearch_Agent
from task import Task

num_episodes = 1000
target_pos = np.array([0., 0., 10.])
task = Task(target_pos=target_pos)
agent = PolicySearch_Agent(task) 

for i_episode in range(1, num_episodes+1):
    state = agent.reset_episode() # start a new episode
    while True:
        action = agent.act(state) 
        next_state, reward, done = task.step(action)
        agent.step(reward, done)
        state = next_state
        if done:
            print("\rEpisode = {:4d}, score = {:7.3f} (best = {:7.3f}), noise_scale = {}".format(
                i_episode, agent.score, agent.best_score, agent.noise_scale), end="")  # [debug]
            break
    sys.stdout.flush()


# 这个智能体的性能想必十分糟糕！现在轮到你出场了！
# 
# ## 定义任务，设计并训练你的智能体！
# 
# 修改 `task.py` 文件来指定你所选择的任务。如果你不确定选择什么任务，你可以教你的四轴飞行器起飞、悬停、着陆或是达到指定位置。
# 
# 
# 在指定任务后，使用 `agents/policy_search.py` 中的智能体示例作为模板，来在 `agents/agent.py` 中定义你自己的智能体。你可以随意从智能体示例中借用你需要的元素，包括如何模块化你的代码（使用 `act()`，`learn()` 和 `reset_episode_vars()` 等辅助方法）。
# 
# 请注意，你指定的第一个智能体和任务**极有可能**无法顺利进行学习。你将需要改进不同的超参数和奖励函数，直到你能够获得不错的结果。
# 
# 在开发智能体的时候，你还需要关注它的性能。参考下方代码，建立一个机制来存储每个阶段的总奖励值。如果阶段奖励值在逐渐上升，说明你的智能体正在学习。

# In[25]:


## TODO: Train your agent here.
import keras
import sys
import pandas as pd
from agent import DDPG
from task import Task

num_episodes = 2000
target_pos = np.array([0., 0., 100.])
init_pose=np.array([0.,0.,0.,0.,0.,0.])
task = Task(target_pos=target_pos,init_pose=init_pose)
agent = DDPG(task)

reward_labels = ['episode', 'reward']
reward_results = {x : [] for x in reward_labels}


# In[18]:


for i_episode in range(1, num_episodes+1):
    state = agent.reset_episode() # start a new episode
    while True:
        action = agent.act(state) 
        next_state, reward, done = task.step(action)
        agent.step(action, reward, next_state, done)
        state = next_state
        if done:
            print("\rEpisode = {:4d}, score = {:7.3f} (best = {:7.3f})".format(
                i_episode, agent.score, agent.best_score), end="")  # [debug]
            break
    reward_results['episode'].append(i_episode)
    reward_results['reward'].append(agent.score)
    sys.stdout.flush()


# 下一段代码参考了 [amihalik 的 Github](https://github.com/amihalik/RL-Quadcopter-2/blob/master/Quadcopter_Project.ipynb)。

# In[19]:


#Run the Simulation with the latest results
done = False
labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
          'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
          'psi_velocity', 'rotor_speed1', 'reward']
results = {x : [] for x in labels}


state = agent.reset_episode() # start a new episode
while True:
    action = agent.act(state) 
    next_state, reward, done = task.step(action*np.ones(4))
    agent.step(action, reward, next_state, done)
    state = next_state

    to_write = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(action) + list([reward])
    for ii in range(len(labels)):
        results[labels[ii]].append(to_write[ii])
    if done:
        break


# In[20]:


import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure(figsize=(10, 10))
ax = fig.gca(projection='3d')

# Prepare arrays x, y, z
z = results['z']
x = results['x']
y = results['y']

# Draw trajectory
ax.plot(x, y, z, label='Trajectory')
# Add start position
ax.scatter(x[0], y[0], z[0], c='g', label='Start')
ax.text(x[0], y[0], z[0], '(%.1f, %.1f, %.1f)' % (x[0], y[0], z[0]))
# Add final position
ax.scatter(x[-1],y[-1],z[-1], marker='x', c='r', label='Final')
ax.text(x[-1], y[-1], z[-1], '(%.1f, %.1f, %.1f)' % (x[-1], y[-1], z[-1]))
ax.legend()

plt.show()


# In[21]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(16, 8))
plt.plot(results['time'], results['x'], label='x')
plt.plot(results['time'], results['y'], label='y')
plt.plot(results['time'], results['z'], label='z')
plt.legend()
_ = plt.ylim()


# ## 绘制阶段奖励
# 
# 请绘制智能体在每个阶段中获得的总奖励，这可以是单次运行的奖励值，也可以是多次运行的平均值。

# In[22]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(16, 8))
plt.plot(reward_results['episode'], reward_results['reward'], label='reward/episode')
plt.legend()
_ = plt.ylim()


# In[24]:


# Final Performance 
print("Final Performance (Mean Reward over last 10 episodes): {}".format(np.sum(results['reward'][-10:])/10))


# ## 回顾
# 
# **问题 1**：请描述你在 `task.py` 中指定的任务。你如何设计奖励函数？
# 
# **回答**：因为时间原因，我只将目标设置为起飞；
# 任务设定如下：
# ```py
# target_pos = np.array([0., 0., 100.])
# init_pose=np.array([0.,0.,0.,0.,0.,0.])
# task = Task(target_pos=target_pos,init_pose=init_pose)
# ```
# 奖励函数如下所示：
# ```py
# reward_xy = np.tanh(1.-.009*(abs(self.sim.pose[:2] - self.target_pos[:2]))).sum()
# reward_z = np.tanh(1.-.003*(abs(self.sim.pose[2] - self.target_pos[2]))).sum()
# reward = reward_xy + reward_z
# ```
# 为了保证无人机直线上升而不偏离 z 轴，将 reward 设置为 x 和 y 轴方向偏离时惩罚较多，而沿着 z 轴上升时奖励较多。
# 
# **问题 2**：请简要描述你的智能体，你可以参考以下问题：
# 
# - 你尝试了哪些学习算法？哪个效果最好？
# - 你最终选择了哪些超参数（比如 $\alpha$，$\gamma$，$\epsilon$ 等）？
# - 你使用了什么样的神经网络结构（如果有的话）？请说明层数、大小和激活函数等信息。
# 
# **回答**：  
# - 1.在这个项目里，我仅仅尝试使用了项目建议的 DDPG 算法。在最初的尝试中，我忘记将目标的位置设置为与初始位置不同了，导致最后结果虽然看上去正确，但是实际上并未实现真正的起飞任务。于是调整了目标的位置，并尝试调整了正则化系数。感谢老师指出解决方案！
# - 2.最终参数的选择如下：
#     - Soft update of target parameters $\tau$ = 0.01
#     - 折扣率 $\gamma$ = 0.99
#     - Actor 的学习速率为 0.0001
#     - Critic 的学习速率为 0.001
# - 3.最终的神经网络结构如下：
# 
#     - Critic 设置（`critic.py`）：
#         - For state pathway:
#         ```py
#         state_layer1: states -> Dense(512, regularizers.l2(1e-6)) -> BatchNormalization -> relu
#         ```
#         - For action pathway:
#         ```py
#         state_layer2: state_layer1 -> Dense(256, regularizers.l2(1e-6)) -> relu
#         action_layer1: actions -> Dense(256, regularizers.l2(1e-6))-> relu
#         ```
#         - Combine state and action pathways:
#         ```py
#         net: [net_states, net_actions] -> Add -> relu    
#         ```
#     
#         - Final output Q values:
#         ```py
#         Q_values: net -> Dense(1, initializers.RandomUniform(-0.003, 0.003))
#         ```
#         - Optimizer:
#         ```py
#         optimizers.Adam(lr=0.001)
#         ```
#     - Actor 设置（`actor.py`）：
#     ```py
#     layer1: state -> Dense(512, regularizers.l2(1e-6)) -> BatchNormalization -> relu
#     layer2: layer1 -> Dense(256, regularizers.l2(1e-6)) -> BatchNormalization -> relu
#     layer3: layer2 -> Dense(action_size, initializers.RandomUniform(-0.003, 0.003))
#     optimizers.Adam(lr=0.0001)
#     ```
# 
# **问题 3**：根据你绘制的奖励图，描述智能体的学习状况。
# 
# - 学习该任务是简单还是困难？
# - 该学习曲线中是否存在循序渐进或急速上升的部分？
# - 该智能体的最终性能有多好？（比如最后十个阶段的平均奖励值）
# 
# **回答**：
# - 我认为本学习任务相对来说比较简单，只是单纯的起飞任务，但又有一些限制，比如尽量减少对 z 轴的偏离；
# - 二者都存在，在最初阶段，智能体可能在做一些随意的尝试，并没有获得很好的分数，而之后逐渐学会往目标方向飞行，分数也随之增长。1250出有一个急速攀升过程，1250之后基本保持一个稳定的分数。
# - 我们可以从图形中可以看到，在经历一个急剧上升的阶段之后，分数趋于平稳并接近于训练过程中的最好成绩，最终的性能尚可。最后十次的分数均值为：899.9577799301151
# 
# **问题 4**：请简要总结你的本次项目经历。你可以参考以下问题：
# 
# - 本次项目中最困难的部分是什么？（例如开始项目、运行 ROS、绘制、特定的任务等。）
# - 关于四轴飞行器和你的智能体的行为，你是否有一些有趣的发现？
# 
# **回答**：
# - 对我来说最困难的部分应该是理解连续控制任务这一概念；另外从最开始上手拿到这个任务有点懵，以往的项目都已经在项目中把所有的模块都给出来了，只是需要自己去修改一些部分和网络结构，这次项目需要的几个比较重要的模块都在课程网站上，并不在项目文件夹中，需要自己去新建文件。一开始并没有完全理解算法的同学可能会感到无从下手。

# ### (可选)Plot Actor 及 Critic 结构
# 建议使用 ```from keras.utils import plot_model``` 来显示模型结构；

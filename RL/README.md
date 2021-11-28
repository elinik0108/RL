# AI-2, Assignment 2 - Reinforcement Learning
In this assignment, you will learn to solve simple reinforcement learning problems.
The assignment is split into two parts.
In Part 1, you have to improve a naive multi-armed bandit implementation.
In Part 2, you will implement a Q-learning agent that plays the *Pong* game.
Both parts must be covered by one report and one submission.
You will submit two distinct implementations, one for each part.

**You can (and are encouraged to) work in pairs**.

## Requirements
You need to have the following software installed and accounts set up to solve this exercise:

* [Python 3.7.x or higher](https://www.python.org/);

* You can use [git](https://git-scm.com/) to check out the repository; this is not required, but you will need git for dependency installation purposes, see below.

* A text editor or development environment like [Visual Studio Code](https://code.visualstudio.com/) or [PyCharm](https://www.jetbrains.com/pycharm/).

We recommend using a Unix-based operating system (Mac OS, Linux) or Windows with a bash emulator for this exercise.
If you don't know ``Python``, search online for some tutorials.
If you don't know ``git``, consider learning it, but you'll be able to solve the assignment without git knowledge.

## Checking out the Code
**Note that we will have an online "lab session" to get you started with the assignment.**
**Attending the session will help you solve the assignment quickly and accurately.**

Open the GitLab repository we use for this exercise: [https://git.cs.umu.se/courses/5dv181ht21.git](https://git.cs.umu.se/courses/5dv181ht21.git).

Download (click the **Download** button) or ``clone`` the repository:

```
git@git.cs.umu.se:courses/5dv181ht21.git
```

If you have already cloned the repository, you can run ``git pull --rebase`` to get the latest changes.
In case you have made changes to the repository, you need to stash them beforehand (``git stash``); you can apply them afterwards with ``git stash apply``.


## Multi-Armed Bandits
Multi-armed bandits are a simple reinforcement learning method that forms the foundation of many large-scale recommender systems.

To get an overview of multi-armed bandits in their implementation take a look at:

* ["Towards Data Science" introduction to multi-armed bandits](https://towardsdatascience.com/solving-multiarmed-bandits-a-comparison-of-epsilon-greedy-and-thompson-sampling-d97167ca9a50)

* [Academic overview paper](https://arxiv.org/pdf/1402.6028)

In this part of the assignment, you will implement a multi-armed bandit to solve an abstract example problem.
Your bandit will need to beat a “naïve” benchmark.

### Getting Started

Navigate into the ``assignments/Assignment_2_RL/bandit`` folder in the project you have downloaded.
Install the dependencies for this part of the assignment (instead of using the ``user`` flag, you can consider setting up a fresh [virtual environment](https://docs.python.org/3/tutorial/venv.html)):

```
pip install --user -r requirements.txt
```

**Please do not make use of additional dependencies.**

Open the ``MyBandit.py`` file in the project's root directory. You will see an implementation of a simple epsilon-greedy bandit.
Your task is to improve the bandit and so that you can beat the initial bandit's performance reliably.
Your improvements must be algorithmically and not just parameter tuning.

**Success criterion:** Out of 20 simulation runs with 1.000 +/-500 "pulled arms" each, your new  bandit should outperform the reference bandit by at least 35% (35% more reward gained) in at least 16 runs.
**Note that the rewards per arm will be adjusted after each of the 20 simulation runs; i.e., your bandit must be capable of adjusting to these changes.**

To test your implementation, run ``pytest`` in the repository's root directory.
Note that you are working in a stochastic world.
Still, a reasonable implementation should make the test pass.
Conversely, it may be possible to 'hack' a solution that makes the tests but does not provide any generally applicable improvements.
Don't do this, we'll check carefully enough!

### Report Section
Once you have achieved satisfactory performance, don't hesitate to improve further ;-), but more importantly, write a short report section that describes:

1. how you proceeded, *i.e.* describe your solution approach conceptually and precisely;

2. what your results are, *i.e.* provide a quantification and optionally visualization of your results;

3. how you could improve further, *i.e.* describe and motivate potential improvements.

This report section should be approximately one page long; not much shorter, not much longer.

## Q-Learning for Pong
This part of the assignment is about Q-learning.
In particular, the task is to implement a Q-learning agent that plays *Pong*, see: [https://github.com/koulanurag/ma-gym/wiki/Environments#pongduel](https://github.com/koulanurag/ma-gym/wiki/Environments#pongduel).
Note that you will install a slightly customized version of the pong environment, i.e.: [https://github.com/TimKam/ma-gym](https://github.com/TimKam/ma-gym)

There are plenty of Q-learning tutorials out there, for example [this one](https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/).
We do not guarantee that the provided information is academically precise, but it will probably help you get the idea.

### Getting Started
Make sure you have git [installed and configured](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).
You will need it for dependency installation (not for version control).
Navigate into the ``assignments/Assignment_2_RL/pong`` folder in the project you have downloaded.

**Important:** If you work on Windows, install [ffmpeg](https://windowsloop.com/install-ffmpeg-windows-10/).

Install the dependencies for this part of the assignment:

```
pip install --user -r requirements.txt
```

**Please do not make use of additional dependencies.**
Exceptions are the optional stretch tasks.

Open the ``Agent.py`` file in the project's root directory.
Here, you find the skeleton of the ``Agent`` class you need to implement.
When running ``pong.py``, i.e. the simulation environment, you will see that your agent can already interact with the environment.
However, it does not learn!
More precisely, while the agent already keeps track of the performance of its actions (it creates a Q-table), the data is not used to decide on the action that is to be executed in a given state.
By implementing the methods sketched out in ``Agent.py``, your task is to make the agent learn and become a decent pong player.
**Write an agent that uses basic Q-learning (not deep Q-learning) and that learns fast enough to have a `#wins - #losses` performance of more than 1000 after around 350 episodes (an episode consists of 20 games).**
Record a GIF that shows how your agent performs around this stage, see this example:

![pong.gif](pong.gif)

Note that to achieve the expected performance, you _may_ want to write a script that tries out different parameters (gamma, epsilon, et cetera, but also different extents of state simplification) and logs the number of wins after 350 episodes for every parameter combination.
The script is merely a helper and you do not need to submit it.

As stretch tasks, you may want to implement a planner that achieves the same objective, or a Deep learner.
If so, please still submit a copy of the basic Q-learner.
Note that the stretch tasks are entirely optional and may take a lot of time to complete.

### Report Section
Analogously to the first part of the assignment, document in the report:

1. how you proceeded, *i.e.* describe your solution approach conceptually and precisely;

2. what your results are, *i.e.* provide a quantification and optionally visualization of your results;

3. how you could improve further, *i.e.* describe and motivate potential improvements.

This report section should be approximately one page long; not much shorter, not much longer.


## Hand-in

Hand-in the report, the GIF, and a copy of your code in [Labres](https://webapps.cs.umu.se/labresults/v2/courseadmin.php?courseid=458).
The report must have a title page including your name, your username at computing science (of both students if you work in pair), the course name, course code, and the assignment name.

The only program code you need to hand in are the ``MyBandit.py`` and ``Agent.py`` files.
**Do not hand in a .zip file.**
**Also, ensure your implementations only depend on changes made to ``MyBandit.py`` and ``Agent.py``.**

Upon hand-in, the test (``pytest``) will automatically run to check if your ``MyBandit.py`` submission fulfills some technical requirements.
Please check to make sure the test passes. If not, you can re-submit.
Still, please note that the automatic test merely checks if the performance of your multi-armed bandit has improved.
A passing test is no guarantee that your code is actually good enough (but it is a strong indicator).



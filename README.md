![alt text](https://raw.githubusercontent.com/5Volts/Reinforcement-Learning-For-Cartpole/master/img.png)
# Reinforcement Learning For Cartpole

This is a simple reinforcement learning program to ace the Cartpole challenge. An A.I. will be trained to balance a cartpole by moving the cart left and right.

# Dependencies

Ensure that you have the following modules installed:
`tensorflow`
`gym`
`numpy`

If not just type `pip install tensorflow gym numpy` into terminal/commandline.

# Get started

After downloading, cd into the directory and type `python RL_for_cartpole.py`, it will start training the A.I. for 100 episodes.

There are severals variables you can pass:

'--watch_it_train'
A boolean, put it as True if you want to watch the AI train.
Ex: `python RL_for_cartpole.py --watch_it_train True`                      

'--explore_proba'
Explore probability. Default is set to 0.98 .Bear in mind that putting it too low might cause the model to never converge.

'--train_step'
Number of Training steps                      

'--test_episodes'
Number of Testing episodes                      

'--select_top_best'
Select from top K examples
                      
'--sample'
Number of generated samples per training steps


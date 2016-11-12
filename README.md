# DNC-theano : A theano implementation of DeepMind's DNC model

### What's this

This is a naive implementation of DeepMind's Differentiable Neural Computer model.

### Requirements
 - python3
 - numpy
 - theano

### Usage

Clone this repository first. Then run `dnc.py` script in interactive mode.

**NOTE: If you don't run in interactive mode it will just quit silently.**

    $ python3 -i dnc.py
    Compiling ... done
    >>>

Compilation in theano is a bit slow so it may take a while. Once compiled, you can just call python functions defined in `dnc.py`. The default training task is the conditional copy task, the network must produce input signal in same or reversed order, depending on the first "condition" signal. You may want to modify the code to introduce new tasks.

Most useful functionalities are shown below:

    >>> train() #train a new model, model is saved in "params.pkl" every 100 iters
    Iter 100/10000 loss: 0.040833
    Iter 200/10000 loss: 0.034119
    Iter 300/10000 loss: 0.030109
    Iter 400/10000 loss: 0.028633
    ...
    Iter 10000/10000 loss: 0.004308
    >>> g_optimizer.lr = 5e-4 #change learning rate
    >>> train(4,6) #customize training plan
    Iter 6/24 loss: 0.004296
    ...
    Iter 24/24 loss: 0.004265
    >>> save_params() #save model parameters manually
    >>> exit()
    $ python3 -i dnc.py #go back
    >>> load_params() #load previously saved model parameters
    >>> X,Y = gen_episode() #generate a test data point
    >>> fn_rst() #reset DNC states
    >>> Y_pred = predict(X) #make a prediction with DNC
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(Y_pred+X); plt.show() #make a visualization of DNC prediction

### Bugs/Issues
 - 'nan' crash on rare occasion

import numpy as np
import theano as th
T = th.tensor

'''Common optimizers'''

class Optimizer(object):
    '''
    Abstract class for all optimizers
    '''
    pass

class ParamOptimizer(Optimizer):
    '''
    Abstract class for all optimizer that only optimizes parameters of specific model
    '''
    def __init__(self):
        pass
    def fit(self, v_y_, *v_x_):
        pass

class SGDOptimizer(ParamOptimizer):
    '''
    class for all SGD based optimizer

    Attribs:
        timestep:
            Records current timestep in training, starts from 1.
            Can be manually set to any non-negative integer.
            May have an impact on certain optimization algorithm

        lr:
            global learn rate, can be set to any float number
    '''
    def __init__(self, lr_=1e-3):
        '''
        Setups hyperparameter for optimizer, but not compiling against specifit model

        Args:
            lr_: learn rate
        '''
        self.lr = lr_
        self.s_lr = T.fscalar('learn_rate')
        self.timestep = 0
        if type(self) == SGDOptimizer:
            raise RuntimeError('This class shouldn\'t be instantiated')

    def compile(
        self,s_inputs_, s_loss_, v_params_, s_grads_=None, s_reg_=0,
        fetches_=None, updates_=None, givens_=None,
        trunc_grad_=None
    ):
        '''
        compile optimizer against specific model

        Args:
            s_inputs_: list of symbolic input tensors, including label
            s_loss_: optimization loss, symbolic scalar
            v_params_: list of shared parameters to optimize
            s_grads: list of gradients to apply, must be same order as v_params_, default is None (use autodiff).
            s_reg_: symbolic regularization term, default 0 (no regularization)
            updates: update operation for shared values after a step of optimization,
                usually RNN states. Takes form [(v_var, s_new_var), ...]

        Returns: None
        '''
        self.s_loss = s_loss_
        self.s_reg = s_reg_
        if s_grads_ is None:
            s_grads_ = T.grad(
                self.s_loss + self.s_reg, list(v_params_))

        if type(trunc_grad_)==float:
            self.s_grads = [T.clip(g,-trunc_grad_,trunc_grad_) for g in s_grads_]
        else:
            self.s_grads = s_grads_

    def fit(self,*v_x_li_):
        '''
        Perform an iteration of SGD

        Args:
            *v_x_li_: sequence of inputs

        Returns:
            same format as "fetches_" argument pass to compile()
        '''
        self.timestep += 1
        return self.fn_train(self.lr,*v_x_li_)


class VanillaSGD(SGDOptimizer):
    '''
    Just calculates gradients and applys.

    Guaranteed no suspicious memory overhead.
    Training efficiency not guaranteed.
    '''
    def __init__(self, lr_=1e-3):
        super(VanillaSGD,self).__init__(lr_)

    def compile(self,s_inputs_, s_loss_, v_params_, s_grads_=None, s_reg_=0, fetches_=None, updates_=None, givens_=None, trunc_grad_=None):
        if type(s_inputs_) not in (list, tuple):
            s_inputs_ = [s_inputs_]
        super(VanillaSGD,self).compile(
            s_inputs_, s_loss_, v_params_, s_reg_=s_reg_, s_grads_=s_grads, trunc_grad_=trunc_grad_)
        apply_grad = [(p, p-g*self.s_lr) for p,g in zip( v_params_,self.s_grads)]
        self.fn_train = th.function(
            [self.s_lr]+s_inputs_,
            fetches_,
            updates=apply_grad+(updates_ if updates_ else []),
            givens=givens_,
            on_unused_input='warn'
        )
        return self.fn_train


class AdamSGD(SGDOptimizer):
    def __init__(self, lr_=1e-3, beta_=(0.9,0.999), eps_=1e-6):
        super(AdamSGD,self).__init__(lr_,)
        self.beta = beta_
        self.eps = eps_

    def compile(self,s_inputs_, s_loss_, v_params_, s_grads_=None, s_reg_=0, fetches_=None, updates_=None, givens_=None, trunc_grad_=None):
        def get_shared_shape(v):
            return v.get_value(borrow=True, return_internal_type=True).shape
        if type(s_inputs_) not in (list, tuple):
            s_inputs_ = [s_inputs_]
        super(AdamSGD,self).compile(
            s_inputs_, s_loss_, v_params_, s_reg_=s_reg_, s_grads_=s_grads_, trunc_grad_=trunc_grad_)
        v_m = [th.shared(value=np.zeros(get_shared_shape(p), th.config.floatX)) for p in v_params_]
        v_v = [th.shared(value=np.zeros(get_shared_shape(p), th.config.floatX)) for p in v_params_]
        s_b1 = T.scalar()
        s_b2 = T.scalar()
        s_b1s = T.scalar()
        s_b2s = T.scalar()
        update_m = [(m, (m*s_b1 + (1.-s_b1)*g)) for m,g in zip(v_m,self.s_grads)]
        update_v = [(v, (v*s_b2 + (1.-s_b2)*g*g)) for v,g in zip(v_v,self.s_grads)]
        apply_grad = [(p, p-(s_b1s*m*self.s_lr)/(T.sqrt(s_b2s*v)+self.eps)) for p,m,v in zip(v_params_,v_m,v_v)]
        self.fn_train = th.function(
            [self.s_lr]+s_inputs_+[s_b1,s_b2,s_b1s,s_b2s],
            fetches_,
            updates=update_m+update_v+apply_grad+(updates_ if updates_ else []),
            givens=givens_)
        return self.fn_train

    def fit(self, *v_x_li_):
        self.timestep += 1
        b1,b2 = self.beta
        b1s = 1./(1.-b1**self.timestep)
        b2s = 1./(1.-b2**self.timestep)
        return self.fn_train(self.lr,*v_x_li_,b1,b2,b1s,b2s)

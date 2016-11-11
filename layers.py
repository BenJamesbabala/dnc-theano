from time import time
from math import sqrt, pi

from six.moves import cPickle as pickle

import numpy as np
import theano as th
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

_default_params = {}
_g_params_di = _default_params

def set_current_params(di_):
    global _g_params_di
    _g_params_di = di_

def get_current_params():
    return _g_params_di

def save_params(p_, f_):
    #FIXME: this does not save shared_variable properties like "strict" or "allow_downcast"
    pickle.dump({k:v.get_value() for k,v in p_.items()}, f_)

def load_params(p_, f_):
    di = pickle.load(f_)
    for k,v in di.items():
        p_[k].set_value(v)

def get_variable(name_, shape_, init_range_=None, dtype_=th.config.floatX):
    '''
    get a shared tensor variable with name_, return existing one if exists, otherwise create a new one
    behaves like tf.get_variable(.)

    Args:
        name_: name of the variable
        shape_: tensor shape
        init_range_:
            when creating a new shared var, initialize uniformly. None->zeros constant->constant init tuple->uniform_distribution
    '''
    global _g_params_di
    if name_ in _g_params_di:
        #TODO: add shape/dtype check?
        return _g_params_di[name_]
    if init_range_ is None:
        v = th.shared(
            np.zeros(shape_,dtype=dtype_),
            name=name_
        )
    elif type(init_range_) in [list,tuple]:
        v = th.shared(
            np.asarray(np.random.uniform(
                *init_range_,
                size=shape_),
                dtype=dtype_),
            name=name_
        )
    else:
        v = th.shared(
            np.full(shape_, init_range_, dtype=dtype_),
            name=name_)
    _g_params_di[name_] = v
    return v

g_rng = RandomStreams(seed=int(time()*100)%(2**32))

def op_dropout(s_x_, s_p_):
    return s_x_ * g_rng.binomial(n=1, p=1.-s_p_, size=T.shape(s_x_), dtype=th.config.floatX)

def lyr_conv(name_, s_x_, idim_, odim_, fsize_=3, init_scale_ = None):
    global _g_params_di
    name_conv_W = '%s_w'%name_
    name_conv_B = '%s_b'%name_
    ir = 1.4/sqrt(idim_*fsize_*fsize_+odim_) if init_scale_ is None else init_scale_
    v_conv_W = get_variable(name_conv_W, (odim_,idim_,fsize_,fsize_),(-ir,ir))
    v_conv_B = get_variable(name_conv_B, (odim_))
    return T.nnet.conv2d(
        s_x_, v_conv_W,
        filter_shape=(odim_, idim_, fsize_, fsize_),
        border_mode = 'half'
    )+v_conv_B.dimshuffle('x',0,'x','x')

def lyr_linear(name_, s_x_, idim_, odim_, init_scale_=None, bias_=0.):
    global _g_params_di
    name_W = name_+'_w'
    name_B = name_+'_b'
    ir = 1.4/sqrt(idim_+odim_) if init_scale_ is None else init_scale_
    v_W = get_variable(name_W, (idim_,odim_), (-ir,ir))
    if bias_ is None:
        s_ret = T.dot(s_x_, v_W)
    else:
        v_B = get_variable(name_B, (odim_,), bias_)
        s_ret = T.dot(s_x_, v_W) + v_B
    if s_x_.ndim == 1:
        return s_ret.flatten()
    else:
        return s_ret

def lyr_gru(
    name_,
    s_x_, s_state_,
    idim_, hdim_,
    axis_=0,
    lyr_linear_=lyr_linear,
    op_act_=T.tanh,
    op_gate_=T.nnet.sigmoid):
    global _g_params_di
    s_inp = T.join(axis_, s_x_, s_state_)
    s_igate = lyr_linear_(name_+'_igate', idim_+hdim_, idim_)
    s_inp_gated = T.join(axis_, s_x_ * op_gate_(s_igate), s_state_)
    s_gate_lin, s_state_tp1_lin = T.split(lyr_linear_(name_+'_gate', idim_+hdim_, hdim_*2), [hdim_,hdim_], 2, axis_)
    s_gate = op_gate_(s_gate_lin)
    return s_state_*s_gate + op_act_(s_state_tp1_lin)*(1.-s_gate)

def lyr_lstm(
    name_,
    s_x_, s_cell_, s_hid_,
    idim_, hdim_,
    axis_=-1,
    lyr_linear_=lyr_linear,
    op_act_=T.tanh,
    op_gate_=T.nnet.sigmoid):
    global _g_params_di
    s_inp = T.join(axis_, s_x_, s_hid_)
    s_gates_lin, s_inp_lin = T.split(
        lyr_linear_(name_+'_rec', s_inp, idim_+hdim_, hdim_*4),
        [hdim_*3,hdim_], 2, axis=axis_)
    s_igate, s_fgate, s_ogate = T.split(op_gate_(s_gates_lin), [hdim_]*3, 3, axis=axis_)
    s_cell_tp1 = s_igate*op_act_(s_inp_lin) + s_fgate*s_cell_
    s_hid_tp1 = op_act_(s_cell_tp1)*s_ogate
    return s_cell_tp1, s_hid_tp1

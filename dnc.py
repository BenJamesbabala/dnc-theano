import sys
from random import randint

import numpy as np
import theano as th
import theano.tensor as T
from theano.scan_module import until

import layers as lyr
from optimize import AdamSGD, VanillaSGD

th.config.floatX = 'float32'

#hyperparameters are in CAPS
SEQ_LEN = 32
INP_DIMS = 8
OUT_DIMS = 8
N_CELLS = 32 #number of memory cells
CELL_SIZE = 64 #size of each memory cell
N_READS = 4 #number of read heads
B_SIZE = 32 #unused for now
LR = 1e-3 #learn rate
EPS = 1e-6 #to avoid division by zero

g_params = {}
g_states = {}
g_optimizer = AdamSGD()
g_optimizer.lr = LR
fn_predict = None

#temporal hack for theano issue #5197 at github
#TODO: revert to normal implementation once the issue is fixed
if th.config.device[:3] == 'cpu':
    def op_cumprod_hack(s_x__, axis_=None):
        return T.extra_ops.cumprod(s_x_*0.99+0.01, axis=axis_)
else:
    def op_cumprod_hack(s_x_, axis_=None):
        #due to cumprod has only CPU implementation
        return T.exp(T.extra_ops.cumsum(T.log(s_x_*0.99+0.01), axis=axis_))

def build_model():
    global g_params, g_states, g_optimizer, fn_predict, fn_rst
    ctrl_inp_size = CELL_SIZE*N_READS+INP_DIMS
    itrface_size = CELL_SIZE*N_READS+3*CELL_SIZE+5*N_READS+3
    ctrl_wm_size = OUT_DIMS+itrface_size

    #states
    lyr.set_current_params(g_states)
    v_lstm_cell = lyr.get_variable('lstm_cell',(ctrl_wm_size,))
    v_lstm_hid = lyr.get_variable('lstm_hid',(ctrl_wm_size,))
    v_usage = lyr.get_variable('usage',(N_CELLS,))
    v_preced = lyr.get_variable('preced',(N_CELLS,))
    v_link = lyr.get_variable('link',(N_CELLS,N_CELLS))
    v_mem = lyr.get_variable('mem',(N_CELLS, CELL_SIZE))
    v_read_val = lyr.get_variable('r_val',(N_READS,CELL_SIZE))
    v_read_wgt = lyr.get_variable('r_wgt',(N_READS,N_CELLS))
    v_write_wgt = lyr.get_variable('w_wgt',(N_CELLS,))

    #build the actual model
    lyr.set_current_params(g_params)
    def dnc_step(
        s_x_,
        s_lstm_cell_,
        s_lstm_hid_,
        s_usage_,
        s_preced_,
        s_link_,
        s_mem_,
        s_read_val_,
        s_read_wgt_,
        s_write_wgt_):
        s_states_li_ = [
            s_lstm_cell_,
            s_lstm_hid_,
            s_usage_,
            s_preced_,
            s_link_,
            s_mem_,
            s_read_val_,
            s_read_wgt_,
            s_write_wgt_]
        s_inp = T.join(-1, s_x_, s_read_val_.flatten())

        s_lstm_cell_tp1, s_lstm_hid_tp1 = lyr.lyr_lstm(
            'ctrl',
            s_inp, s_lstm_cell_, s_lstm_hid_,
            ctrl_inp_size, ctrl_wm_size
        )
        s_out, s_itrface = T.split(
            lyr.lyr_linear(
                'ctrl_out', s_lstm_hid_tp1, ctrl_wm_size, ctrl_wm_size, bias_=None),
            [OUT_DIMS,itrface_size],2, axis=-1)
        splits_len = [
            N_READS*CELL_SIZE, N_READS, CELL_SIZE, 1,
            CELL_SIZE, CELL_SIZE, N_READS, 1, 1, 3*N_READS
        ]
        s_keyr, s_strr, s_keyw, s_strw, \
            s_ers, s_write, s_freeg, s_allocg, s_writeg, s_rmode = \
            T.split(s_itrface, splits_len, 10, axis=-1)

        s_keyr = T.reshape(s_keyr, (CELL_SIZE,N_READS))
        s_strr = 1.+T.nnet.softplus(s_strr)
        s_strw = 1.+T.nnet.softplus(s_strw[0])
        s_ers = T.nnet.sigmoid(s_ers)
        s_freeg = T.nnet.sigmoid(s_freeg)
        s_allocg = T.nnet.sigmoid(s_allocg[0])
        s_writeg = T.nnet.sigmoid(s_writeg[0])
        s_rmode = T.nnet.softmax(T.reshape(s_rmode,(N_READS,3))).dimshuffle(1,0,'x')

        s_mem_retention = T.prod(
            1.-s_freeg.dimshuffle(0,'x')*s_read_wgt_, axis=0)

        s_usage_tp1 = s_mem_retention*(
            s_usage_+s_write_wgt_-s_usage_*s_write_wgt_)
        s_usage_order = T.argsort(s_usage_tp1)
        s_usage_order_inv = T.inverse_permutation(s_usage_order)
        s_usage_tp1_sorted = s_usage_tp1[s_usage_order]

        s_alloc_wgt = ((1.-s_usage_tp1_sorted)*(
            T.join(
                0,np.array([1.],dtype=th.config.floatX),
                op_cumprod_hack(s_usage_tp1_sorted[:-1])
            )))[s_usage_order_inv]

        s_content_wgt_w = T.nnet.softmax(
            s_strw*T.dot(s_mem_, s_keyw)/(
                T.sqrt(
                    EPS+T.sum(T.sqr(s_mem_),axis=-1)*T.sum(T.sqr(s_keyw))))
        ).flatten()

        s_write_wgt_tp1 = s_writeg*(
            s_allocg*s_alloc_wgt+(1.-s_allocg)*s_content_wgt_w)

        s_mem_tp1 = s_mem_*(
            1.-T.outer(s_write_wgt_tp1,s_ers))+T.outer(s_write_wgt_tp1,s_write)
        s_preced_tp1 = (1.-T.sum(s_write_wgt_))*s_preced_ + s_write_wgt_tp1

        s_link_tp1 = (
            1.-s_write_wgt_tp1-s_write_wgt_tp1.dimshuffle(0,'x')
        )*s_link_ + T.outer(s_write_wgt_tp1,s_preced_)
        s_link_tp1 = s_link_tp1 * (1.-T.identity_like(s_link_tp1))#X
        s_fwd = T.dot(s_read_wgt_, s_link_tp1.transpose())#X
        s_bwd = T.dot(s_read_wgt_, s_link_tp1)#X

        s_content_wgt_r= T.nnet.softmax(T.dot(s_mem_tp1, s_keyr)/(T.sqrt(
            EPS+T.outer(
                T.sum(T.sqr(s_mem_tp1),axis=-1),T.sum(T.sqr(s_keyr),axis=0)
            )))).transpose()
        s_read_wgt_tp1 = s_bwd*s_rmode[0]+s_content_wgt_r*s_rmode[1]+s_fwd*s_rmode[2]
        s_read_val_tp1 = T.dot(s_read_wgt_tp1, s_mem_tp1)

        s_y = s_out + lyr.lyr_linear(
            'read_out',
            s_read_val_tp1.flatten(),
            CELL_SIZE*N_READS,OUT_DIMS,
            bias_=None)
        return [
            s_y,
            s_lstm_cell_tp1,
            s_lstm_hid_tp1,
            s_usage_tp1,
            s_preced_tp1,
            s_link_tp1,
            s_mem_tp1,
            s_read_val_tp1,
            s_read_wgt_tp1,
            s_write_wgt_tp1]

    s_x_li = T.matrix()
    s_y_target_li = T.matrix()
    v_states_li = [
            v_lstm_cell,
            v_lstm_hid,
            v_usage,
            v_preced,
            v_link,
            v_mem,
            v_read_val,
            v_read_wgt,
            v_write_wgt
    ]

    s_outputs_li, _ = th.scan(
        dnc_step,
        sequences=[s_x_li],
        outputs_info=[None]+v_states_li
    )

    s_y_li = s_outputs_li[0]

    s_loss = T.mean(T.sqr(s_y_li - s_y_target_li))

    new_states = [s[-1] for s in s_outputs_li[1:]]
    print('Compiling ... ', end='')
    sys.stdout.flush()
    g_optimizer.compile(
        [s_x_li, s_y_target_li],
        s_loss,
        list(g_params.values()),
        updates_=list(zip(v_states_li, new_states)),
        fetches_= s_loss
    )

    fn_predict = th.function([s_x_li], s_y_li)
    fn_rst = th.function([], updates=[(v,T.zeros_like(v)) for v in v_states_li])
    print('Done')

def predict(v_x_):
    return fn_predict(v_x_)

def reset_states():
    fn_rst()

def gen_episode(lenr_ =(2,4)):
    global SEQ_LEN, INP_DIMS, OUT_DIMS
    X = np.zeros((SEQ_LEN,INP_DIMS),dtype=th.config.floatX)
    Y = np.zeros((SEQ_LEN,OUT_DIMS),dtype=th.config.floatX)
    data_len = min(randint(*lenr_),(SEQ_LEN-3)//2)
    tot_len = data_len*2+4
    offset = randint(0,SEQ_LEN-1-tot_len)
    m = np.random.binomial(1,0.5,(data_len,INP_DIMS))
    coinflip = randint(0,1)
    if coinflip:
        X[offset:offset+1,INP_DIMS//2:] = 1.
        X[1+offset:1+offset+data_len] = m
        Y[offset+data_len+4:offset+4+data_len*2] = m
    else:
        X[offset:offset+1,:INP_DIMS//2] = 1.
        X[1+offset:1+offset+data_len] = m
        Y[offset+data_len+4:offset+4+data_len*2] = m[::-1]
    return X, Y

def save_params():
    global g_params, g_states
    with open('params.pkl','wb') as f:
        lyr.save_params(g_params, f)
        lyr.save_params(g_states, f)

def load_params():
    global g_params, g_states
    with open('params.pkl','rb') as f:
        lyr.load_params(g_params, f)
        lyr.load_params(g_states, f)

def train(nsess_=100, nitr_=100, lenr_=(2,4)):
    '''
    Trains model for some sessions, parameters are automatically saved to file after each session.

    Args:
        nsess_: number of training episodes.
        nitr_: number of iteration.
        lenr_: signal length range of copy task, must be tuple.
    '''
    try:
        for i in range(nsess_):
            loss = 0.
            for j in range(nitr_):
                X,Y = gen_episode(lenr_)
                loss += g_optimizer.fit(X,Y)
                if str(loss)=='nan': break
                sys.stdout.write('.')
                sys.stdout.flush()
            else:
                loss /= nitr_
                print('\nIter %d/%d loss: %f'%((i+1)*nitr_,nsess_*nitr_,loss))
                save_params()
                continue
            break
        else:
            print('Training finished.')
            return
        print('\nIter %d/%d loss: ???'%(i*nitr_+j,nsess_*nitr_))
        print('ERROR: Model crashed, stop.')
    except KeyboardInterrupt:
        print('User hit CTRL-C, stop.')

build_model()

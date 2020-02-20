import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
class_names=['aeroplane', 'ship', 'box']
class_ratios=4690./np.array([297,306,248,215,376,362,290,229,
                       353,259,324,505,510,339,798,257,4690,486,
                       514,1250],np.float32)#计算loss时，跟据每个类的类别不同，对不同类的loss进行均衡
class_id_dict=dict(zip(class_names,range(1,len(class_names)+1)))
id_class_dict=dict(zip(range(1,len(class_names)+1),class_names))
def reshape_list(data,shape=None):
    result=[]
    if shape is None:
        for a in data:
            if isinstance(a,(list,tuple)):
                result+=list(a)
            else:
                result.append(a)
    else:
        i = 0
        for s in shape:
            if s == 1:
                result.append(data[i])
            else:
                result.append(data[i:i + s])
            i += s
    return  result

def smooth_L1(x):
    abs_x=tf.abs(x)
    result=tf.where(abs_x<1,tf.square(x)*0.5,abs_x-0.5)
    return result

def tensor_shape(x,rank=4):
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape=x.get_shape().with_rank(rank).as_list()
        dynamic_shape=tf.unstack(tf.shape(x),rank)
        return [s if s is not None else d for s,d in zip(static_shape,dynamic_shape)]

def get_variables_to_restore(scope_to_include, suffix_to_exclude):
    """to parse which var to include and which
    var to exclude"""
    vars_to_include = []
    for scope in scope_to_include:
        vars_to_include += slim.get_variables(scope)

    vars_to_exclude = set()
    for scope in suffix_to_exclude:
        vars_to_exclude |= set(
            slim.get_variables_by_suffix(scope))

    return [v for v in vars_to_include if v not in vars_to_exclude]


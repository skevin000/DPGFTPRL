import logging
import os
import tensorflow as tf


def setup_tf():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logging.root.removeHandler(logging.getLogger('absl').handlers[0])

class EasyDict(dict):
    """Dictionary with attribute-style access."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

def get_fn(general_params, privacy_params, paramss, find_next=False):
    """Generate output file name based on parameters."""
    base_dir = '_'.join([f"{k}{v}" for k, v in general_params.items()]) + '/'
    
    privacy_fn = 'nonpriv' if not privacy_params.dpsgd else '_'.join(
        [f"{k}{v}" for k in sorted(privacy_params) if k not in ['dpsgd', 'completion'] and privacy_params[k]]
    )
    if privacy_params.completion:
        privacy_fn += '_completion'

    params_fn = '_'.join(
        '_'.join([k if isinstance(v, bool) else f"{k}{v}" for k, v in sorted(params.items()) if v]) 
        for params in paramss
    )
    
    fn = f"{base_dir}{privacy_fn}_{params_fn}".rstrip('_')

    if find_next:
        i = 1
        while os.path.exists(f"{fn}_{i}"):
            i += 1
        return f"{fn}_{i}"
    
    return fn

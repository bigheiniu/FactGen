#!/usr/bin/env python
"""Train models."""
import os
import glob
import numpy as np
import signal
import torch

import onmt.opts as opts
import onmt.utils.distributed

from onmt.utils.logging import logger
from onmt.train_single import main as single_main
from onmt.utils.parse import ArgumentParser


def main(opt):
    ArgumentParser.validate_train_opts(opt)
    ArgumentParser.update_model_opts(opt)
    ArgumentParser.validate_model_opts(opt)

    nb_gpu = len(opt.gpu_ranks)

    if opt.world_size > 1:
        mp = torch.multiprocessing.get_context('spawn')
        # Create a thread to listen for errors in the child processes.
        error_queue = mp.SimpleQueue()
        error_handler = ErrorHandler(error_queue)
        # Train with multiprocessing.
        procs = []
        for device_id in range(nb_gpu):
            procs.append(mp.Process(target=run, args=(
                opt, device_id, error_queue, ), daemon=True))
            procs[device_id].start()
            logger.info(" Starting process pid: %d  " % procs[device_id].pid)
            error_handler.add_child(procs[device_id].pid)
        for p in procs:
            p.join()

    elif nb_gpu == 1:  # case 1 GPU only
        single_main(opt, 0)
    else:   # case only CPU
        single_main(opt, -1)


def run(opt, device_id, error_queue):
    """ run process """
    try:
        gpu_rank = onmt.utils.distributed.multi_init(opt, device_id)
        if gpu_rank != opt.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")
        single_main(opt, device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((opt.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def _get_parser():
    parser = ArgumentParser(description='train.py')

    opts.config_opts(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    
    if opt.config is None or opt.run_name is None:
        raise ValueError('base config and run_name must be set during training')
    
    config_name = opt.config.split('/')[-1]
    config_name = ''.join(config_name.split('.')[:-1])

    dataset_name = opt.data.split('/')[-2]+'_'+opt.data.split('/')[-1]
    output_dir = 'output/'+dataset_name+'/'+config_name+'_'+opt.run_name+'/'
    os.makedirs(output_dir, exist_ok=True)
    if opt.save_model == 'model':
        setattr(opt, 'save_model', output_dir+'checkpoints/model')
    setattr(opt, 'save_config', output_dir+'config.yml')
    setattr(opt, 'tensorboard_log_dir', 'output/'+dataset_name+'/tblogs/'+config_name+'_'+opt.run_name)
    parser.write_config_file(opt, [output_dir+'config.yml'])

    if opt.autorestart:
        filenames = []
        step_nums = []
        for filename in glob.glob(output_dir+'checkpoints/*.pt'):
            filenames.append(filename)
            step_num = os.path.basename(filename).split('_')[-1][:-3]
            step_nums.append(int(step_num))
        
        if len(filenames) > 0:
            indices = np.argsort(step_nums)
            filenames = np.array(filenames)[indices]
            
            opt.train_from = filenames[-1]
            opt.gpt2_init_embanddec = False
            opt.encoder_from = None
            opt.gpt2_params_path = None

    main(opt)

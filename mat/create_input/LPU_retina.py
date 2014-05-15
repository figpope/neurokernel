#!/usr/bin/env python

"""
Local Processing Unit (LPU) draft implementation.
"""

import collections

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import numpy as np
import networkx as nx

# Work around bug that causes networkx to choke on GEXF files with boolean
# attributes that contain the strings 'True' or 'False'
# (bug already observed in https://github.com/networkx/networkx/pull/971)
nx.readwrite.gexf.GEXF.convert_bool = {'false':False, 'False':False,
                                        'true':True, 'True':True}

from neurokernel.core import Module
import neurokernel.base as base

from types import *
from collections import Counter

from utils.simpleio import *
import utils.parray as parray
from neurons import *
from synapses import *
import LPU

import pdb

class LPU_retina(LPU.LPU, object):
    def __init__(self, dt, n_dict, s_dict, input_file=None, device=0,
                 output_file=None, port_ctrl=base.PORT_CTRL,
                 port_data=base.PORT_DATA, id=None, debug=False):
        super(LPU_retina, self).__init__(dt, n_dict, s_dict, input_file = input_file, 
                                         device = device, output_file = output_file,
                                         port_ctrl=port_ctrl, port_data=port_data,
                                         id=id, debug = debug)

        
    def pre_run(self):
        super(LPU_retina, self).pre_run()
        assert self.I_ext.shape[1] == self.num_input
        
        # note that self.photon_input contains all the input in each step.
        self.photon_input = garray.empty(self.num_input, np.double)
        

    def _read_external_input(self):
        if not self.input_eof or self.frame_count<self.frames_in_buffer:
            cuda.memcpy_dtod(int(self.photon_input.gpudata), \
                             int(int(self.I_ext.gpudata) + self.frame_count*self.I_ext.ld*self.I_ext.dtype.itemsize),
                             self.num_input * self.synapse_state.dtype.itemsize)
            self.frame_count += 1
        else:
            self.logger.info('Input end of file reached. Subsequent behaviour is undefined.')
        if self.frame_count >= self._one_time_import and not self.input_eof:
            input_ld = self.input_h5file.root.array.shape[0]
            if input_ld - self.file_pointer < self._one_time_import:
                h_ext = self.input_h5file.root.array.read(self.file_pointer, input_ld)
            else:
                h_ext = self.input_h5file.root.array.read(self.file_pointer, self.file_pointer + self._one_time_import)
            if h_ext.shape[0] == self.I_ext.shape[0]:
                self.I_ext.set(h_ext)
                self.file_pointer += self._one_time_import
                self.frame_count = 0
            else:
                pad_shape = list(h_ext.shape)
                self.frames_in_buffer = h_ext.shape[0]
                pad_shape[0] = self._one_time_import - h_ext.shape[0]
                h_ext = np.concatenate(h_ext, np.zeros(pad_shape), axis=0)
                self.I_ext.set(h_ext)
                self.file_pointer = input_ld
                
            if self.file_pointer == self.input_h5file.root.array.shape[0]:
                self.input_eof = True
                    


import numpy as np
from tools import read_acquisitionData
import re #reordering part


def get_reordered_unsplit_measures(spectral_data_file, meta_data_file):
    
    # Get spectral data
    raw_data = np.load(spectral_data_file, allow_pickle=True)["spectral_data"]
    P, L = raw_data.shape

    # Get pattern order
    meta = read_acquisitionData(meta_data_file)
    acquisition_parameters = meta['acquisition_params']
    
    
    wavelengths = acquisition_parameters.wavelengths # list of wavelengths measure
    N = acquisition_parameters.pattern_dimension_x  # probably np.sqrt(P//2) for uncompressed sensing
    s = acquisition_parameters.patterns_wp # order of acquisition patterns - some acquisitions taken in non crono order

    print("reordering and unsplitting raw measures...")

    #reordering & unsplit
    order = acquisition_parameters.patterns # order of acquisition patterns
    ind = np.array(order) # corresponding acquisition rows

    m_reordered = np.zeros((P, L)) # create space first
    m_reordered[ind, :]  = raw_data # place rows of m_unsplit following order of ind :)
    m_unsplit = m_reordered[0::2, :] - m_reordered[1::2, :]


    ind_hadamard = np.array(ind[0::2])//2
    
    print("raw_data: ", raw_data.shape)
    print("m_reordered: ", m_reordered.shape)
    print("m_unsplit: ", m_unsplit.shape)
    
    print("done.")
    
    return m_unsplit, ind_hadamard
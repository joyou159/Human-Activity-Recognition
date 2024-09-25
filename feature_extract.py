from DSP_tools import *

def get_first_n_peaks(x,y,no_peaks = 5): 
    """ Extract the first "n_peaks" peaks of the transformed signal. If the number of peaks
        is not sufficient it fills the rest with zeros. 
        
        args:
            x: the x-coordinate of the transformed signal (time difference or frequency)
            y: the transformed signal values
            n_peaks: the number of peaks to extract from the signal
    """
    x_,y_ = list(x),list(y)
    if len(x_) >= no_peaks:
        return x_[:no_peaks],y_[:no_peaks]
    else: # filling the rest values with zeros if it's not sufficient
        missing_no_peaks = no_peaks - len(x_)
        return x_ + [0]*missing_no_peaks, y_ + [0]*missing_no_peaks 
    

def get_features(x_values,signal,mph): 
    """ extract some peaks from a transformed data at features.
            args:
                x_values: x-coordinate of the signal (i.e.,time difference & frequency)
                y_values: the transformed data (FFT, PSD or Autocorrelation values) 
                mph: the minimum peak height 
            
            return: 
                (peaks_x + peaks_y): a list of x and y coordinates concatenated 
    """
    indices_peaks = detect_peaks(signal,mph = mph) 
    peaks_x, peaks_y = get_first_n_peaks(x_values[indices_peaks],signal[indices_peaks])
    return peaks_x + peaks_y 



def extract_features_labels(dataset, T, N, f_s,denominator):
    """ Extract the features from the components of each signal and concatenate them together. 

        args: 
            dataset: 3d-ndarray consisting of the components of each signal
            T: sampling interval 
            N: number of samples
            f_s: sampling frequency 
            denominator: controller of mph parameter of the detect_peaks() as function of the signal values 
    """
    percentile = 5 
    list_of_features = list()
    for signal_no in range(len(dataset)): # iterate over each signal 
        features = list()
        for signal_comp in range(dataset.shape[2]): # iterate over each component
            signal = dataset[signal_no,:,signal_comp] 

            # technique of controlling the mph as a function of the signal values
            signal_min = np.percentile(signal,percentile)            
            signal_max = np.percentile(signal,100 - percentile)
            mph = signal_min + (signal_max - signal_min)/denominator
            

            features += get_features(*get_fft_values(signal,T,N,f_s),mph)
            features += get_features(*get_psd_values(signal,T,N,f_s),mph)
            features += get_features(*get_autocorr_values(signal,T,N,f_s),mph)
        list_of_features.append(features)
    return np.array(list_of_features)
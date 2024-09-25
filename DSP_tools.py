from scipy.signal import welch
import numpy as np
from detecta import detect_peaks 


def get_values(signal,T,N,f_s):
    """Compute the tick values corresponding to the given sampled data y_values. 

        args:
            y_values: sampled values of the signal
            T: sampling interval 
            N: number of samples
            f_s: sampling frequency  

        return: 
            x_values: tick values of sampled signal
    """
    x_values = [T * i for i in range(len(signal))]
    return x_values, signal 



def get_fft_values(signal,T,N,f_s):
    """Compute FFT of the sampled data besides the corresponding frequency values (0 --> Nyquist frequency)
    Nyquist frequency is is the highest possible frequency can be captured from a sampled signal, which 
    is equal to (f_s/2).

        args: 
            signal: sampled values of the signal
            T: sampling interval 
            N: number of samples
            f_s: sampling frequency  

        return:
            f_values: corresponding frequency values 
            fft_values: amplitude of each sinusoid at each frequency

        """
    f_values = np.linspace(0.0,1.0/(2.0*T),N//2) 
    # we have considered half of the sampled data points due to symmetry nature of FFT    
    fft_values = np.fft.fft(signal,N) # complex coefficients of fft 
    fft_values = (2/N) * np.abs(fft_values[:N//2])
    return f_values, fft_values 




def get_psd_values(signal,T,N,f_s):
    """Computes the power spectral density of the signal using Welch's method, which is an
    improvement on the standard periodogram spectrum estimating method. 

        args: 
            signal: sampled values of the signal
            T: sampling interval 
            N: number of samples
            f_s: sampling frequency  

        return:
            f_values: corresponding frequency values 
            psd_values: The power distributed over the signal for each frequency.   
             
    """
    f_values, psd_values = welch(signal,fs = f_s) 
    return f_values, psd_values 





def autocorr(x): 
    """Compute the autocorrelation of discrete signal by means of the convolution 
    of the signal with its inverse. so the time difference goes from (-N+1,N-1) where N is 
    the number samples in the signal.

        args:
            x: the sampled values of the signal
        
    """
    result = np.correlate(x,x,mode = "full") 
    return result[len(result)//2:] # correlation of the signal in the positive time difference 




def get_autocorr_values(signal,T, N, f_s):
    """ Compute the autocorrelation of discrete signal and the corresponding time differences  

        args: 
            signal: sampled values of the signal
            T: sampling interval 
            N: number of samples
            f_s: sampling frequency  

        return: 
            dt_values: the time differences values 
            autocorr_values: the autocorrelation of the discrete signal over the time differences
              
    """
    autocorr_values = autocorr(signal)
    dt_values = np.array([T * i for i in range(N)])
    return dt_values, autocorr_values 
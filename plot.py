import matplotlib.pyplot as plt 
import numpy as np
from detecta import detect_peaks 


def sample_plotting(signals, activity_name, list_functions, T, N, f_s):
    """Plot different signal transformations for a given activity.

    Args:
        signals: 2D array of signal data with x, y, z components.
        activity_name: Name of the activity, used in the plot title.
        list_functions: List of functions for time, FFT, PSD, and autocorrelation.
        T: Sampling interval.
        N: Number of samples.
        f_s: Sampling frequency.

    Returns:
        None: The function generates a plot with signal transformations.
    """
    labels = ['x-component', 'y-component', 'z-component']
    colors = ['r', 'g', 'b']
    suptitle = f"Different signals for the activity: {activity_name}"
    
    xlabels = ['Time [sec]', 'Freq [Hz]', 'Freq [Hz]', 'Time lag [s]']
    ylabel = 'Amplitude'
    axtitles = [['Acceleration', 'Gyro', 'Total acceleration'],
                ['FFT acc', 'FFT gyro', 'FFT total acc'],
                ['PSD acc', 'PSD gyro', 'PSD total acc'],
                ['Autocorr acc', 'Autocorr gyro', 'Autocorr total acc']
            ]

    f, axarr = plt.subplots(nrows=4, ncols=3, figsize=(12,12))
    f.suptitle(suptitle, fontsize=16)
    
    for row_no in range(0,4):
        for comp_no in range(0,9):
            col_no = comp_no // 3 # (0,1,2) 0, (3,4,5) 1, (6,,7,8) 2
            plot_no = comp_no % 3 # (0,1,2) --> (x,y,z)
            color = colors[plot_no]
            label = labels[plot_no]
    
            axtitle  = axtitles[row_no][col_no]
            xlabel = xlabels[row_no]
            value_retriever = list_functions[row_no]
    
            ax = axarr[row_no, col_no]
            ax.set_title(axtitle, fontsize=16)
            ax.set_xlabel(xlabel, fontsize=16)
            if col_no == 0:
                ax.set_ylabel(ylabel, fontsize=16)
    
            signal_component = signals[:, comp_no]
            x_values, y_values = value_retriever(signal_component, T, N, f_s)
            ax.plot(x_values, y_values, linestyle='-', color=color, label=label)
            if row_no > 0:
                max_peak_height = 0.1 * np.nanmax(y_values)
                indices_peaks = detect_peaks(y_values, mph=max_peak_height)
                ax.scatter(x_values[indices_peaks], y_values[indices_peaks], c=color, marker='*', s=60)
            if col_no == 2:
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))            
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.6)
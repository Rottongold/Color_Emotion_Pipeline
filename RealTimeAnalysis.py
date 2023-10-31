import numpy as np
from pylsl import StreamInlet, resolve_stream
from collections import deque
import time

def collect_stream_data(inlet, duration = 5):
    '''
    Function to extract EEG stream data 

    @param inlet: the incoming stream
    @param duration: int, the length of data, unit = second

    @return collected_data: array, EEG stream, dimension should be channel*sample
    '''
    collected_data = deque()

    sample_rate = inlet.info().nominal_srate()
    max_samples = int(sample_rate * duration)

    print(f"Collecting data for {duration} seconds...")

    start_time = time.time()

    while time.time() - start_time < duration:
        # Get the next sample
        sample, _ = inlet.pull_sample()
        collected_data.append(sample)

        # If more than max_samples are in the deque, remove the oldest
        while len(collected_data) > max_samples:
            collected_data.popleft()
    
    # collected_data should have the dimension of channel*sample (transpose of sample*channel)
    return np.array(collected_data).T

def frequency_bands(fs, EEG):
    '''
    Converting incoming EEG to frequency band distributions 

    @param fs: int, sampling frequency
    @param EEG: array, EEG data, dimension: channel*sample

    @return eeg_band_collection dict, average band distribuction across all channels 
    
    '''

    num_channels = EEG.shape[0]
    eeg_band_collection = []
    
    # Define the insterested frequency bands
    eeg_bands = {'Beta_Fast': (20, 30),
             'Beta_Middle': (16, 20),
             'Beta_Low': (12, 16),
             'Alpha_Fast': (12, 14),
             'Alpha_Middle': (9, 12),
             'Alpha_Low': (8, 9),
             'Theta': (4,7),
             'Gamma': (31, 50)}
    
    # Create dictionary to strore band amplitudes for the epoch
    eeg_band_fft = {
                 'Beta_Fast': [],
                 'Beta_Middle': [],
                 'Beta_Low': [],
                 'Alpha_Fast': [],
                 'Alpha_Middle': [],
                 'Alpha_Low': [],
                 'Theta': [],
                 'Gamma': []}
        
    for c in range(num_channels):
        
        epoch = EEG[c, :]
        
        # Get real amplitudes of FFT (only in postive frequencies)
        fft_vals = np.absolute(np.fft.rfft(epoch))

        # Get frequencies for amplitudes in Hz
        fft_freq = np.fft.rfftfreq(len(epoch), 1.0/fs)

        # Take the mean of the fft amplitude for each EEG band
        for band in eeg_bands:  
            freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & 
                               (fft_freq <= eeg_bands[band][1]))[0]
            eeg_band_fft[band].append(np.mean(fft_vals[freq_ix]))
           
    sum_power = 0
    #Compute the average power across all channels 
    for band in eeg_bands:
        eeg_band_fft[band] = np.mean(eeg_band_fft[band])
        sum_power += np.mean(eeg_band_fft[band])
        
    #Compute the band power distributions
    for band in eeg_bands:
        eeg_band_fft[band] = (eeg_band_fft[band]/sum_power)*100
        
    eeg_band_collection.append(eeg_band_fft)

    return eeg_band_collection

def generate_text(band_distribution):
    '''
    Translate the bands to corresponding colors, output the summary in string

    @param band_distribution: dict, frequency band distribuction 

    @return string, summary formmated in (color1): (percentage1)%, (color2): (percentage2)% ...
    '''
    
    # Define Color Translation 
    color2band = {
        'Red': 'Beta_Fast',
        'Orange': 'Beta_Middle',
        'Yellow': 'Beta_Low',
        'Green': 'Alpha_Fast',
        'Sky Blue': 'Alpha_Middle',
        'Indigo': 'Alpha_Low',
        'Violet': 'Theta',
        'Gold': 'Gamma'
    }

    text = []
    for color in color2band:
        text.append(f"{color}: {band_distribution[color2band[color]]:.2f}%")
    
    return ', '.join(text)

def main():

    #sampling rate
    fs = 100

    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

    while True:
        data = collect_stream_data(inlet, duration = 5)
        bands = frequency_bands(fs, data)
        print(generate_text(bands))        

if __name__ == '__main__':
    main()
from scipy.signal import find_peaks
import numpy as np
import copy

sinc_kernel = dict()
kernel_size = 10

def add_white_noise(original_signal, snr_db) -> np.ndarray:
	"""Adds white noise to the original signal to achieve the desired SNR."""
	if snr_db is None or snr_db == np.inf:
		return original_signal
	snr_linear = 10**(snr_db/10)
	p_noise = np.var(original_signal) / snr_linear
	noise = np.sqrt(p_noise) * np.random.randn(original_signal.shape[0])
	return original_signal + noise

def two_largest_peaks(ir, fdf_res=None):
	"""Returns the two largest peaks in an impulse response"""

	peaks, _ = find_peaks(ir, distance=10)
	if len(peaks) < 2:
		return None
	peak_vals = ir[peaks]               # get heights
	sorted_idxs = np.argsort(peak_vals) # get indices of sorted heights
	peaks = peaks[sorted_idxs[-2:]]     # get indices of two largest peaks
	pdp, pecho = np.sort(peaks)         # sort them into direct path and echo path

	if fdf_res is not None:
		pdp += find_peak_fdf(ir[pdp-kernel_size:pdp+kernel_size], kernel_size, fdf_res) - kernel_size
		pecho += find_peak_fdf(ir[pecho-kernel_size:pecho+kernel_size], kernel_size, fdf_res) - kernel_size
			
	return pdp, pecho

def find_peak_fdf(ir, p0, fdf_res):
	"""Finds the fractional sample value of the peak in the ir. Assumes the peak is at p0. Assumes a single peak in ir[p0-kernel_size:p0+kernel_size]

	Args:
		ir (array): The array containing an impulse response.
		p0 (int): Initial guess of the peak location, an integer.
		fdf_res (int): Fractional sample resolution.

	Returns:
		float: Best guess of the peak location in fractional samples.
	"""
	irsnip = ir[p0-kernel_size:p0+kernel_size]

	global sinc_kernel
	if fdf_res not in sinc_kernel.keys():
		sinc_kernel[fdf_res] = np.sinc(np.arange(-2*kernel_size, 2*kernel_size, 1/fdf_res))
	
	idvals = np.arange(-fdf_res, fdf_res, 1, dtype=int)
	xcorr = np.zeros(idvals.shape)
	for id in idvals:
		sk = sinc_kernel[fdf_res][kernel_size*fdf_res-id:3*kernel_size*fdf_res-id:fdf_res]
		xcorr[id+fdf_res] = np.dot(irsnip, sk)

	return p0 + idvals[np.argmax(xcorr)] / fdf_res

def gen_ir_seconds(siglen, fs, delays, amps):
	"""Generate impulse response from delays and amplitudes using sinc function.
	
	Args:
		length: float, in seconds
		fs: int, sampling frequency in Hz
		delays: list of floats, delays in seconds
		amps: list of floats, amplitudes
	"""
	t = np.linspace(0, siglen, int(siglen*fs), endpoint=False)
	ir = np.zeros_like(t)
	for delay, amp in zip(delays, amps):
		ir += amp * np.sinc(fs*(t - delay))
	return ir

def gen_ir_samples(siglen, fs, delays, amps):
	"""Generate impulse response from delays and amplitudes using sinc function.
	
	Args:
		length: int, in samples
		fs: int, sampling frequency in Hz
		delays: list of floats, delays in samples
		amps: list of floats, amplitudes
	"""
	return gen_ir_seconds(siglen/fs, fs, 
		np.array(delays) / fs, amps)

def shift_signal_fourier(signal, shift_samples):
	"""Shift signal in time domain by multiplying exponential term 
	in frequency domain. shift is in samples. """
	# get length of signal
	siglen = len(signal)
	# get frequency domain
	freq = np.fft.rfft(signal)
	# get frequency bins
	freqbins = np.fft.rfftfreq(siglen)
	# multiply exponential term
	freq = freq * np.exp(-1j * 2 * np.pi * freqbins * shift_samples)
	# get time domain
	shifted_signal = np.fft.irfft(freq)
	return shifted_signal

def centre_ir_from_cuts(ir, leftcut, rightcut):
	"""Centres ir around sample_num and snips to window sized 2*win_len_half+1. If leftcut or rightcut are out of bounds, pads with zeros. """
	if isinstance(leftcut, float):
		shiftval = leftcut - int(leftcut)
		ir_new = shift_signal_fourier(ir, -shiftval)
		leftcut = int(leftcut)
		rightcut = int(rightcut)
	else:
		ir_new = copy.deepcopy(ir)

	if leftcut < 0:
		ir_new = np.hstack((np.zeros(-leftcut), ir_new[:rightcut]))
	elif rightcut > len(ir):
		ir_new = np.hstack((ir_new[leftcut:], np.zeros(rightcut - len(ir_new))))
	else:
		ir_new = ir_new[leftcut:rightcut]

	return ir_new

def get_cuts(sample_num, win_len_half=30):
	"""Returns the left and right cut indices for a window of size 2*win_len_half+1 centred at sample_num. """
	win_len = win_len_half * 2 + 1
	leftcut = sample_num - win_len_half
	rightcut = sample_num + win_len_half + 1
	return leftcut, rightcut

def centre_ir(ir, sample_num, win_len_half=30):
	"""Centres ir around sample_num and snips to window sized 2*win_len_half+1. """
	win_len = win_len_half * 2 + 1
	assert win_len < len(ir), "Window length too long"

	leftcut = sample_num - win_len_half
	rightcut = sample_num + win_len_half + 1
	if leftcut < 0:
		ir_new = np.hstack((np.zeros(-leftcut), ir[:rightcut]))
	elif rightcut > len(ir):
		ir_new = np.hstack((ir[leftcut:], np.zeros(rightcut - len(ir))))
	else:
		ir_new = ir[leftcut:rightcut]

	return ir_new

if __name__ == "__main__":
	import gir
	import gir.config as gc
	from gir.dataset.panel_data import PanelData, REFERENCE
	from gir.ml.preprocess import get_panel_names
	import matplotlib.pyplot as plt

	ir = PanelData(get_panel_names()[4]).impulse_response_from_combination(2074)
	plt.plot(ir, '-o')
	dp, echo = two_largest_peaks(ir, 10)
	print(f'dp: {dp}, echo: {echo}')
	plt.axvline(dp)
	plt.axvline(echo)
	# dp -= int(dp)
	# echo -= int(echo)
	# plt.plot(shift_signal_fourier(ir, -dp), alpha=0.5, label='shifted dp')
	# plt.plot(shift_signal_fourier(ir, -echo), alpha=0.5, label='shifted echo')
	plt.legend()
	plt.show()

	plt.plot(centre_ir_from_cuts(ir, dp-10, dp+20))
	plt.plot(centre_ir_from_cuts(ir, int(dp)-10, int(dp)+20), alpha=0.3)
	plt.show()
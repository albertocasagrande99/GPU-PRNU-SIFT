## Main code
The provided code takes all H1 combinations (same Fingerprint device and video device), and tries to compute the basescaling parameter using the first frame of the video, if a match is found (PCE>50) both the parameters array and the basescaling (in ICIP format) will be printed, otherwise a failed message will appear with the best PCE value (<50) and the standard parameter array ([0.8, 0, 0, 0, 0])

## Noise and Fingerprint classes:
All the information about Noise and Fingerprint, like size, transformed tensors, normalization factors etc. are stored in these class instances to simplify the function calls and the clearity of the code.
With the self.get_gaussian(sigma) a gaussian smoothed version of the fingerprint or the noise can be retrieved in-line when calling other functions.

## find_basescaling(Class Noise, Class Fingerprint)
Compute basescaling factor from Fingerprint and Noise instances
2 step process with gaussian smoothing (controlled by sigma), calls transformation() and compute_PCE_matrices() on the basis of an internal-defined array of possible scales.
If no scaling factor is found check with no_smoothing flag set to True
The output of the function is the array of best_parameters in the format [scale, rotation (deg), _, _, _]
The internally defined 'rot_arr' define the search over the rotation parameters, set to [0, np.rad2deg(np.pi)] to check both frame orientations, otherwise use [0] to speed up the processing

## memlimit
The memlimit parameter in compute_PCE_matrices controls the amount of vram in Bytes that each step can take advantage of, set it to the maximum possible for your configuration
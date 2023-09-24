import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.getLogger('tensorflow').setLevel(logging.FATAL)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PRNU', 'CameraFingerprint'))
import src.Functions as Fu
import src.Filter as Ft

import numpy as np
import cv2
import tensorflow as tf
import tensorflow_addons as tfa
import support_function as sf
import glob

from scipy.io import loadmat
from scipy.io import savemat

def find_basescaling(noise, fingerprint, no_smoothing=False):
    size_fing = fingerprint.size_fing
    size_noise = noise.size_noise

    step = 0
    pcethres = 50
    bestparameters = [0.8, 0, 0, 0, 0]
    failed = True
    extended = False
    
    while step < 2:
        factor = (10 ** -step)

        if no_smoothing:
            sigma = 0
            grid_size = 0.001
        else:
            sigma = np.exp(-step)*2
            grid_size = 0.005

        if extended:
            # extend range to +- (0.1, 0.2)
            scale_arr = np.arange(0.1 * factor, 0.2 * factor, grid_size * factor)
        else:
            # try with range +- (0, 0.1)
            scale_arr = np.arange(0, 0.1 * factor, grid_size * factor)

        scale_arr = sorted(np.concatenate([scale_arr, -scale_arr[1:]]), key=abs)
        #rot_arr = [0, np.rad2deg(np.pi)] # check frame e rotated frame
        rot_arr = [0] 

        matrices, ranges, parameters = transformation(scale_arr, rot_arr, shear_arr = [0], proj_arr = [[0, 0]], hom = np.diag(np.ones(3)), bestparameters = bestparameters, noise_shape = size_noise, K_shape = size_fing)
        pce_arr = compute_PCE_matrices(noise.get_gaussian(sigma), fingerprint, matrices, ranges, early_stop=True)
        index_max = np.argmax(pce_arr)

        if pce_arr[index_max] > pcethres:
            pcethres = pce_arr[index_max]
            bestparameters = parameters[index_max]
            # if match found failed is False
            failed = False
            # if match found extended needs to be false for the future step
            extended = False
        
        if failed and (step == 0 and not extended):
            extended = True

        elif failed and (step == 0 and extended):
            break

        else:
            step = step + 1

    return bestparameters, pcethres, failed


def compute_PCE_matrices(noise, fingerprint, matrices, ranges, memlimit=8e9, early_stop=False):
    pce_values = []
    size_fing = fingerprint.size_fing
    size = (size_fing[0] * size_fing[1]) * len(matrices) * 64
    arr_max = np.int32(np.floor(len(matrices)*memlimit/size))
    div = np.int32(np.ceil(len(matrices)/arr_max))
    early_stop_counter = 0

    for i in range(div):
        #print(i+1, "/", div)
        arr_start = arr_max*i
        arr_end = min(arr_max*(i+1), len(matrices))
        list_Wrs = tf.expand_dims(tf.repeat(tf.expand_dims(tf.convert_to_tensor(noise.noise, dtype=tf.float32), axis=0), repeats=len(matrices[arr_start:arr_end]), axis=0), axis=-1)
        batch_Wrs_transformed = tfa.image.transform(tf.convert_to_tensor(list_Wrs, dtype=tf.float32), matrices[arr_start:arr_end], 'BILINEAR', output_shape = [size_fing[0], size_fing[1]])
        XC = (crosscorr_Fingeprint_GPU(batch_Wrs_transformed, fingerprint.TA_tf, fingerprint.norm2, fingerprint.size_fing))
        pce_values_batch = list(parallel_PCE(XC.numpy(), len(XC), ranges[arr_start:arr_end]))

        if pce_values and np.max(pce_values_batch) < np.max(pce_values)/5 and early_stop:
            early_stop_counter = early_stop_counter + 1
        
        pce_values.extend(pce_values_batch)

        if early_stop_counter > 2:
            break
    
    return pce_values

def transformation(scale_arr, rot_arr, shear_arr, proj_arr, hom, bestparameters, noise_shape, K_shape):
    ranges = []
    parameters = []
    matrices = []

    for i1 in range(len(scale_arr)):
        for i2 in range(len(rot_arr)):
            for i3 in range(len(shear_arr)):
                for i4 in range(len(proj_arr)):
                    scale = bestparameters[0] + scale_arr[i1]
                    rotation = bestparameters[1] + rot_arr[i2]
                    shear = bestparameters[2] + shear_arr[i3]
                    proj_1 = bestparameters[3]/noise_shape[1] + proj_arr[i4][0]
                    proj_2 = bestparameters[4]/noise_shape[0] + proj_arr[i4][1]


                    scale_rot_mat =  np.r_[cv2.getRotationMatrix2D((noise_shape[1] / 2, noise_shape[0] / 2), rotation, scale), [[0, 0, 1]]]

                    shear_mat = np.diag(np.ones(3))
                    shear_mat[0,1] = shear
                    shear_mat[1,0] = shear
                    proj_mat = np.diag(np.ones(3))
                    proj_mat[2,0] = proj_1/noise_shape[1]
                    proj_mat[2,1] = proj_2/noise_shape[0]
                    
                    matrix = np.matmul(np.matmul(np.matmul(hom, scale_rot_mat), shear_mat), proj_mat)

                    corners_origin = np.asarray([[0, 0, 1], [noise_shape[1], 0, 1], [0, noise_shape[0], 1], [noise_shape[1], noise_shape[0], 1]])
                    corners_mat = np.asarray([np.matmul(matrix, point) for point in corners_origin])

                    if np.max(corners_mat[:,1]) - np.min(corners_mat[:,1]) > K_shape[0] or np.max(corners_mat[:,0]) - np.min(corners_mat[:,0]) > K_shape[1]:
                        continue
                    
                    matrix[0,2] -= np.min(corners_mat[:,0])
                    matrix[1,2] -= np.min(corners_mat[:,1])

                    inv_matrix = np.linalg.inv(matrix)
                    inv_matrix = inv_matrix/inv_matrix[2,2]

                    parameters.append([scale, rotation, shear, proj_1, proj_2])
                    matrices.append(inv_matrix.reshape(9))
                    ranges.append(np.asarray([K_shape[0] - np.max(corners_mat[:,1]) + np.min(corners_mat[:,1]), K_shape[1] - np.max(corners_mat[:,0]) + np.min(corners_mat[:,0])]).astype(int))

    if len(matrices):
        return np.asarray(matrices)[:, :8], np.asarray(ranges), np.asarray(parameters)
    
    else: 
        return None, None, None

def crosscorr_Fingeprint_GPU(batchW, TA, norm2, sizebatch_K):
    meanW_batch = (tf.repeat(tf.repeat((tf.expand_dims(tf.expand_dims(tf.reduce_mean(batchW,
                                                       axis=[1, 2]), axis=2), axis=3)),
                              repeats=sizebatch_K[0], axis=1), repeats=sizebatch_K[1], axis=2))
    batchW = batchW - meanW_batch
    normalizator = tf.math.sqrt(tf.reduce_sum(tf.math.pow(batchW, 2)) * norm2)
    FA = tf.signal.rfft2d(tf.cast(tf.squeeze(batchW, axis= 3), tf.float32))
    AC = tf.multiply(FA, tf.repeat(tf.expand_dims(TA, axis=0), axis=0, repeats=len(batchW.numpy())))
    return tf.signal.irfft2d(AC) / normalizator


def parallel_PCE(CXC, idx, ranges, squaresize=11):
    out = np.zeros(idx)
    for i in range(0, idx):
        shift_range = ranges[i]
        Out = dict(PCE=[], pvalue=[], PeakLocation=[], peakheight=[], P_FA=[], log10P_FA=[])
        C = CXC[i]
        Cinrange = C[-1-shift_range[0]:,-1-shift_range[1]:]
        [max_cc, imax] = np.max(Cinrange.flatten()), np.argmax(Cinrange.flatten())
        [ypeak, xpeak] = np.unravel_index(imax,Cinrange.shape)[0], np.unravel_index(imax,Cinrange.shape)[1]
        Out['peakheight'] = Cinrange[ypeak,xpeak]
        del Cinrange
        Out['PeakLocation'] = [shift_range[0]-ypeak, shift_range[1]-xpeak]
        C_without_peak = _RemoveNeighborhood(C,
                                         np.array(C.shape)-Out['PeakLocation'],
                                         squaresize)
        # signed PCE, peak-to-correlation energy
        PCE_energy = np.mean(C_without_peak*C_without_peak)
        out[i] = (Out['peakheight']**2)/PCE_energy * np.sign(Out['peakheight'])
    return out


def _RemoveNeighborhood(X,x,ssize):
    # Remove a 2-D neighborhood around x=[x1,x2] from matrix X and output a 1-D vector Y
    # ssize     square neighborhood has size (ssize x ssize) square
    [M, N] = X.shape
    radius = (ssize-1)/2
    X = np.roll(X,[np.int32(radius-x[0]),np.int32(radius-x[1])], axis=[0,1])
    Y = X[ssize:,:ssize];   Y = Y.flatten()
    Y = np.concatenate([Y, X.flatten()[int(M*ssize):]], axis=0)
    return Y

class Fingerprint:
    def __init__(self, fingerprint, sigma = 0):
        self.size_fing_native = np.shape(fingerprint)
        self.fingerprint = cv2.resize(fingerprint, (0, 0), fx=(1920/fingerprint.shape[1]), fy=(1920/fingerprint.shape[1]))
        if sigma:
            self.fingerprint = cv2.GaussianBlur(self.fingerprint, (0,0), sigma)
        self.size_fing = np.shape(self.fingerprint)
        array2 = self.fingerprint.astype(np.double)
        array2 = array2 - array2.mean()
        tilted_array2 = np.fliplr(array2)
        tilted_array2 = np.flipud(tilted_array2)
        self.norm2 = np.sum(np.power(array2, 2))
        self.TA_tf = tf.signal.rfft2d(tf.cast(tilted_array2, tf.float32))

    def get_gaussian(self, sigma):
        return Fingerprint(self.fingerprint, sigma)

class Noise:
    def __init__(self, image, sigma = 0):
        self.image = image
        if sigma:
            tmp = cv2.GaussianBlur(self.image, (0,0), sigma)
        tmp = Ft.NoiseExtractFromImage(self.image, sigma=2.)
        tmp = Fu.WienerInDFT(tmp, np.std(tmp))

        self.noise = tmp
        self.size_noise = np.shape(tmp)

    def get_gaussian(self, sigma):
        return Noise(self.image, sigma)
    
FLAGS = tf.compat.v1.flags.FLAGS
# dataset
tf.compat.v1.flags.DEFINE_string('videos', 'VISION/videos/', 'path to videos')
tf.compat.v1.flags.DEFINE_string('fingerprint', 'fingerprints/', 'path to fingerprint')
tf.compat.v1.flags.DEFINE_string('output', 'OUTPUT/', 'path to output')
tf.compat.v1.flags.DEFINE_string('gpu_dev', '/gpu:0', 'gpu device')

os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu_dev[-1]

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)

with tf.device(FLAGS.gpu_dev):
    fingerprints = sorted(glob.glob(os.path.join(FLAGS.fingerprint, '*.mat')))
    videos = sorted([video for device in list(set([fingerprint.split('Fingerprint_')[-1][:3] for fingerprint in fingerprints])) for video in glob.glob(os.path.join(FLAGS.videos, device+'*'))])
    combinations_H1 = [(fingerprint, video) for video in videos for fingerprint in fingerprints if fingerprint.split('Fingerprint_')[-1][:3] in video]

    for combination in combinations_H1:
        fingerprint_path = combination[0] 
        video_path = combination[1]

        device = os.path.basename(os.path.splitext(fingerprint_path)[0]).split('_')[-1]

        print("Device:", device)
        print("Fingerprint:", fingerprint_path)
        print("Video:", video_path)

        vid = cv2.VideoCapture(video_path)
        height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)

        if height>width:
            print('Skipping vertical video')
            continue

        K = loadmat(fingerprint_path)
        fingerprint = Fingerprint(K['fing'])

        cap = cv2.VideoCapture(video_path)
        _, frame = cap.read()
        noise = Noise(frame)
        scaling_parameters, pce, failed = find_basescaling(noise, fingerprint)

        #if failed:
        #    print("failed")
        #    scaling_parameters, pce, failed = find_basescaling(noise, fingerprint, no_smoothing = True)

        print("Basescaling search failed!" if failed else "Found basescaling!")
        print("With PCE:", pce)
        print("Parameters:", scaling_parameters)

        if not failed:
            factor = fingerprint.size_fing_native[1]/1920
            original_scaling = factor*scaling_parameters[0]
            print("Original format basescaling:", original_scaling)


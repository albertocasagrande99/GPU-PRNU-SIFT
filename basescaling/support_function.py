import sys
#sys.path.insert(1, 'PRNU/CameraFingerprint/')
import src.Functions as Fu
import src.Filter as Ft
import src.maindir as md
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_addons as tfa
import statistics
import math
import os
from math import log10, sqrt
from skimage.color import rgb2gray


def PSNR(original, compressed):
    original = np.float32(rgb2gray(original))/255.0
    compressed = np.float32(rgb2gray(compressed))/255.0
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def frame_selector_wpsnr(path_video, index_frame, flag_init=0):
    cap = cv2.VideoCapture(path_video)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    cap.set(cv2.CAP_PROP_POS_FRAMES, index_frame[0])
    ret, oframe = cap.read()
    mov_array = []
    for idx in index_frame[1:len(index_frame)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret == True:
            mov_array.append(PSNR(oframe, frame))
            oframe = frame
    min_mov = np.where(mov_array == np.max(mov_array))[0][0]
    return min_mov


def frame_selector_new(path_video, index_frame, flag_init=0):
    cap = cv2.VideoCapture(path_video)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    cap.set(cv2.CAP_PROP_POS_FRAMES, index_frame[0])
    ret, oframe = cap.read()

    sift = cv2.SIFT_create()
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    mov_array = []
    for idx in index_frame[1:len(index_frame)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret == True:
            frame_k, frame_d = sift.detectAndCompute(frame, None)
            if 'oframe_k' not in locals() and 'oframe_d' not in locals():
                oframe_k, oframe_d = sift.detectAndCompute(oframe, None)

            if frame_d is not None and oframe_d is not None:
                match_all = matcher.knnMatch(frame_d, oframe_d, k=2)
                matches = []
                if np.shape(match_all)[1] == 2:
                    for m, n in match_all:
                        if m.distance < n.distance * 0.6:
                            matches.append(m)

                    p_frame = np.zeros((len(matches), 2))
                    p_oframe = np.zeros((len(matches), 2))
                    if len(matches) and len(matches)>4:
                        for i in range(len(matches)):
                            p_frame[i, :] = frame_k[matches[i].queryIdx].pt
                            p_oframe[i, :] = oframe_k[matches[i].trainIdx].pt

                        hom, _ = cv2.findHomography(p_frame, p_oframe, cv2.USAC_DEFAULT)
                        if hom is not None:
                            p_est = np.zeros(np.shape(p_oframe))
                            p_est = np.dot(hom, np.append(p_oframe, np.ones((np.shape(p_oframe)[0], 1)), axis=1).transpose())[
                                     :2].transpose()

                            errors = np.zeros(np.shape(p_oframe))
                            errors[:, 0] = np.asarray(
                                        [math.sqrt((p_frame[i, 0] - p_est[i, 0]) ** 2 + (p_frame[i, 1] - p_est[i, 1]) ** 2) for i in
                                        range(np.shape(p_frame)[0])])
                            errors[:, 1] = np.asarray([np.arctan2(p_frame[i, 0] - np.shape(frame)[0] / 2,
                                                      p_frame[i, 1] - np.shape(frame)[1] / 2) - np.arctan2(
                            p_est[i, 0] - np.shape(frame)[0] / 2, p_est[i, 1] - np.shape(frame)[1] / 2) for i in
                                              range(np.shape(p_frame)[0])])
                            errors_points = np.asarray(
                                    [np.asarray([oframe_k[matches[i].trainIdx], frame_k[matches[i].queryIdx]]) for i in
                                    range(len(matches))])
                            errors_final = []
                            if 'errors_cum' in locals() and 'errors_cum_points' in locals():
                                errors_cum_temp = []
                                errors_cum_points_temp = []
                                match_index = []
                                alpha = 0.35
                                for index, keypoint in enumerate(errors_points):
                                    if keypoint[0] in errors_cum_points[:, 1]:
                                        cum_idx = np.where(errors_cum_points[:, 1] == keypoint[0])[0][0]
                                        errors_cum_temp.append([alpha * errors[index, 0] + (1 - alpha) * errors_cum[cum_idx, 0],
                                                        alpha * errors[index, 1] + (1 - alpha) * errors_cum[cum_idx, 1]])
                                        errors_final.append([alpha * errors[index, 0] + (1 - alpha) * errors_cum[cum_idx, 0],
                                                     alpha * errors[index, 1] + (1 - alpha) * errors_cum[cum_idx, 1]])
                                        match_index.append(index)
                                    else:
                                        errors_cum_temp.append(list(errors[index]))
                                        errors_cum_points_temp.append(keypoint)
                                errors_final = np.asarray(errors_final)
                                errors_cum = np.asarray(errors_cum_temp)
                                errors_cum_points = np.asarray(errors_cum_points_temp)
                            else:
                                errors_cum = errors.copy()
                                errors_cum_points = errors_points.copy()
                                match_index = range(len(matches))

                            if len(errors_final) == 0:
                                errors_final = errors.copy()
                                match_index = range(len(matches))
                            if len(errors_final) > 1:
                                dist_thresh = \
                                sorted(errors_final, key=lambda error: error[0])[np.uint32(np.rint(len(errors_final) * 0.7))][0]
                                rot_thresh = \
                                sorted(errors_final, key=lambda error: error[1])[np.uint32(np.rint(len(errors_final) * 0.7))][1]

                                filtered_matches = [matches[index] for pos, index in enumerate(match_index) if
                                        errors_final[pos, 0] < dist_thresh and errors_final[pos, 1] < rot_thresh]
                                if len(filtered_matches):
                                    for i in range(len(filtered_matches)):
                                        p_frame[i, :] = frame_k[filtered_matches[i].queryIdx].pt
                                        p_oframe[i, :] = oframe_k[filtered_matches[i].trainIdx].pt
                                    #qui calcolo distanza per stima del moto
                                    mov_array.append(statistics.mean([math.sqrt(
                                            (frame_k[filtered_matches[i].queryIdx].pt[0] - oframe_k[filtered_matches[i].trainIdx].pt[0]) ** 2 + (
                                                frame_k[filtered_matches[i].queryIdx].pt[1] - oframe_k[filtered_matches[i].trainIdx].pt[1]) ** 2) for
                                                          i in range(len(filtered_matches))]))
                                    print("test motion estimation: ",mov_array[-1])
                                else:
                                    mov_array.append(1000)
                            else:
                                mov_array.append(1000)
                        else:
                            mov_array.append(1000)
                    else:
                        mov_array.append(1000)
                else:
                    mov_array.append(1000)
            oframe_k = frame_k
            oframe_d = frame_d
    cap.release()
    min_mov = np.where(mov_array == np.min(mov_array))[0][0]
    if mov_array[min_mov] > 10:
        min_mov = 0
    return min_mov

def frame_selector(path_video, index_frame, flag_init=0, array=False):
    cap = cv2.VideoCapture(path_video)
    mov_array = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, index_frame[0])
    ret, oframe = cap.read()
    #print(np.shape(oframe))
    for idx in index_frame[1:len(index_frame)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret is True:
                orb = cv2.SIFT_create()
                queryKeypoints, queryDescriptors = orb.detectAndCompute(frame, None)
                if 'trainKeypoints' not in locals():
                    trainKeypoints, trainDescriptors = orb.detectAndCompute(oframe, None)
                if queryDescriptors is not None and trainDescriptors is not None:
                    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
                    matches = matcher.match(queryDescriptors, trainDescriptors)
                    matches = sorted(matches, key=lambda x: x.distance)
                    matches = matches[:int(len(matches) * 0.9)]
                    if len(matches):
                        med = statistics.median([math.sqrt(
                            (queryKeypoints[match.queryIdx].pt[0] - trainKeypoints[match.trainIdx].pt[0]) ** 2 + (
                                queryKeypoints[match.queryIdx].pt[1] - trainKeypoints[match.trainIdx].pt[1]) ** 2) for
                                         match in matches])
                        matches = [match for match in matches if math.sqrt(
                            (queryKeypoints[match.queryIdx].pt[0] - trainKeypoints[match.trainIdx].pt[0]) ** 2 + (
                                queryKeypoints[match.queryIdx].pt[1] - trainKeypoints[match.trainIdx].pt[
                                1]) ** 2) < med * 10]
                        no_of_matches = len(matches)
                        mov_array.append(statistics.mean([math.sqrt(
                                (queryKeypoints[match.queryIdx].pt[0] - trainKeypoints[match.trainIdx].pt[0]) ** 2 + (
                                    queryKeypoints[match.queryIdx].pt[1] - trainKeypoints[match.trainIdx].pt[1]) ** 2) for
                                                  match in matches]))
                        #print("test motion estimation: ",mov_array[-1])
                        trainKeypoints = queryKeypoints
                        trainDescriptors = queryDescriptors
                    else:
                        mov_array.append(1000)
                        trainKeypoints = queryKeypoints
                        trainDescriptors = queryDescriptors
                else:
                    mov_array.append(1000)
                    del trainKeypoints, trainDescriptors
                    oframe = frame
        else:
                cap.release()
                break
    if array:
        return mov_array
    else:
        min_mov = np.where(mov_array == np.min(mov_array))[0][0]
        if mov_array[min_mov] > 10:
            #print("min mov more than 10")
            min_mov = 0
        return min_mov

def frame_selector_flow(path_video, index, flag_init=0, array=False, rescaling=0.5, stride=1):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RAFT', 'core'))
    import flow_utils as FU
    import torch
    models = FU.flow_methods()
    sys.path.remove(sys.path[0])
    def nearest_intersection(points, dirs):
        dirs_mat = torch.matmul(dirs[:, :, None], dirs[:, None, :])
        I = torch.eye(2).to('cuda')
        solution = torch.linalg.lstsq((I - dirs_mat).sum(dim=0),(torch.matmul((I - dirs_mat), points[:, :, None])).sum(dim=0))
        return solution.solution

    cap = cv2.VideoCapture(path_video)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    score_arr = []

    for gop_number, gop_start_frame in enumerate(index[:-1]):
        gop_indexes = range(index[gop_number], index[gop_number+1])
        gop_score = 0        

        for frame_idx in gop_indexes[::stride]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret is True:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, None, fx=rescaling, fy=rescaling)
                size_frame = np.shape(frame)

                if frame_idx == gop_indexes[0]:
                    oframe = frame
                    continue
                    
                flow = models.get_flow(oframe, frame)[0]

                oframe = frame

                flow_no_translation = flow.permute(1, 2, 0).sub(flow.mean(dim=-1).mean(dim=-1))

                normalization = torch.linalg.norm(flow_no_translation, dim=2)
                flow_direction_tang = torch.stack((flow_no_translation[...,0], flow_no_translation[...,1]), dim=2).div(torch.stack((normalization, normalization), dim=2))
                flow_direction_norm = torch.stack((-flow_direction_tang[...,1], flow_direction_tang[...,0]), dim=2)

                application_grid = torch.stack(torch.meshgrid(torch.arange(0,size_frame[1], dtype=torch.float), torch.arange(0,size_frame[0], dtype=torch.float), indexing='xy'), axis=2).to('cuda').reshape((-1, 2))

                rotation_center = nearest_intersection(application_grid, flow_direction_norm.reshape((-1, 2)))[:,0]
                zoom_center = nearest_intersection(application_grid, flow_direction_tang.reshape((-1, 2)))[:,0]

                application_grid_transposed_rotation = application_grid - rotation_center
                application_grid_transposed_zoom = application_grid - zoom_center
                
                endpoints_transposed_rotation = application_grid_transposed_rotation - flow_no_translation.reshape((-1, 2))
                endpoints_transposed_zoom = application_grid_transposed_zoom - flow_no_translation.reshape((-1, 2))

                dot_product = torch.mul(application_grid_transposed_rotation, endpoints_transposed_rotation).sum(dim=-1)
                angle_sign = torch.linalg.cross(torch.nn.functional.pad(application_grid_transposed_rotation, (0,1), "constant", 0), torch.nn.functional.pad(endpoints_transposed_rotation, (0,1), "constant", 0), dim=1)[:,2].sign() 
                normalization = torch.mul(torch.linalg.norm(application_grid_transposed_rotation, dim=-1), torch.linalg.norm(endpoints_transposed_rotation, dim=-1))  

                rotation = (torch.acos(torch.clamp(dot_product.div(normalization), -1 + 1e-7, 1 - 1e-7)) * angle_sign).mean() 
                scale = (torch.linalg.norm(endpoints_transposed_zoom, dim=-1).div(torch.linalg.norm(application_grid_transposed_zoom, dim=-1)).sub(1)).mean() 

                s = torch.sin(rotation)
                c = torch.cos(rotation)
                z = torch.tensor(0).to('cuda')
                
                rotation_mat = torch.stack([torch.stack([c, -s]),
                                            torch.stack([s, c])])
                scale_mat = torch.stack([torch.stack([-scale, z]),
                                        torch.stack([z, -scale])])

                rotation_diff = (torch.matmul(application_grid_transposed_rotation, rotation_mat) - application_grid_transposed_rotation).reshape(flow_no_translation.shape)
                scale_diff = torch.matmul(application_grid_transposed_zoom, scale_mat).reshape(flow_no_translation.shape)


                flow_no_translation_rotation_scale = flow_no_translation.sub(rotation_diff).sub(scale_diff)

                frame_score = flow_no_translation_rotation_scale.abs().sum().to('cpu').numpy()
                gop_score = gop_score + frame_score

        score_arr.append(gop_score)

    if array:
        return score_arr
    else:
        return np.argmin(score_arr)

# crosscorrelation function with selected fingerprint
def crosscorr_Fingerprint(array1, TA, norm2):
    array1 = array1.astype(np.double)
    array1 = array1 - array1.mean()
    normalizator = np.sqrt(np.sum(np.power(array1, 2)) * norm2)
    FA = np.fft.fft2(array1)

    del array1
    AC = np.multiply(FA, TA)
    del FA

    if normalizator == 0:
        ret = None
    else:
        ret = np.real(np.fft.ifft2(AC)) / normalizator
    return ret

# Compute PCE from noise
def compute_PCE(noise, TA, norm2, Fingerprint):
    Noisex1 = np.zeros_like(Fingerprint)
    Noisex1[:noise.shape[0], :noise.shape[1]] = noise
    shift_range = [Fingerprint.shape[0] - noise.shape[0], Fingerprint.shape[1] - noise.shape[1]]
    #shift_range = [160, 240]
    C = crosscorr_Fingerprint(Noisex1, TA, norm2)
    det, det0 = md.PCE(C, shift_range=shift_range)
    return det['PCE']


# compute PCE from image
def compute_PCE_Check(image, TA, norm2, Fingerprint):
    Noisex = Ft.NoiseExtractFromImage(image, sigma=2.)
    Noisex = Fu.pWienerInDFT(Noisex, np.std(Noisex))
    Noisex1 = np.zeros_like(Fingerprint)
    Noisex1[:Noisex.shape[0], :Noisex.shape[1]] = Noisex
    shift_range = [Fingerprint.shape[0] - Noisex.shape[0], Fingerprint.shape[1] - Noisex.shape[1]]
    C = crosscorr_Fingerprint(Noisex1, TA, norm2)
    det, det0 = md.PCE(C, shift_range=shift_range)
    return det['PCE']

def calibration(homography, noise, centerrot, centerres, step, TA, norm2, Fingerprint):
    bestpce=0
    rotation=0
    scaling=0
    print(step)
    count=0
    modified=False
    modifiedcheck = False
    for i in sorted(np.arange(-((5 if step==0 else 0.5)*(10**-step)), ((5 if step==0 else 0.5)*(10**-step)), 0.1*(10**-step)), key=abs):
        matrix = homography + np.r_[cv2.getRotationMatrix2D((noise.shape[0] / 2, noise.shape[1] / 2), 2 * (centerrot+i), 1.0), [[0, 0, 1]]]
        matrix[2,2] = matrix[2,2] / (1+centerres)
        rotated = cv2.warpPerspective(noise, matrix, (np.shape(noise)[1], np.shape(noise)[0]))
        pcerot = compute_PCE(rotated, TA, norm2, Fingerprint)
        if pcerot > bestpce:
            bestpce = pcerot
            rotation = centerrot+i
            count = 0
            modified = True
            modifiedcheck = True
        else:
            count+=1
        if count>2 and modified:
            break

    count = 0
    modified = False
    for i in sorted(np.arange(-(0.05*(10**-step)), (0.05*(10**-step)), 0.01*(10**-step)), key=abs):
        scale = (1 + centerres + i)
        matrix = homography + np.r_[cv2.getRotationMatrix2D((noise.shape[0] / 2, noise.shape[1] / 2), 2 * rotation, 1.0), [[0, 0, 1]]]
        matrix[2, 2] = matrix[2, 2] / scale

        resized = cv2.warpPerspective(noise, matrix, (np.uint32(np.rint(np.shape(noise)[1] * scale)), np.uint32(np.rint(np.shape(noise)[0] * scale))))
        pceres = compute_PCE(resized, TA, norm2, Fingerprint)

        if pceres > bestpce:
            bestpce = pceres
            scaling = centerres+i

            count=0
            modified = True
            modifiedcheck = True

        else:
            count+=1

        if count>2 and modified:
            break

    if(step<3) and modifiedcheck:
        matrix, bestpce, rotation, scaling = calibration(homography, noise, rotation, scaling, step+1, TA, norm2, Fingerprint)

    matrix = homography + np.r_[cv2.getRotationMatrix2D((noise.shape[0] / 2, noise.shape[1] / 2), 2 * rotation, 1.0), [[0, 0, 1]]]
    matrix[2, 2] = matrix[2, 2] / (scaling + 1)

    if step==0:
        best = cv2.warpPerspective(noise, matrix, (np.uint32(np.rint(np.shape(noise)[1] * (scaling+1))), np.uint32(np.rint(np.shape(noise)[0] * (scaling+1)))))
        bestpce = compute_PCE(best, TA, norm2, Fingerprint)

    return matrix, bestpce, rotation, scaling

def calibration_GPU(homography, noise, centerrot, centerres, step, TA, norm2, Fingerprint):
    #pre calcolo i parametri dei cicli for, prima rotation e poi scaling,
    #parallelizzo su tfa.transform. RICORDA CHE L'ANGOLO IN ALTO A SINISTRA PER QUESTO METODO DEVE RIMANERE
    #IN ALTO A SINISTRA.

    #Usa tf function per stima in parallelo di cross corr e PCE
    bestpce=0
    rotation=0
    scaling=0

    count=0
    modified=False
    modifiedcheck = False
    # rotation estimation
    matrix = rotation_matrix_estimator(homography, noise.shape, centerrot, centerres, step)

    list_Wrs = tf.repeat(tf.expand_dims(tf.convert_to_tensor(noise, dtype=tf.float32), axis=0), repeats=samp, axis=0)
    rotated = tfa.image.transform(list_Wrs, matrix)

    #
    '''
    for i in sorted(np.arange(-((5 if step==0 else 0.5)*(10**-step)), ((5 if step==0 else 0.5)*(10**-step)), 0.1*(10**-step)), key=abs):
        matrix = homography + np.r_[cv2.getRotationMatrix2D((noise.shape[0] / 2, noise.shape[1] / 2), 2 * (centerrot+i), 1.0), [[0, 0, 1]]]
        matrix[2,2] = matrix[2,2] / (1+centerres)
        rotated = cv2.warpPerspective(noise, matrix, (np.shape(noise)[1], np.shape(noise)[0]))
        pcerot = compute_PCE(rotated, TA, norm2, Fingerprint)
        if pcerot > bestpce:
            bestpce = pcerot
            rotation = centerrot+i
            count = 0
            modified = True
            modifiedcheck = True
        else:
            count+=1
        if count>2 and modified:
            break
    '''
    count = 0
    modified = False
    for i in sorted(np.arange(-(0.05*(10**-step)), (0.05*(10**-step)), 0.01*(10**-step)), key=abs):
        scale = (1 + centerres + i)
        matrix = homography + np.r_[cv2.getRotationMatrix2D((noise.shape[0] / 2, noise.shape[1] / 2), 2 * rotation, 1.0), [[0, 0, 1]]]
        matrix[2, 2] = matrix[2, 2] / scale

        resized = cv2.warpPerspective(noise, matrix, (np.uint32(np.rint(np.shape(noise)[1] * scale)), np.uint32(np.rint(np.shape(noise)[0] * scale))))
        pceres = compute_PCE(resized, TA, norm2, Fingerprint)

        #print("PCE: %f" %pceres)

        if pceres > bestpce:
            bestpce = pceres
            scaling = centerres+i

            count=0
            modified = True
            modifiedcheck = True

        else:
            count+=1

        if count>2 and modified:
            break

    if(step<3) and modifiedcheck:
        #check this, it doesn't seem very correct
        #print("bestpce: %f" % bestpce)
        #print("bestrot: %f" % rotation)
        #print("bestscale: %f" % scaling)
        matrix, bestpce, rotation, scaling = calibration(homography, noise, rotation, scaling, step+1, TA, norm2, Fingerprint)

    matrix = homography + np.r_[cv2.getRotationMatrix2D((noise.shape[0] / 2, noise.shape[1] / 2), 2 * rotation, 1.0), [[0, 0, 1]]]
    matrix[2, 2] = matrix[2, 2] / (scaling + 1)

    if step==0:
        #print("bestpce: %f" % bestpce)
        #print("bestrot: %f" % rotation)
        #print("bestscale: %f" % scaling)
        best = cv2.warpPerspective(noise, matrix, (np.uint32(np.rint(np.shape(noise)[1] * (scaling+1))), np.uint32(np.rint(np.shape(noise)[0] * (scaling+1)))))
        bestpce = compute_PCE(best, TA, norm2, Fingerprint)

    return matrix, bestpce, rotation, scaling


def rotation_matrix_estimator(hom, noise_shape, centerrot, centerres, step):
    idx_hom = 0
    rotation_arr = sorted([i for i in np.arange(-((5 if step==0 else 0.5)*(10**-step)), ((5 if step==0 else 0.5)*(10**-step)), 0.1*(10**-step))], key=abs)
    matrix = np.zeros([100, 3, 3])
    for i in rotation_arr:
        matrix[idx_hom] = hom[idx_hom] + np.r_[cv2.getRotationMatrix2D((noise_shape[0] / 2, noise_shape[1] / 2), 2 * (centerrot+i), 1.0), [[0, 0, 1]]]
        matrix[idx_hom, 2, 2] = matrix[idx_hom, 2, 2] / (1+centerres)
        idx_hom += 1
    mat_reshape = matrix.reshape([100, 9])
    return mat_reshape[:, 0:8]



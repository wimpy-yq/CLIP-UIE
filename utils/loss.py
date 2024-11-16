import torch
import numpy as np

def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2
    fft_amp = fft_im[:,:,:,:,0]**2 + fft_im[:,:,:,:,1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2( fft_im[:,:,:,:,1], fft_im[:,:,:,:,0] )
    return fft_amp, fft_pha

def phase_consistency_loss(src_img, trg_img):
    score = 0
    for i in range(trg_img[0]):
    # get fft of both source and target
        fft_src = torch.fft.rfft2(src_img.clone(), dim=(-2,-1))
        fft_trg = torch.fft.rfft2(trg_img.clone(), dim=(-2,-1))

    # extract phase of both ffts
        _, pha_src = extract_ampl_phase(fft_src.clone())
        _, pha_trg = extract_ampl_phase(fft_trg.clone())
    # compute phase consistency loss
        FP_tensordot = torch.dot(pha_src.flatten(),pha_trg.flatten())
        pha_src_L2 = torch.norm(pha_src)
        pha_trg_L2 = torch.norm(pha_trg)
        FP_L2 = pha_src_L2*pha_trg_L2
        FP_ratio = FP_tensordot/FP_L2
        FP_loss =1-FP_ratio
        score = score + FP_loss

    return score
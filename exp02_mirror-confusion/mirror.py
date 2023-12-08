stim_file_name = '/kaggle/input/perception/natural_stim_50_rotated_90.mat'
mat = scipy.io.loadmat(stim_file_name)
stim=mat['stim']
features=extract_features(stim,net)
def check_mirror_confusion(features):
    N = 100 # There are 100 unique stimuli
    nL = len(features[0]) - 1 # Skipping the last layers
    mirror_confusion_index = np.zeros((N, nL)) # Horizonatal , Vertical
    for img in range(100):
        img_numbers = [img, N+img, 2*N+img]
        for L in range(nL):
#             fi = np.squeeze(features[img_numbers[0]][L])
            fi=np.ravel(np.asarray(features[img_numbers[0]][L]))
            fYm=np.ravel(np.asarray(features[img_numbers[1]][L]))
            fXm=np.ravel(np.asarray(features[img_numbers[2]][L]))
#             fYm = np.squeeze(features[img_numbers[1]][L])
#             fXm = np.squeeze(features[img_numbers[2]][L])
            print(fi.shape)
            print(fi.ravel().shape)
            dYm = np.linalg.norm(fYm - fi, 2) # MIRROR ABOUT X-axis
            dXm = np.linalg.norm(fXm - fi, 2) # MIRROR ABOUT Y-axis
            mirror_confusion_index[img, L] = (dXm - dYm) / (dXm + dYm)
        
    return mirror_confusion_index
mi=check_mirror_confusion(features)
plot(mi,"MASKED AUTO ENCODER","MIRROR CONFUSION")
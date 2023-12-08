def check_occlusion(features, dist_type):
    nL = len(features[0])   # Skipping the last layers

    # check the basic effect (Stim in figure-1 of Rensink,1997)
    M_basic_effect = np.zeros((nL,2))
                              
    M_dept_ordering = np.zeros((nL,4))
    for l in range(nL):
        # first set
        img1 = np.ravel(np.asarray(features[0][l]))
        img2 = np.ravel(np.asarray(features[1][l]))
        img3 = np.ravel(np.asarray(features[2][l]))
        d12=np.linalg.norm(img1-img2,2)
        print(np.sqrt(np.sum((img1 - img2) ** 2)))
        d13=np.linalg.norm(img1-img3,2)
        print(d12,d13)
#         d12 = distance_calculation(img1, img2, dist_type)
#         d13 = distance_calculation(img1, img3, dist_type)
        M_basic_effect[l, 0] = (d13 - d12) / (d13 + d12)

        # 180 degreee rotated set
        img4 = np.ravel(np.asarray(features[3][l]))
        img5 = np.ravel(np.asarray(features[4][l]))
        img6 = np.ravel(np.asarray(features[5][l]))
        d45=np.linalg.norm(img4-img5,2)
        d46=np.linalg.norm(img4-img6,2)
#         d45 = distance_calculation(img4, img5, dist_type)
        
#         d46 = distance_calculation(img4, img6, dist_type)
        M_basic_effect[l, 1] = (d46 - d45) / (d46 + d45)

    # Check the effect of depth ordering
    for l in range(nL):
        # first control set
        img7 = np.ravel(np.asarray(features[6][l]))
        img8 = np.ravel(np.asarray(features[7][l]))
        img2 = np.ravel(np.asarray(features[1][l]))
        d78=np.linalg.norm(img7-img8,2)
        d72=np.linalg.norm(img7-img2,2)
#         d78 = distance_calculation(img7, img8, dist_type)
#         d72 = distance_calculation(img7, img2, dist_type)
        M_dept_ordering[l, 0] = (d78 - d72) / (d78 + d72)

        # 180 degree rotated set
        img9 = np.ravel(np.asarray(features[8][l]))
        img10 = np.ravel(np.asarray(features[9][l]))
        img5 = np.ravel(np.asarray(features[1][l]))
        d910=np.linalg.norm(img9-img10,2)
        d95=np.linalg.norm(img9-img5,2)
#         d910 = distance_calculation(img9, img10, dist_type)
#         d95 = distance_calculation(img9, img5, dist_type)
        M_dept_ordering[l, 1] = (d910 - d95) / (d910 + d95)

        # second control set
        img2 = np.ravel(np.asarray(features[1][l]))
        img7 = np.ravel(np.asarray(features[6][l]))
        img11 = np.ravel(np.asarray(features[10][l]))
        d27=np.linalg.norm(img2-img7,2)
        d211=np.linalg.norm(img2-img11,2)
#         d27 = distance_calculation(img2, img7, dist_type)
#         d211 = distance_calculation(img2, img11, dist_type)
        M_dept_ordering[l, 2] = (d211 - d27) / (d211 + d27)

        # 180 rotated second control set
        img5 = np.ravel(np.asarray(features[4][l]))
        img9 = np.ravel(np.asarray(features[8][l]))
        img12 = np.ravel(np.asarray(features[11][l]))
        d59=np.linalg.norm(img5-img9,2)
        d512=np.linalg.norm(img5-img12,2)
#         d59 = distance_calculation(img5, img9, dist_type)
#         d512 = distance_calculation(img5, img12, dist_type)
        M_dept_ordering[l, 3] = (d512 - d59) / (d512 + d59)



    
    print(M_basic_effect.shape)
    print(M_dept_ordering.shape)
    print(np.nanmean(M_basic_effect, axis=1).shape)
    print( np.nanmean(M_dept_ordering, axis=1).shape)
    mi_occlusion_mean = []
    mi_occlusion_sem = []
    mi_occlusion_mean.append(np.nanmean(M_basic_effect, axis=1))
    mi_occlusion_mean.append(np.nanmean(M_dept_ordering, axis=1))
#     print(mi_occusion_mean.shape)
#     mi_occlusion_sem.append(np.nansem(M_basic_effect, axis=1))
#     mi_occlusion_sem.append(np.nansem(M_dept_ordering, axis=1))
    return mi_occlusion_mean

stim_file_name = 'CNN-perception-5b36b6c5f5a34ba5927879677f66ea9fd9a39fb7/exp10_occlusion/codes/occlusion_set1.mat'
mat = scipy.io.loadmat(stim_file_name)
stim=mat['stim']
# specify model,net
features=extract_features(stim,net)
for ind in range(len(stim)):
    #     stim[ind]
    stim[ind][0]=stim[ind][0][10:-10,10:-10,:]
mi=check_occlusion(features,'euclidean')
fig, ax = plt.subplots()
ax.plot(mi[1],'o',color='blue')
ax.plot(mi[1],color='blue')
ax.plot(mi[0],'o',color='red')
ax.plot(mi[0],color='red')
# plt.ylabel(ylabel)
# plt.xlabel(xlabel)

ax.set_ylim(-1,1)
ax.set_xlim(0,21)


rect3 = matplotlib.patches.Rectangle((0,0 ),
                                  21, 1,
                                  color ='lightgray')
ax.add_patch(rect3)
ax.text(0.8,0.8,'Human Perception',color='black',fontsize=10)

plt.show()
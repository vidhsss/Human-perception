
def CheckThatcherEffect(features):

  nL=len(features[0])
  # % print(features{1})
  N=20
  # features=features.reshape(N,4) 
  dist_type='Euclidean'
  thatcherIndex=np.zeros([N,nL])
  for ind in range (20):
      for L in range(nL):
          v1=features[ind][L];
          v2=features[ind+20][L];
          v3=features[ind+40][L];
          v4=features[ind+60][L];

          v12=distance_calculation(v1,v2,dist_type)
          v34=distance_calculation(v3,v4,dist_type)
          thatcherIndex[ind][L]=(v12-v34)/(v12+v34)                 
  return thatcherIndex
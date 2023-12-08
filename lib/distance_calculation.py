from scipy.spatial.distance import cosine
import numpy as  np
def distance_calculation(f1,f2,type):
# % [irow icol d]=size(f1);
# % temp=reshape(permute(f1,[2,1,3]),[irow*icol,d]);
# % [irow icol d]=size(f2);
# % temp2=reshape(permute(f2,[2,1,3]),[irow*icol,d]);
# % X=[temp;temp2];
  f1=f1.T
  f2=f2.T
  X=np.concatenate((f1.T, f2.T))
  if type =='Euclidean':
          # % d=norm(X(1,:)-X(2,:),2);
      try:
        # d=np.sqrt(np.sum((f1[:] - f2[:]) ** 2))
        d=np.linalg.norm(f1-f2,2)
      except:
        d=np.linalg.norm(X[0,:]-X[1,:])

  elif type =='CityBlock':
      d=np.linalg.norm(X[0,:]-X[1,:],1)
  elif type == 'Cosine':
          # % d=pdist(X,'cosine');
        
      d= cosine(X[0,:],X[1,:])
  # % elif type== 'pearson':
  # %         d=pdist(X,'correlation');
  # % elif type=='spearman' :
  # %          d=pdist(X,'spearman');
  else:
            d=0  
  return d
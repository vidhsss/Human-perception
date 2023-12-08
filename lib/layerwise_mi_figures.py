import matplotlib
import matplotlib.pyplot as plt 
import numpy as np
# model.add(layers.Dense(64, activation=opt))im
# def layerwise_mi_figures(mean_data,sem_data,file_name,reference_mi,reference_name,legend_name,y_label,shaded_region_name,y_limits):
# % GJ 17-12-2018

# % update log
# % 1-1-19 : Specify the file to saves for layerwise modulation index data
# % 9-1-19 : Adding legends
# % 2-7-2020: Restructured
# % 29-08-2020 : Corrected legend names, size of the pdf generated.

# % INPUTS
# % mean_data  : expects means data across layers (matrix N x 37 with values in the range -1 to 1)
# % sem_data   : expects standard error of data (matrix N x 37 )
# % file_name  : expects a string, generated figure will be saved in this name

# % INPUTS  (optional)
# % reference_mi      : Scale value between -1 and 1
# % reference_name    : String, Reference from which the behavior was taken.
# % legend_name       : Cell of strings of size N x1
# % y_label           : String, y axis label
# % shaded_region_name : String, meaning of the shaded region (default = 'Human perception')
# % ylimits              : Expected a 2x1 vector, [y_lowerlimit,y_upperlimit]
# % OUTPUT
# % Pdf in the specififed "file_name" will be generated.
# if not 'y_limits' or len(y_limits)==0:
#     y_limits=[-1,1]

# plt.figure(figsize=(10, 4))
# % drawing the human perception rectangle
# rS=1/y_limits[2] % relative scale to adjust the plots according to y-limits
# nL=123
# xL=1.5
# xU=nL
# yL=y_limits[0]
# yU=y_limits[1]
# dx=0.08
# dy=0.005
# FontSize=10
# %% Drawing the Rectangle
# rectangle_color=[0.8,0.8,0.8]
# h=rectangle('Position',[1+dx,0+dy,nL-dx,1-dy],'FaceColor',rectangle_color,'Edgecolor','none');
# %% Naming the layers by drawing the bottom rectangles
# % layer_grouping=[1,17.45;17.55 33.45;33.55,51.45;51.55,78.45;78.55,96.45;96.55,119];
# layer_grouping=[[1,1.1],[11.755 23.45],[23.55,35.45],[35.55,37.45],[37.55,39.45],[39.55,41]]
# layer_colour=[[0.8,0.8,0.8]]*5
# layer_colour.append([0.5, 0.5,0.5])
# box_width=0.2/rS
# % draw reactangles
# for ind in range(1):
#     plt.Rectangle([layer_grouping[ind,0]+dx,yL+dy,layer_grouping[ind,1]-layer_grouping[ind,0]-dx,box_width-dy],facecolor=layer_colour[ind,:],edgecolor='none')
#     plt.hold(True)
# % write text over it
# box_tex_position=[2,17.7,34,51.75,79,107.2]
# box_name={'','','','','',''}
# % for ind =1:size(layer_grouping,1)
# %     text(box_tex_position(ind),double(yL)+box_width/2+dy,box_name{ind},'color','k','FontSize',9);
# % end

# %% Plotting the data
# % layer IDS
# % index_input=1;
# % index_conv=[2,4,7,9,12,14,16,19,21,23,26,28,30];
# % index_relu=[3,5,8,10,13,15,17,20,22,24,27,29,31,34,36];
# % index_pool=[6,11,18,25,40];
# % index_fc=[33,35,37];
# index_input=1
# index_conv=[]
# index_relu=[]
# index_pool=[]
# index_fc=[]
# line_colour = colormap('Lines');
# %% Plotting the Data
# marker_size=2;
# layer_ind=1:nL;
# for ind in range(mean_data.shape[0]):
#     if (sem_data is not None and sem_data.any()):
#         shadedErrorBar(range(1, nL), mean_data[ind, :], sem_data[ind, :], lineprops={'-','markerfacecolor',line_colour[ind,:],\
#             'color',line_colour[ind,:],'LineWidth',0.5}, transparent=True, patchSaturation=0.3)
#     else:
#         plot(range(1, nL), mean_data[ind, :], '-','markerfacecolor',line_colour[ind,:],\
#             'color',line_colour[ind,:],'LineWidth',0.5)
# %     end
#     plt.plot(layer_ind[index_input],mean_data[ind,index_input],markeredgecolor=line_colour[ind,:],markerfacecolor=line_colour[ind,:],markersize=marker_size);plt.hold(True)
#     plt.plot(layer_ind[index_conv],'o',markersize=marker_size,markeredgecolor=line_colour[ind,:])
#     plt.plot(layer_ind[index_relu],'o',markersize=marker_size,markeredgecolor=line_colour[ind,:],markerfacecolor=line_colour[ind,:])
#     plt.plot(layer_ind[index_pool],'d',markersize=marker_size,markeredgecolor=line_colour[ind,:],markerfacecolor=line_colour[ind,:])
#     plt.plot(layer_ind[index_fc],'s',markersize=marker_size,markeredgecolor=line_colour[ind,:])end
# %% Visual search modulation index
# % if(exist('reference_mi')&& (~isempty(reference_mi)))
# %     if(length(reference_mi)==1)
# %         line([1 nL],[reference_mi reference_mi],'Marker','none','LineStyle','--','LineWidth',0.5, 'Color',[0 0 0]);
# %         text(2,reference_mi+0.1,reference_name,'FontSize',6)
# %     else
# %         for i=1:length(reference_mi)
# %             line([1 nL],[reference_mi(i) reference_mi(i)],'Marker','none','LineStyle','--','LineWidth',0.5, 'Color',[0 0 0]);
# %             text(2,reference_mi(i)+0.1,reference_name{i},'FontSize',6)
# %         end
# %     end
# % end
# %% Naming the Shaded region
# if 'shaded_region_name' in locals() and (not (shaded_region_name is None)):
#     text(3,0.9, shaded_region_name, fontsize=6)
# else:
#     text(3,0.9, 'Human perception', fontsize=6)
# %% Correcting the plot
# % set(gcf,'position',[5,5,6.4,4]); % Setting the size of the plot
# % xlim([xL,xU]);ylim([yL,yU]); % Setting x and y limits
# % H = gca; set(H,'XTick',[1 nL],'XTickLabel',[1 nL],'YTick',[yL,0,yU],'YTickLabel',[yL,0,yU]); %Limiting X and Y Ticks.
# % ylabel(y_label);xlabel('Swav Layers');
# % % Legends
# % h=get(gca,'Children'); % grab all the axes handles at once
# % N=size(mean_data,1);Nh=length(h);
# % sel_legends=zeros(N,1);for i=1:N,sel_legends(i)=Nh-(13+(i-1)*5);end
# % legend(h(sel_legends),legend_name,'FontSize',FontSize,'Location','best','Box','off');

# % end

def plot(data,xlabel,ylabel):
  fig, ax = plt.subplots()
  ax.plot(np.mean(data,axis=0),'o',color='blue')
  ax.plot(np.mean(data,axis=0),color='blue')
  
  plt.ylabel(ylabel)
  plt.xlabel(xlabel)

  ax.set_ylim(-1,1)
  ax.set_xlim(0,37)


  rect3 = matplotlib.patches.Rectangle((0,0 ),
                                      37, 1,
                                      color ='lightgray')
  ax.add_patch(rect3)
  ax.text(0.8,0.8,'Human Perception',color='black',fontsize=10)

  plt.show()
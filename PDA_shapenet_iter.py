import numpy as np
from pointnet.model import PointNetCls
from pointnet.dataset import ShapeNetDataset
from scipy.spatial.distance import cdist,squareform
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import os
from datetime import datetime
import matplotlib.pyplot as plt
import gc
from mpl_toolkits.mplot3d import Axes3D
plt.switch_backend('agg')


def create_sampling_grid(points,r_samp):
    # create sampling 1d sampling points for each dimension
    #print(type(points))
    x_max,y_max,z_max = np.max(points,axis=0)
    #print(x_max)
    #print(y_max)
    #print(z_max)
    x_min,y_min,z_min = np.min(points,axis=0)
    x_samp_pnts = np.arange(x_min+r_samp,x_max,r_samp)
    y_samp_pnts = np.arange(y_min+r_samp,y_max,r_samp)
    z_samp_pnts = np.arange(z_min+r_samp,z_max,r_samp)

    # create 3d grid based on 1d sampling points
    #print(x_min)
    #print(y_min)
    #print(z_min)
    #assert False
    Xgrid,Ygrid,Zgrid = np.meshgrid(x_samp_pnts,y_samp_pnts,z_samp_pnts)
    samp_grid = np.asarray([Xgrid.flatten(),Ygrid.flatten(),Zgrid.flatten()]).T
    #print(samp_grid)
    #print(samp_grid.shape)
    #assert False
    return samp_grid

def get_point_sets(points,sampling_grid,r_samp):
    dist_mat = cdist(sampling_grid,points,'euclidean')
    #print(dist_mat.shape)
    #assert False
    pnt_sets = [np.nonzero(dist<r_samp)[0] for dist in dist_mat]
    #print(len(pnt_sets))
    #assert False
    return pnt_sets

if not os.path.exists("./logs"):
    os.mkdir("./logs")

log_path = "./logs/" + datetime.now().strftime('%b-%d_%H:%M:%S')

try:
    os.mkdir(log_path)
except FileExistsError:
    print("Directory " , log_path ,  " already exists")

# load the shapenet dataset
model_pth = './utils/cls/cls_model_49.pth'
num_points = 2500
test_dataset = ShapeNetDataset(
    root='/home/eojin/PycharmProject/pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0',
    split='test',
    classification=True,
    npoints=num_points,
    data_augmentation=False)
num_classes = len(test_dataset.classes)

testdataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=False)


# initialize pointnet classifier with pretrained weights
classifier = PointNetCls(k=num_classes)
classifier.cuda()
classifier.load_state_dict(torch.load(model_pth))
classifier.eval()

# get one batch of the test dataloader
_, (points, target) = next(enumerate(testdataloader, 0))
target = target[:, 0]
points = points.transpose(2, 1)


print("Point size : {}".format(points.size()))
print("Target class : {}".format(target))

# original prediction
classifier = classifier.eval()
base_pred, _, _ = classifier(points.cuda())

base_pred_ind = base_pred.cpu().data.max(1)[1].numpy()[0]
print("Predicted class : {}".format(base_pred_ind))

softmax = nn.Softmax(dim=1).cuda()
base_output = softmax(base_pred).cpu().data.numpy()[0][base_pred_ind]
print("Predicted prob  : {}".format(base_output))
#assert False


# create sampling grid and extract PDA point sets
r_samp=0.05

points = points.transpose(1,2).view(-1,3)
points = points.data.numpy()

samp_grid = create_sampling_grid(points,r_samp)
#print(samp_grid)
#assert False
pda_pnt_sets = get_point_sets(points,samp_grid,r_samp)
#print(pda_pnt_sets)
pnt_indexes = np.arange(len(points))
#assert False

remove_points = np.array([]).astype(int)
remove_points_loc = []
remove_pda = []
for iter_num in range(1000):
    pda_predictions = []
    pda_grid_loc = []
    print("removed points : {}".format(remove_points))
    print("removed points shape : {}".format(remove_points.shape))

    for ind in range(0,len(pda_pnt_sets)):
        #print(len(pda_pnt_sets))
        #assert False
        pnt_set = pda_pnt_sets[ind]

        if remove_points.shape[0] > 0:
            pnt_set = np.concatenate((pnt_set, remove_points), axis=0)

        grid_loc = samp_grid[ind]
        # choose a random point not in the point set
        rand_pnt = np.random.choice(np.delete(pnt_indexes, pnt_set))
        # set all points in the point set to the value of the random point
        pda_points = np.copy(points)
        pda_points[pnt_set,:] = points[rand_pnt,:]
        # make prediction for modified point set
        points_tensor = torch.from_numpy(pda_points).float().cuda()
        points_tensor = points_tensor.transpose(0,1).view(1,3,-1)
        pda_pred, _, _ = classifier(points_tensor)
        pda_pred_output = softmax(pda_pred).cpu().data.numpy()[0][base_pred_ind]
        pda_predictions.append(base_output - pda_pred_output)
        pda_grid_loc.append(grid_loc)

    pda_max_point_ind = pda_predictions.index(max(pda_predictions))
    pda_max_point_loc = pda_grid_loc[pda_max_point_ind]
    pda_max_point_sets = pda_pnt_sets[pda_max_point_ind]

    remove_points = np.concatenate((remove_points, pda_max_point_sets), axis=0)
    remove_points = np.unique(remove_points)
    remove_points_loc.append(pda_max_point_loc)
    remove_pda.append(max(pda_predictions))

    if iter_num % 20 == 0:
        fig = plt.figure()

        ax = fig.gca(projection='3d')

        remove_points_np = np.array(remove_points_loc)
        X = remove_points_np[:, 0]
        Y = remove_points_np[:, 1]
        Z = remove_points_np[:, 2]

        pl = ax.scatter(X, Y, Z, c=(np.arange(len(remove_pda)) + 1), alpha=1.0, cmap='coolwarm_r', s=50)  # 150 when 0.1

        fig.colorbar(pl)

        X_ori = points[:, 0]
        Y_ori = points[:, 1]
        Z_ori = points[:, 2]

        ax.scatter(X_ori, Y_ori, Z_ori, alpha=0.01, c='g')

        ax.set_zlim(-1, 1)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        # plt.show(fig)
        print("Current status : {}".format(iter_num))
        plt.savefig(log_path + "/3d_" + str(iter_num))

        del fig, ax, X, Y, Z, X_ori, Y_ori, Z_ori
        gc.collect()
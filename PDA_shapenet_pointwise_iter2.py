import numpy as np
from pointnet.model import PointNetCls
from pointnet.dataset import ShapeNetDataset
from scipy.spatial.distance import cdist,squareform
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import matplotlib.pyplot as plt
import os
from datetime import datetime
import gc
plt.switch_backend('agg')
from mpl_toolkits.mplot3d import Axes3D

def get_point_sets(points,r_samp):
    dist_mat = cdist(points,points,'euclidean')
    #print(dist_mat.shape)
    #assert False
    pnt_sets = []
    for dist in dist_mat:
        pnt_sets.append(np.nonzero(dist<r_samp)[0])
    #print(len(pnt_sets))
    #assert False
    return pnt_sets

def check_loc(point, point_list):
    for point_tmp in point_list:
        if point == point_tmp:
            return True
    return False

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
r_samp=0.1
# original 0.05

points = points.transpose(1,2).view(-1,3)
points = points.data.numpy()

#samp_grid = create_sampling_grid(points,r_samp)
#print(samp_grid)
#assert False
pda_pnt_sets = get_point_sets(points,r_samp)
#print(pda_pnt_sets)
pnt_indexes = np.arange(len(points))
#assert False

remove_points_loc = []
remove_pda = []
for iter_num in range(1000):
    pda_predictions = []
    pda_grid_loc = []
    print("removed points : {}".format(remove_points_loc))

    best_pda = - 1.0
    best_pda_loc = None
    for ind in range(0,len(pda_pnt_sets)):
        pnt_set = pda_pnt_sets[ind]
        point_loc = points[ind]

        print(point_loc)
        print(remove_points_loc)
        if check_loc(point_loc, remove_points_loc):
            print(point_loc)
            continue
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

        if (base_output-pda_pred_output) >= best_pda:
            best_pda = (base_output-pda_pred_output)
            best_pda_loc = point_loc

    remove_points_loc.append(best_pda_loc)
    remove_pda.append(best_pda)

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
        plt.savefig(log_path + "/3d_"+str(iter_num))

        del fig, ax, X, Y, Z, X_ori, Y_ori, Z_ori
        gc.collect()


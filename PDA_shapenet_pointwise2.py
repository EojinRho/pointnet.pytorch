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
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.switch_backend('agg')


def get_point_sets(points,r_samp):
    dist_mat = cdist(points,points,'euclidean')
    point_sampled_sets = [(index,np.nonzero(dist<r_samp)[0]) for index, dist in enumerate(dist_mat)]
    point_sampled_sets = [samp for samp in point_sampled_sets if np.size(samp[1])>0]
    pnt_sets = [pnt_set for _,pnt_set in point_sampled_sets]
    point_index = [index for index,_ in point_sampled_sets]
    return pnt_sets, point_index

def main(args):
    # load the shapenet dataset
    model_pth = './utils/cls/cls_model_20.pth'
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
    points = None
    target = None
    for i, (points_tmp, target_tmp) in enumerate(testdataloader, 0):
        #print(target_tmp[:, 0].cpu().data.numpy()[0])
        if i == 3:
            points = points_tmp
            target = target_tmp
            break

    #_, (points, target) = next(enumerate(testdataloader, 0))
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


    points = points.transpose(1,2).view(-1,3)
    points = points.data.numpy()

    sphere_radius = args.sphere_radius
    pda_pnt_sets,_ = get_point_sets(points,sphere_radius)

    pnt2sets = {}
    for i in range(len(points)):
      pnt2sets[i] = [idx for idx,pnt_set in enumerate(pda_pnt_sets) if i in pnt_set]

    pnt_idxs = np.arange(len(points))
    pda_diff = []
    classifier = classifier.eval()
    for pnt_set in pda_pnt_sets:
        # choose a random point not in the point set
        rand_pnt = np.random.choice(np.delete(pnt_idxs, pnt_set))
        # set all points in the point set to the value of the random point
        pda_points = np.copy(points)
        pda_points[pnt_set,:] = points[rand_pnt,:]
        # make prediction for modified point set
        points_cuda = torch.tensor(np.expand_dims(pda_points,0))
        points_cuda = points_cuda.transpose(2, 1).cuda()
        pda_pred, _, _ = classifier(points_cuda)
        pda_pred_val = np.exp(pda_pred[0][base_pred_ind].item()) # convert to probability
        diff_val = base_output - pda_pred_val # get PDA value
        if args.PDA_mean == "False":
            pda_diff.append(diff_val)
        elif args.PDA_mean == "True":
            pda_diff.append(diff_val/len(pda_points))
        else:
            assert False, "PDA mean argument is not valid"
    pda_diff = np.array(pda_diff)

    pda_pnt_vals = [np.mean(pda_diff[pnt_set_idxs]) for pnt,pnt_set_idxs in pnt2sets.items()]

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111, projection='3d')
    img3d = ax.scatter(points[:,0],points[:,1],points[:,2],s=20,c=pda_pnt_vals,cmap = plt.cm.get_cmap('coolwarm'),vmin=min(pda_pnt_vals),vmax=max(pda_pnt_vals))
    fig.colorbar(img3d,shrink=0.7)
    plt.tight_layout()

    if not os.path.exists("./logs"):
        os.mkdir("./logs")

    log_path = "./logs/" + datetime.now().strftime('%b-%d_%H:%M:%S')

    try:
        os.mkdir(log_path)
    except FileExistsError:
        print("Directory " , log_path ,  " already exists")

    plt.savefig(log_path + "/3d_" + "PDA_shapenet_pointwise")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input the args")
    parser.add_argument("--sampling_radius", type=float, required=False, default=0.1,
                        help="Distance between each sample grid")
    parser.add_argument("--sphere_radius", type=float, required=False, default=0.2,
                        help="Radius of the sample sphere")
    parser.add_argument("--PDA_mean", type=str, required=False, default="False",
                        help="Take mean of PDA by number of points in sphere")
    args = parser.parse_args()
    main(args)
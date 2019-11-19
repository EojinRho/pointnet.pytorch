import numpy as np
from pointnet.model import PointNetCls
from pointnet.dataset import ModelNetDataset
from scipy.spatial.distance import cdist,squareform

def create_sampling_grid(points,r_samp):
    # create sampling 1d sampling points for each dimension
    x_max,y_max,z_max = np.max(points,axis=0)
    x_min,y_min,z_min = np.min(points,axis=0)
    x_samp_pnts = np.arange(x_min+r_samp,x_max,r_samp)
    y_samp_pnts = np.arange(y_min+r_samp,y_max,r_samp)
    z_samp_pnts = np.arange(z_min+r_samp,z_max,r_samp)
    # create 3d grid based on 1d sampling points
    Xgrid,Ygrid,Zgrid = np.meshgrid(x_samp_pnts,y_samp_pnts,z_samp_pnts)
    samp_grid = np.asarray([Xgrid.flatten(),Ygrid.flatten(),Zgrid.flatten()]).T
    return samp_grid

def get_point_sets(points,sampling_grid,r_samp):
    dist_mat = cdist(sampling_grid,points,'euclidean')
    pnt_sets = [np.nonzero(dist<r_samp)[0] for dist in dist_mat]
    return pnt_sets


# load the modelnet dataset
model_net_path = ""
num_points = 2500
test_dataset = ModelNetDataset(
    root=modelnet_path,
    split='test',
    npoints=num_points,
    data_augmentation=False)
num_classes = len(dataset.classes)

# initialize pointnet classifier with pretrained weights
classifier = PointNetCls(k=num_classes)
# pretrained_dict = torch.load('pointnet_pretrained_dict.pth')
# classifier.load_state_dict(pretrained_dict)

# get one batch of the test dataloader
points, target = next(testdataloader)
target = target[:, 0]
points = points.transpose(2, 1)

# original prediction
classifier = classifier.eval()
base_pred, _, _ = classifier(points)

# create sampling grid and extract PDA point sets
r_samp=0.2
samp_grid = create_sampling_grid(points,r_samp)
pda_pnt_sets = get_point_sets(points,sampling_grid,r_samp)
pnt_indexs = np.arange(len(points))

pda_predictions = []
for pnt_set in pda_pnt_sets:
    # choose a random point not in the point set
    rand_pnt = np.random.choice(np.delete(pnt_indexes, pnt_set))
    # set all points in the point set to the value of the random point
    pda_points = np.copy(points)
    pda_points[pnt_set,:] = points[rand_pnt,:]
    # make prediction for modified point set
    pda_pred, _, _ = classifier(pda_points)
    pda_predictions.append(pda_pred)










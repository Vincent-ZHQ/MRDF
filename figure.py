from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from mpl_toolkits.mplot3d import Axes3D
import time
import numpy as np

# colors_per_class = {
#     0 : [0, 204, 0],
#     1 : [0, 0, 204],
#     2 : [0, 0, 204],
#     3 : [0, 0, 204],
# }

# maker_per_class = {
#     0 : 'o',
#     1 : '+',
#     2 : 's',
#     3 : 'x',
# }

# label_per_class = {
#     0 : 'RA-RV',
#     1 : 'RA-FV',
#     2 : 'FA-RV',
#     3 : 'FA-FV',
# }

# size_per_class = {
#     0 : 100,
#     1 : 250,
#     2 : 90,
#     3 : 200,
# }

colors_per_class = {
    0 : [0, 0, 204],
    1 : [0, 204, 0],
    2 : [0, 0, 204],
    3 : [0, 204, 0],
}

maker_per_class = {
    0 : MarkerStyle('o', fillstyle='none'),#'+',
    1 : MarkerStyle('o', fillstyle='none'),
    2 : MarkerStyle('D', fillstyle='none'), #'x',
    3 : MarkerStyle('D', fillstyle='none')
}

label_per_class = {
    0 : 'FA',
    1 : 'RA',
    2 : 'FV',
    3 : 'RV',    
}

size_per_class = {
    0 : 50, # 100
    1 : 50,
    2 : 50, # 80
    3 : 50,
}


# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def visualize_tsne_points_2(name, tx, ty, labels):
    # initialize matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format:
        # BGR -> RGB, divide by 255, convert to np.array
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255
        marker = maker_per_class[label]
        label_txt = label_per_class[label]
        size = size_per_class[label]

        # add a scatter plot with the correponding color and label
        ax.scatter(current_tx, current_ty, s=size, c=color, label=label_txt, marker=marker)


    # build a legend using the labels we set previously
    ax.legend(loc='upper right', fontsize='x-large')
    plt.axis('off')
    plt.show()

    t = round(time.time() * 1000)
    t_str = time.strftime('%H_%M_%S', time.localtime(t / 1000))

    img_path = '/home/users/ntu/heqing00/Research/Code/LAVDF/output/figure/' + t_str + '_' + name + '.png'
    # finally, show the plot
    fig.savefig(img_path, dpi=500, bbox_inches='tight')


def visualize_tsne_2(name, tsne, labels):
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # visualize the plot: samples as colored points
    visualize_tsne_points_2(name, tx, ty, labels)

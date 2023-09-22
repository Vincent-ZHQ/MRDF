from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

colors_per_class = {
    0 : [0, 0, 0],
    1 : [255, 107, 107],
    2 : [100, 100, 255],
    3 : [16, 172, 132],
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

        # add a scatter plot with the correponding color and label
        ax.scatter(current_tx, current_ty, s=1, c=color, label=label)

    # build a legend using the labels we set previously
    ax.legend(loc='best')
    plt.show()

    t = round(time.time() * 1000)
    t_str = time.strftime('%H_%M_%S', time.localtime(t / 1000))

    img_path = '/home/users/ntu/heqing00/Research/Code/LAVDF/output/figure/' + t_str + '_' + name + '.png'
    # finally, show the plot
    fig.savefig(img_path, dpi=fig.dpi)


def visualize_tsne_2(name, tsne, labels):
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # visualize the plot: samples as colored points
    visualize_tsne_points_2(name, tx, ty, labels, params)

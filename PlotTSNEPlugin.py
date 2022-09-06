# Objective:
#   The purpose of the script is to apply t-sne to visually see if user and non-user groups form clusters.

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np

import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn import preprocessing



import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
RS = 123

def get_labels(samples, metadata_path, sample_id_col, label_col):

    # samples - list of sample ID's
    # return list of labels

    metadata_df = pd.read_csv(metadata_path, sep="\t")
    # metadata_df["group"] = metadata_df["COCAINE USE"].apply(lambda x: 1 if x=="Non-User" else 2)
    metadata_df.index = metadata_df[sample_id_col]
    metadata_df = metadata_df[[label_col]]
    metadata_dict = metadata_df.to_dict()[label_col]

    labels = []
    for sample in samples:
        labels.append(metadata_dict[sample])
    return labels

def load_dataset(data_path, metadata_path, sample_id_col, label_col, vis_type="samples", transpose=False):
    # Input:
    #   * data_path        path to abundance file
    #                       The abundance file should have the following format:
    #                       columns are taxa and rows are samples
    #
    #   * metadata_path    path to file that maps sample ID with label
    #
    #   * sample_id_col    name of the column with sample IDs in metadata file
    #
    #   * label_col        name of the column with labels in metadata file

    # Get data

    data_np = np.genfromtxt(data_path, delimiter=",", dtype=str)

    if transpose:
        data_np = np.transpose(data_np)

    if vis_type=="taxa":
        data_np = np.transpose(data_np)

    data_np = data_np[1:,:]

    # Get labels
    sample_ids = data_np[:,0]

    # get labels from sample ID's

    sample_ids = np.transpose(sample_ids)
    print(sample_ids)
    if vis_type=="samples":
        sample_ids_list = list(sample_ids)
        labels = get_labels(sample_ids_list, metadata_path, sample_id_col, label_col)
        labels = np.asarray(labels)
    elif vis_type=="taxa":
        labels = np.ones(len(sample_ids))

    data_np = data_np[:,1:]

    return data_np, labels, sample_ids

def plot_pca(data_np, labels):
    time_start = time.time()

    pca = PCA(n_components=4)
    pca_result = pca.fit_transform(data_np)

    print('PCA done! Time elapsed: {} seconds'.format(time.time() - time_start))

    pca_df = pd.DataFrame(columns=['pca1', 'pca2', 'pca3', 'pca4'])

    pca_df['pca1'] = pca_result[:, 0]
    pca_df['pca2'] = pca_result[:, 1]
    pca_df['pca3'] = pca_result[:, 2]
    pca_df['pca4'] = pca_result[:, 3]

    print('Variance explained per principal component: {}'.format(pca.explained_variance_ratio_))
    top_two_comp = pca_df[['pca1', 'pca2']]  # taking first and second principal component
    fashion_scatter(top_two_comp.values, labels)  # Visualizing the PCA output


def fashion_scatter(x, labels, label_clusters=True, show=True, Title="", savefig=None, show_legend=True):
    import matplotlib.patches as mpatches

    # get colors:
    unique_labels = np.unique(labels)
    labels_color_dict = {}

    for i, label in enumerate(unique_labels):
        labels_color_dict[label] = i

    colors_list = []
    for label in labels:
        colors_list.append(labels_color_dict[label])
    colors = np.asarray(colors_list)

    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # Legends:
    if show_legend:
        rgb_colors = palette[list(labels_color_dict.values())]
        rgb_labels = list(labels_color_dict.keys())
        patches = []
        for i,rgb_color in enumerate(rgb_colors):
            patches.append(mpatches.Patch(color=rgb_color, label=rgb_labels[i]))

        # rgb_colors_unique = palette[colors.astype(np.int)].unique()
        # rgb_colors = []
        # for c in rgb_colors:
        # red_patch = mpatches.Patch(color=palette[colors.astype(np.int)], label='The red data')
        # blue_patch = mpatches.Patch(color=colors[1], label='The blue data')

        plt.legend(handles=patches, loc=1, bbox_to_anchor=(1.08, 0.95), borderaxespad=0.)
    plt.title(Title)

    # add the labels for each digit corresponding to the label
    txts = []

    if label_clusters:
        for i in range(num_classes):

            # Position of each label at median of data points.

            xtext, ytext = np.median(x[colors == i, :], axis=0)
            txt = ax.text(xtext, ytext, str(i), fontsize=24)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)

    if show:
        plt.show()
    if savefig!=None:
        plt.savefig(savefig)

    return f, ax, sc, txts

def plot_tSNE(data_path, metadata_path, sample_id_col, label_col, out_plot, title, vis_type="samples", show_legend=True, transpose=False):

    data_np, labels, sample_ids = load_dataset(data_path, metadata_path, sample_id_col, label_col, vis_type=vis_type, transpose=transpose)
    data_np = preprocessing.scale(data_np)

    # Stage 1 - PCA analysis
    #plot_pca(data_np, labels)
    # n_components = 50
    # pca_model = PCA(n_components=n_components)
    #
    # pca_result = pca_model.fit_transform(data_np)
    # print('Cumulative variance explained by {} principal components: {}'.format(n_components, np.sum(pca_model.explained_variance_ratio_)))

    t_sne_model = TSNE(random_state=RS).fit_transform(data_np)
    fashion_scatter(t_sne_model, labels, label_clusters=False, Title=title, savefig=out_plot, show=False, show_legend=show_legend)

# General path:

#First, based only on abundance
# data_path = "/Users/stebliankin/Desktop/SabrinaProject/abundance/combined/abund.norm.csv"
# out_plot = "/Users/stebliankin/Desktop/SabrinaProject/tsne/tsne_abundance_samples.png"
# title = "t-SNE visualization of samples based on abundance"
# plot_tSNE(data_path, metadata_path, sample_id_col, label_col, out_plot, title, show_legend=True)
#
# # # Second, visualize taxa based on abundance
# data_path = "/Users/stebliankin/Desktop/SabrinaProject/abundance/combined/abund.norm.csv"
# out_plot = "/Users/stebliankin/Desktop/SabrinaProject/tsne/tsne_abundance_taxa.png"
# title = "t-SNE visualization of taxa based on abundance"
# plot_tSNE(data_path, metadata_path, sample_id_col, label_col, out_plot, title, show_legend=False, vis_type="taxa")
#
import PyPluMA

class PlotTSNEPlugin:
    def input(self, infile):
        parameterfile = open(infile, 'r')
        self.parameters = dict()
        for line in parameterfile:
           contents = line.strip().split('\t')
           self.parameters[contents[0]] = contents[1]

    def run(self):
        pass

    def output(self, outfile):
       plot_tSNE(PyPluMA.prefix()+"/"+self.parameters["datapath"], PyPluMA.prefix()+"/"+self.parameters["metadatapath"], self.parameters["sampleidcol"], self.parameters["labelcol"], outfile, self.parameters["title"], show_legend=True, vis_type="samples", transpose=True)




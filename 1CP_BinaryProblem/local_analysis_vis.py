import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.gridspec as gridspec
from PIL import Image
import numpy as np
import os
import argparse
import re
import shutil

classname_dict = dict()
classname_dict[0] = "Melanoma"
classname_dict[1] = "Nevus"


def main():
    # get dir
    parser = argparse.ArgumentParser()
    parser.add_argument('-local_analysis_directory', nargs=1, type=str, default='0')
    args = parser.parse_args()

    source_dir = r"C:\1CP_BinaryProblem\NC2\resnet18\run1\ISIC_0055459.jpg\\" #args.local_analysis_directory[0]
    number_prototypes_in_figure=9

    os.makedirs(os.path.join(source_dir, 'visualizations_of_expl'), exist_ok=True)

    pred, truth = read_local_analysis_log(os.path.join(source_dir + 'local_analysis.log'))

    anno_opts_cen = dict(xy=(0.4, 0.5), xycoords='axes fraction',
                    va='center', ha='center')
    anno_opts_symb = dict(xy=(1, 0.5), xycoords='axes fraction',
                    va='center', ha='center')
    anno_opts_sum = dict(xy=(0, -0.1), xycoords='axes fraction',
            va='center', ha='left')
    
    ###### all classes, one expl
    fig = plt.figure(constrained_layout=False)
    fig.set_size_inches(28, 12)

    ncols, nrows = 7, number_prototypes_in_figure #7,18
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)

    f_axes = []
    for row in range(nrows):
        f_axes.append([])
        for col in range(ncols):
            f_axes[-1].append(fig.add_subplot(spec[row, col]))

    plt.rcParams.update({'font.size': 14})

    for ax_num, ax in enumerate(f_axes[0]):
        if ax_num == 0:
            ax.set_title("Test image", fontdict=None, loc='left', color = "k")
        elif ax_num == 1:
            ax.set_title("Test image activation\nby prototype", fontdict=None, loc='left', color = "k")
        elif ax_num == 2:
            ax.set_title("Prototype", fontdict=None, loc='left', color = "k")
        elif ax_num == 3:
            ax.set_title("Self-activation of\nprototype", fontdict=None, loc='left', color = "k")
        elif ax_num == 4:
            ax.set_title("Similarity score", fontdict=None, loc='left', color = "k")
        elif ax_num == 5:
            ax.set_title("Class connection", fontdict=None, loc='left', color = "k")
        elif ax_num == 6:
            ax.set_title("Contribution", fontdict=None, loc='left', color = "k")
        else:
            pass

    plt.rcParams.update({'font.size': 25})

    for ax in [f_axes[r][4] for r in range(nrows)]:
        ax.annotate('x', **anno_opts_symb)

    for ax in [f_axes[r][5] for r in range(nrows)]:
        ax.annotate('=', **anno_opts_symb)

    # get and plot data from source directory

    orig_img = Image.open(os.path.join(source_dir + 'original_img.png'))

    for ax in [f_axes[r][0] for r in range(nrows)]:
        ax.imshow(orig_img)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    top_p_dir = os.path.join(source_dir + 'most_activated_prototypes')
    #sum_tc=0
    sum_tc=calculate_TC(top_p_dir=top_p_dir,number_prototypes=number_prototypes_in_figure)
    for top_p in range(nrows):
        # put info in place
        p_info_file = open(os.path.join(top_p_dir, f'top-{top_p+1}_activated_prototype.txt'), 'r')
        sim_score, cc_dict, class_str, top_cc_str,bias_number,logit_number = read_info(p_info_file)
        p_info_file.close()
        for ax in [f_axes[top_p][4]]:
            ax.annotate(sim_score, **anno_opts_cen)
            ax.set_axis_off()
        for ax in [f_axes[top_p][5]]:
            ax.annotate(top_cc_str + "\n" + class_str, **anno_opts_cen)
            ax.set_axis_off()
        for ax in [f_axes[top_p][6]]:
            tc = float(top_cc_str) * float(sim_score)
            #sum_tc+=tc
            ax.annotate('{0:.3f}'.format(tc) + "\n" + class_str, **anno_opts_cen)
            ax.set_axis_off()
        # put images in place
        p_img = Image.open(os.path.join(top_p_dir, f'top-{top_p+1}_activated_Ppatch_in_ogI.png')) 
        for ax in [f_axes[top_p][2]]:
            ax.imshow(p_img)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
        p_act_img = Image.open(os.path.join(top_p_dir, f'top-{top_p+1}_activated_prototype_self_act.png'))
        for ax in [f_axes[top_p][3]]:
            ax.imshow(p_act_img)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
        act_img = Image.open(os.path.join(top_p_dir, f'prototype_activation_map_by_top-{top_p+1}_prototype_normed.png'))
        for ax in [f_axes[top_p][1]]:
            ax.imshow(act_img)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
    #summary
    if(logit_number>0):  #TC + BIAS={logit_number}>0.
        f_axes[nrows-1][4].annotate(f"This {classname_dict[int(truth)]} lesion is classified as {classname_dict[int(pred)]}.\n Because TC+B = {round(sum_tc,1)} + {round(bias_number,1)} > 0.", **anno_opts_sum) #{sum_tc}+{bias_number}
    else:
        f_axes[nrows-1][4].annotate(f"This {classname_dict[int(truth)]} lesion is classified as {classname_dict[int(pred)]}.\n Because TC+B = {round(sum_tc,1)} + {round(bias_number,1)} < 0.", **anno_opts_sum) # {sum_tc}+{bias_number}

    save_loc1 = os.path.join(source_dir, 'visualizations_of_expl') + f'/all_class_r'+str(nrows)+'.png'
    plt.savefig(save_loc1, bbox_inches='tight', pad_inches=0)
    return

def read_local_analysis_log(file_loc):
    log_file = open(file_loc, 'r')
    for _ in range(30):
        line = log_file.readline()
        if line[0:len("Predicted: ")] == "Predicted: ":
            pred = line[len("Predicted: "):]
        elif line[0:len("Actual: ")] == "Actual: ":
            actual = line[len("Actual: "):]

    log_file.close()
    return pred, actual


def read_info(info_file, per_class=False):
    sim_score_line = info_file.readline()
    proto_index_line = info_file.readline()
    cc_0_line = info_file.readline()
    bias_line=info_file.readline()
    logit_line=info_file.readline()

    

    sim_score = sim_score_line[len("similarity: "):-1]

    circ_cc_str = cc_0_line[len('proto connection to class 0:tensor('):-(len(", device='cuda:0', grad_fn=<SelectBackward>)")+2)]
    print(circ_cc_str)
    circ_cc = float(circ_cc_str)

    bias_string=bias_line[len('last layer bias:tensor('):-(len(", device='cuda:0', grad_fn=<SelectBackward0>)")+1)]
    print(bias_string)
    bias_number=float(bias_string)

    logit_string=logit_line[len("Total Melanoma and only logit:tensor(["):-(len("], device='cuda:0', grad_fn=<SqueezeBackward0>)")+1)]
    print(logit_string)
    logit_number=float(logit_string)

    cc_dict = dict()
    cc_dict[0] = circ_cc    
    class_of_p = 0


    class_str = classname_dict[class_of_p]
    top_cc_str = circ_cc_str

    
    return sim_score, cc_dict, class_str, top_cc_str,bias_number,logit_number

def test():

    im = Image.open('./visualizations_of_expl/' + 'original_img.png')

    fig = plt.figure(constrained_layout=False)
    fig.set_size_inches(28, 12)

    ncols, nrows = 7, 3
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)

    f_axes = []
    for row in range(nrows):
        f_axes.append([])
        for col in range(ncols):
            f_axes[-1].append(fig.add_subplot(spec[row, col]))

    plt.rcParams.update({'font.size': 15})

    for ax_num, ax in enumerate(f_axes[0]):
        if ax_num == 0:
            ax.set_title("Test image", fontdict=None, loc='left', color = "k")
        elif ax_num == 1:
            ax.set_title("Test image activation by prototype", fontdict=None, loc='left', color = "k")
        elif ax_num == 2:
            ax.set_title("Prototype", fontdict=None, loc='left', color = "k")
        elif ax_num == 3:
            ax.set_title("Self-activation of prototype", fontdict=None, loc='left', color = "k")
        elif ax_num == 4:
            ax.set_title("Similarity score", fontdict=None, loc='left', color = "k")
        elif ax_num == 5:
            ax.set_title("Class connection", fontdict=None, loc='left', color = "k")
        elif ax_num == 6:
            ax.set_title("Contribution", fontdict=None, loc='left', color = "k")
        else:
            pass

    plt.rcParams.update({'font.size': 22})

    for ax in [f_axes[r][0] for r in range(nrows)]:
        ax.imshow(im)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])


    anno_opts = dict(xy=(0.4, 0.5), xycoords='axes fraction',
                    va='center', ha='center')

    anno_opts_symb = dict(xy=(1, 0.5), xycoords='axes fraction',
                    va='center', ha='center')

    for ax in [f_axes[r][s] for r in range(nrows) for s in range(4,7)]:
        ax.annotate('Number', **anno_opts)
        ax.set_axis_off()

    for ax in [f_axes[r][4] for r in range(nrows)]:
        ax.annotate('x', **anno_opts_symb)

    for ax in [f_axes[r][5] for r in range(nrows)]:
        ax.annotate('=', **anno_opts_symb)

    os.makedirs('./visualizations_of_expl/', exist_ok=True)
    plt.savefig('./visualizations_of_expl/' + 'test.png')

    # Refs: https://stackoverflow.com/questions/40846492/how-to-add-text-to-each-image-using-imagegrid
    # https://stackoverflow.com/questions/41793931/plotting-images-side-by-side-using-matplotlib

def calculate_TC(top_p_dir,number_prototypes):
    sum_tc=0
    for top_p in range(number_prototypes):
        p_info_file = open(os.path.join(top_p_dir, f'top-{top_p+1}_activated_prototype.txt'), 'r')
        sim_score, cc_dict, class_str, top_cc_str,bias_number,logit_number = read_info(p_info_file)
        p_info_file.close()
        tc = float(top_cc_str) * float(sim_score)
        sum_tc+=tc
    return sum_tc

if __name__ == "__main__":
    main()
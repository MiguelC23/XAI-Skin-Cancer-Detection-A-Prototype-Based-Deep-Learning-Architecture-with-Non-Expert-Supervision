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
classname_dict[0] = "AK"
classname_dict[1] = "BCC"
classname_dict[2] = "BKL"
classname_dict[3] = "DF"
classname_dict[4] = "MEL"
classname_dict[5] = "NV"
classname_dict[6] = "SCC"
classname_dict[7] = "VASC"


def main():
    # get dir
    parser = argparse.ArgumentParser()
    parser.add_argument('-local_analysis_directory', nargs=1, type=str, default='0')
    args = parser.parse_args()

    source_dir = r"C:\ACP_BM_Problem\NC8\resnet18\run1\ISIC_0026092.jpg\\" #args.local_analysis_directory[0]
    number_prototypes_in_figure=5

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
    for top_p in range(nrows):
        # put info in place
        p_info_file = open(os.path.join(top_p_dir, f'top-{top_p+1}_activated_prototype.txt'), 'r')
        sim_score, cc_dict, class_str, top_cc_str = read_info(p_info_file)
        p_info_file.close()
        for ax in [f_axes[top_p][4]]:
            ax.annotate(sim_score, **anno_opts_cen)
            ax.set_axis_off()
        for ax in [f_axes[top_p][5]]:
            ax.annotate(top_cc_str + "\n" + class_str, **anno_opts_cen)
            ax.set_axis_off()
        for ax in [f_axes[top_p][6]]:
            tc = float(top_cc_str) * float(sim_score)
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
    f_axes[nrows-1][4].annotate(f"This {classname_dict[int(truth)]} lesion is classified as {classname_dict[int(pred)]}.", **anno_opts_sum)

    save_loc1 = os.path.join(source_dir, 'visualizations_of_expl') + f'/all_class'+str(nrows)+'.png'
    plt.savefig(save_loc1, bbox_inches='tight', pad_inches=0)
    #os.makedirs('./visualizations_of_expl/', exist_ok=True)
    #save_loc2 = './visualizations_of_expl/' + str(source_dir.replace('/', '__'))[len('__usr__xtmp__IAIABL__saved_models__0129_pushonall_topkk=9_fa=0.001_random=4__pruned_prototypes_epoch50_k6_pt3__'):] + f'all_class.png'
    #shutil.copy2(save_loc1, save_loc2)
    #print(f"Saved in {save_loc2}")
    return

def read_local_analysis_log(file_loc):
    log_file = open(file_loc, 'r')
    for _ in range(30):
        line = log_file.readline()
        if line[0:len("Predicted: ")] == "Predicted: ":
            pred = line[len("Predicted: "):]
        elif line[0:len("Actual: ")] == "Actual: ":
            actual = line[len("Actual: "):]
    # pred = log_file.readline()[len("Predicted: "):]
    # actual = log_file.readline()[len("Actual: "):]
    log_file.close()
    return pred, actual


def read_info(info_file, per_class=False):
    sim_score_line = info_file.readline()
    connection_line = info_file.readline()
    proto_index_line = info_file.readline()
    cc_0_line = info_file.readline()
    cc_1_line = info_file.readline()
    cc_2_line = info_file.readline()
    cc_3_line = info_file.readline()
    cc_4_line = info_file.readline()
    cc_5_line = info_file.readline()
    cc_6_line = info_file.readline()
    cc_7_line = info_file.readline()

    

    sim_score = sim_score_line[len("similarity: "):-1]
    if per_class:
        cc = connection_line[len('last layer connection: '):-1]
    else:
        cc = connection_line[len('last layer connection with predicted class: '):-1]
    _0_cc_str = cc_0_line[len('proto connection to class 0:tensor('):-(len(", device='cuda:0', grad_fn=<SelectBackward>)")+2)]
    _0_cc = float(_0_cc_str)

    _1_cc_str = cc_1_line[len('proto connection to class 1:tensor('):-(len(", device='cuda:0', grad_fn=<SelectBackward>)")+2)]
    _1_cc = float(_1_cc_str)
   
    _2_cc_str = cc_2_line[len('proto connection to class 2:tensor('):-(len(", device='cuda:0', grad_fn=<SelectBackward>)")+2)]
    _2_cc = float(_2_cc_str)

    _3_cc_str = cc_3_line[len('proto connection to class 3:tensor('):-(len(", device='cuda:0', grad_fn=<SelectBackward>)")+2)]
    _3_cc = float(_3_cc_str)

    _4_cc_str = cc_4_line[len('proto connection to class 4:tensor('):-(len(", device='cuda:0', grad_fn=<SelectBackward>)")+2)]
    _4_cc = float(_4_cc_str)

    _5_cc_str = cc_5_line[len('proto connection to class 5:tensor('):-(len(", device='cuda:0', grad_fn=<SelectBackward>)")+2)]
    _5_cc = float(_5_cc_str)

    _6_cc_str = cc_6_line[len('proto connection to class 6:tensor('):-(len(", device='cuda:0', grad_fn=<SelectBackward>)")+2)]
    _6_cc = float(_6_cc_str)

    _7_cc_str = cc_7_line[len('proto connection to class 7:tensor('):-(len(", device='cuda:0', grad_fn=<SelectBackward>)")+2)]
    _7_cc = float(_7_cc_str)



    cc_dict = dict()
    cc_dict[0] = _0_cc
    cc_dict[1] = _1_cc
    cc_dict[2] = _2_cc
    cc_dict[3] = _3_cc
    cc_dict[4] = _4_cc
    cc_dict[5] = _5_cc
    cc_dict[6] = _6_cc
    cc_dict[7] = _7_cc
    
    class_of_p = max(cc_dict, key=lambda k: cc_dict[k])
    top_cc = cc_dict[class_of_p]

    class_str = classname_dict[class_of_p]
    if class_of_p == 0:
        top_cc_str = _0_cc_str
    elif class_of_p == 1:
        top_cc_str = _1_cc_str
    elif class_of_p == 2:
        top_cc_str = _2_cc_str
    elif class_of_p == 3:
        top_cc_str = _3_cc_str
    elif class_of_p == 4:
        top_cc_str = _4_cc_str
    elif class_of_p == 5:
        top_cc_str = _5_cc_str
    elif class_of_p == 6:
        top_cc_str = _6_cc_str
    elif class_of_p == 7:
        top_cc_str = _7_cc_str
    else:
        print("Error. The maximum value class is not found.")
    
    return sim_score, cc_dict, class_str, top_cc_str


if __name__ == "__main__":
    main()
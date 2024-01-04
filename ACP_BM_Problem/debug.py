import os
import re
import shutil
import random
from os.path import join as pjoin
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from helpers import makedir
import torch


def interactive_debugging(path: str,
                       selected_classes_idx: list,
                       n_images_to_show: int = 5,num_classes: int = 2):

    conf_dest_folder = pjoin(path, '..', 'tmp_forbidden_conf')
    remember_dest_folder = pjoin(path, '..', 'tmp_remember_patch')
    images_debug_user_folder=pjoin(path, '..', 'user_images')
    makedir(conf_dest_folder)
    makedir(remember_dest_folder)
    makedir(images_debug_user_folder)
    n_confounded_patch, n_no_confounded_path, n_remember_patch = 0, 0, 0


    if selected_classes_idx is not None:
        print(f'Warning: show only {selected_classes_idx}')
    img_class_idxs = np.load(os.path.join(path, "full_class_id.npy"))
    print(img_class_idxs.shape)
    num_prototypes, _ = img_class_idxs.shape

    assert(num_prototypes % num_classes == 0)
    # a onehot indication matrix for each prototype's class identity
    prototype_class_identity = torch.zeros(num_prototypes,num_classes)
    num_prototypes_per_class =num_prototypes // num_classes
    for j in range(num_prototypes):
            prototype_class_identity[j, j // num_prototypes_per_class] = 1

    proto_class_identity=prototype_class_identity.detach().numpy()
    max_k = min(img_class_idxs.shape[1], n_images_to_show)

    list_unique_names=[]

    for proto_idx, row in zip(range(0, num_prototypes), img_class_idxs):
        proto_class_idx = np.where(proto_class_identity[proto_idx] == 1)[0][0]
        if selected_classes_idx is not None:
            if proto_class_idx not in selected_classes_idx:
                continue
        
        for k in range(1, max_k + 1):
            print('================================')
            print(f'cl={proto_class_idx} pr={proto_idx} i={k}')
            img_idx=k
            "Images from global analysis"
            img_Ni_hap__in_og_img_fromP=Image.open(os.path.join(path,str(proto_idx),'nearest-'+str(img_idx)+'_high_act_patch_in_og_img.png'))
            img_Ni_ogPatch_inI_fromP=Image.open(os.path.join(path,str(proto_idx),'nearest-'+str(img_idx)+'_ogPatch_inI.png'))
            img_Ni_ogwh_fromP=Image.open(os.path.join(path,str(proto_idx),'nearest-'+str(img_idx)+'_original_with_heatmap.png'))
            img_Ni_ogPatch_fromP=Image.open(os.path.join(path,str(proto_idx),'nearest-'+str(img_idx)+'_ogPatch.png'))

            original_image_where_patch_is=Image.open(os.path.join(path,str(proto_idx),'nearest-'+str(img_idx)+'_original.png'))


            # Loop through all files in the directory
            for filename in os.listdir(os.path.join(path,str(proto_idx))):
                # Check if the file matches the pattern
                if filename.startswith('nearest-'+str(img_idx)+'_latent_patch') and filename.endswith(".npy"):
                    # If it does, print the filename
                    latent_name=filename


            latent_patch_numpy=np.load(os.path.join(path,str(proto_idx),latent_name))
            regex = r"ISIC_\d+_\d+"
            match = re.search(regex, latent_name)
            if match:
                unique_name= match.group()

            print(latent_name)
            print(unique_name)

            if unique_name not in list_unique_names:
                list_unique_names.append(unique_name)
    
                img_w, img_h = img_Ni_ogPatch_fromP.size
                from settings import img_size
                background = Image.new(mode="RGB",
                                        size=(img_size,img_size),
                                        color='white')
                bg_w, bg_h = background.size
                offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
                background.paste(img_Ni_ogPatch_fromP, offset)

                fig, axs = plt.subplots(nrows=1, ncols=4)
                fig.subplots_adjust(hspace=0.5,wspace=0.9)

                axs[0].set_title(f'From Global Analysis\n'
                                f'Nearest {img_idx} high activated patch in og img\n'
                                f'The bb is set containing 95% of the act.\n',fontsize=10)
                axs[1].set_title(f'From Global Analysis\n'
                                f'Nearest {img_idx} high activated TRUE patch in og img\n'
                                f'The bb is set according to the true patch size.\n',fontsize=10)
                axs[2].set_title(f'From Global Analysis\n'
                                f'Nearest {img_idx} og img with heatmap\n',fontsize=10)
                axs[3].set_title(f'Patch to decide if it is confounded\n'
                                f'Class:{proto_class_idx}, Proto:{proto_idx}, i:{img_idx}\n',fontsize=10)
                
                
                axs[0].imshow(img_Ni_hap__in_og_img_fromP)
                axs[0].axis('off')

                axs[1].imshow(img_Ni_ogPatch_inI_fromP)
                axs[1].axis('off')

                axs[2].imshow(img_Ni_ogwh_fromP)
                axs[2].axis('off')

                # Display the first image in the first column
                axs[3].imshow(background)
                axs[3].axis('off')
                plt.show(block=False)

                img_Ni_ogPatch_inI_fromP.save(pjoin(images_debug_user_folder, 'C'+str(proto_class_idx)+'_P'+str(proto_idx)+'_i'+str(img_idx)+unique_name+'.png'))

                while True:
                    select = input(
                        f'is this patch from image c={proto_class_idx} i={img_idx} confounded? '
                        f'[y (confounded) |n (no, go next) |r (remember)] ')
                    if select == 'y':
                        n_confounded_patch += 1
                        #np.save(pjoin(conf_dest_folder, 'C'+str(proto_class_idx)+'_P'+str(proto_idx)+'_i'+str(img_idx)+unique_name+'.npy'), latent_patch_numpy)
                        original_image_where_patch_is.save(pjoin(conf_dest_folder, 'C'+str(proto_class_idx)+'_P'+str(proto_idx)+'_i'+str(img_idx)+unique_name+'.png'))
                        makedir(pjoin(images_debug_user_folder, 'forbidden',str(proto_class_idx)))
                        img_Ni_ogPatch_inI_fromP.save(pjoin(images_debug_user_folder, 'forbidden',str(proto_class_idx),'C'+str(proto_class_idx)+'_P'+str(proto_idx)+'_i'+str(img_idx)+unique_name+'.png'))
                        break
                    elif select == 'n':
                        n_no_confounded_path += 1
                        break
                    elif select == 'r':
                        n_remember_patch += 1
                        #np.save(pjoin(remember_dest_folder, 'C'+str(proto_class_idx)+'_P'+str(proto_idx)+'_i'+str(img_idx)+unique_name+'.npy'), latent_patch_numpy)
                        original_image_where_patch_is.save(pjoin(remember_dest_folder, 'C'+str(proto_class_idx)+'_P'+str(proto_idx)+'_i'+str(img_idx)+unique_name+'.png'))
                        makedir(pjoin(images_debug_user_folder, 'remember',str(proto_class_idx)))
                        img_Ni_ogPatch_inI_fromP.save(pjoin(images_debug_user_folder, 'remember',str(proto_class_idx),'C'+str(proto_class_idx)+'_P'+str(proto_idx)+'_i'+str(img_idx)+unique_name+'.png'))
                        break
                    else:
                        print(f'Wrong value {select}')
                        continue
                
                plt.close(fig)

    print(f'\nStats: confounded={n_confounded_patch} '
            f'no={n_no_confounded_path} '
            f'remember={n_remember_patch}')
    
    print(f'\n Number of unique patches Debug = {len(list_unique_names)}')

def move_patches_to_forbidden_remember_folder(path):
    from settings import forbidden_protos_directory,remembering_protos_directory
    conf_dest_folder = pjoin(path, 'tmp_forbidden_conf')
    remember_dest_folder = pjoin(path, 'tmp_remember_patch')
    for source_folder, dest_folder, type in [
        (conf_dest_folder,forbidden_protos_directory, 'confound'),
        (remember_dest_folder,remembering_protos_directory, 'remember')
    ]:
        if not os.path.exists(source_folder):
            continue
        for patch in os.listdir(source_folder):
            #patch_numpy=C0_P5_i1.png"
            print(f'\nSource Numpy file {patch}')            
            if patch.endswith('.png'):
                class_idx, _, _ = re.findall("\d+", patch)[:3]
                print(f'Class of {patch} is {class_idx}')

                makedir(pjoin(dest_folder, f'class_idx_{class_idx}'))
                dst_path=pjoin(dest_folder, f'class_idx_{class_idx}')
                print(f'New destiny folder of {patch} is {dst_path}')

                pattern = r"ISIC_\d+_\d+"
                unique_name= re.search(pattern, patch).group()
                print(unique_name)


                save_name = f'{unique_name}.png'
                print(f'New name is {save_name}\n')

                shutil.copyfile(src=pjoin(source_folder,patch),
                                   dst=pjoin(dest_folder,
                                            f'class_idx_{class_idx}', save_name))

    



if __name__ == '__main__':
    from settings import random_seed_number
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.manual_seed(random_seed_number)
    torch.cuda.manual_seed(random_seed_number)
    np.random.seed(random_seed_number)
    random.seed(random_seed_number)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True

    #'path_to_folder', type=str, help='path to the folder containing the most activated patches for each prototype'
    path_to_folder=r"C:\ACP_BM_Problem\NC2\resnet18\run1\20_2push0.8295_nearest_train"
    # path_to_model_dir path to the folder where is the model
    path_to_model_dir=r"C:\ACP_BM_Problem\NC2\resnet18\run1"
    
    interactive= False #Put this True first, after run it again but with false
    move=not interactive

    num_classes=2 # Number total of classes, in our case 2 when MEL VS NEVUS or 8 when all classes from ISIC 2019
    #classes is a list with 'the (0-based) index of the classes whose prototypes you want to debug'
    classes=list(range(num_classes))#if you want to debug specif class do something like this [0,4] but do not change the num_classes because it is the number of total classes
    #'-n-img', type=int, default=10, help='number of nearest patches to show for each prototype'
    n_img=10

    if(interactive==True):
        interactive_debugging(path_to_folder,classes,n_img,num_classes=num_classes)
    elif(move==True):
        move_patches_to_forbidden_remember_folder(path_to_model_dir)
        
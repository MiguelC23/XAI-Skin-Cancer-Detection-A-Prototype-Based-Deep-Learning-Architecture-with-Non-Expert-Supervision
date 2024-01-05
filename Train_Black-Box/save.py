import os
import torch

def save_model_w_condition(model, model_dir, model_name, ba, target_ba, log=print):
    '''
    model: this is not the multigpu model
    '''
    if ba > target_ba:
        log('\tBA above {0:.2f}%'.format(target_ba * 100))
        # torch.save(obj=model.state_dict(), f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))
        torch.save(obj=model, f=os.path.join(model_dir, (model_name + '_{0:.4f}.pth').format(ba)))

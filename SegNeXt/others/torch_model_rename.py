import torch
import copy
dict = torch.load('/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/SegNeXt_ADE20k/BS16_Tiny_cr4_[comp_up_hf]_[localsim_dlation=2_GN_simoffset_featscale_sigmoid1.0_G4]_freqfusion2encoder_53_up10xlr_160k=43.64/best_mIoU_iter_160000.pth')
print(dict.keys())
new_dict = copy.deepcopy(dict)

model_state_dict_string = 'state_dict'
# model_state_dict_string = 'model'
for k in dict[model_state_dict_string]:
    print(k)
    if 'hamming' in k:
        # new_dict[model_state_dict_string].pop(k)
        pass
    if '.scope.' in k:
        new_dict[model_state_dict_string].pop(k)
        new_dict[model_state_dict_string][k.replace('.scope.', '.direct_scale.')] = dict[model_state_dict_string][k]
    if '.hr_scope.' in k:
        new_dict[model_state_dict_string].pop(k)
        new_dict[model_state_dict_string][k.replace('.hr_scope.', '.hr_direct_scale.')] = dict[model_state_dict_string][k]

torch.save(new_dict, '/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/SegNeXt_ADE20k/BS16_Tiny_cr4_[comp_up_hf]_[localsim_dlation=2_GN_simoffset_featscale_sigmoid1.0_G4]_freqfusion2encoder_53_up10xlr_160k=43.64/best_mIoU_iter_160000_renamed.pth')
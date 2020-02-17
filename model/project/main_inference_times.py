import torch
from torch2trt import torch2trt
from torch2trt import TRTModule
import os 
from model_fitting.evaluate import accuracy
from data_loader.dataset_creator import DatasetCreator

model = torch.load(os.path.join('checkpoints', 'checkpoints.pth'))
model.cuda()
model.eval()

trt_model_path = os.path.join('checkpoints', 'trtcheckpoints.pth')
model_trt_load = TRTModule()
model_trt_load.load_state_dict(torch.load(trt_model_path))

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dataset_creator = DatasetCreator(root_dir='./dataset')
validationset = dataset_creator.get_validation_iterator()
validationloader = torch.utils.data.DataLoader(validationset, batch_size=1, shuffle=False, num_workers=0)

print(accuracy(model, validationloader))
print(accuracy(model_trt_load, validationloader))
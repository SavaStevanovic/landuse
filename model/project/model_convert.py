import torch
from torch2trt import torch2trt
from torch2trt import TRTModule
import os 

model = torch.load(os.path.join('checkpoints', 'checkpoints.pth'))
model.cuda()
model.eval()

x = torch.ones((16, 3, 128, 128)).cuda()
# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x], max_batch_size=1, fp16_mode=False, int8_mode=True)

trt_model_path = os.path.join('checkpoints', 'trtcheckpoints.pth')
torch.save(model_trt.state_dict(), trt_model_path)
model_trt_load = TRTModule()
model_trt_load.load_state_dict(torch.load(trt_model_path))
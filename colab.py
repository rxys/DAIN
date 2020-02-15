import os
from torch.autograd import Variable
import torch
import numpy as np
import networks
from my_args import args
from cv2 import imread, imwrite

import warnings
warnings.filterwarnings("ignore")

src = "./_frames"
results = "./_interpolated"
os.makedirs(results, exist_ok=True)


model = networks.__dict__[args.netName](channel=args.channels,
                            filter_size = args.filter_size ,
                            timestep=args.time_step,
                            training=False).cuda()

args.SAVED_MODEL = './model_weights/best.pth'
print("The testing model weight is: " + args.SAVED_MODEL)
pretrained_dict = torch.load(args.SAVED_MODEL)
model_dict = model.state_dict()
# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict)
# 3. load the new state dict
model.load_state_dict(model_dict)
# 4. release the pretrained dict for saving memory
pretrained_dict = []

model = model.eval() # deploy mode


save_which=args.save_which
dtype = args.dtype

frames = list(os.listdir(src))
for first, second in zip(frames, frames[1:]):
	result_name = f"{os.path.splitext(first)[0]}_{second}"
	print(result_name)
	arguments_strFirst = os.path.join(src, first)
	arguments_strSecond = os.path.join(src, second)
	arguments_strOut = os.path.join(results, result_name)

	X0 =  torch.from_numpy( np.transpose(imread(arguments_strFirst) , (2,0,1)).astype("float32")/ 255.0).type(dtype)
	X1 =  torch.from_numpy( np.transpose(imread(arguments_strSecond) , (2,0,1)).astype("float32")/ 255.0).type(dtype)

	y_ = torch.FloatTensor()

	assert (X0.size(1) == X1.size(1))
	assert (X0.size(2) == X1.size(2))
	assert (X0.size(0) == 3)

	intWidth = X0.size(2)
	intHeight = X0.size(1)
	channel = X0.size(0)

	if intWidth != ((intWidth >> 7) << 7):
		intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
		intPaddingLeft =int(( intWidth_pad - intWidth)/2)
		intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
	else:
		intWidth_pad = intWidth
		intPaddingLeft = 32
		intPaddingRight= 32

	if intHeight != ((intHeight >> 7) << 7):
		intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
		intPaddingTop = int((intHeight_pad - intHeight) / 2)
		intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
	else:
		intHeight_pad = intHeight
		intPaddingTop = 32
		intPaddingBottom = 32

	pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])

	torch.set_grad_enabled(False)
	X0 = Variable(torch.unsqueeze(X0,0))
	X1 = Variable(torch.unsqueeze(X1,0))
	X0 = pader(X0).cuda()
	X1 = pader(X1).cuda()

	y_s,offset,filter = model(torch.stack((X0, X1),dim = 0))
	y_ = y_s[save_which]

	X0 = X0.data.cpu().numpy()
	y_ = y_.data.cpu().numpy()
	offset = [offset_i.data.cpu().numpy() for offset_i in offset]
	filter = [filter_i.data.cpu().numpy() for filter_i in filter]  if filter[0] is not None else None
	X1 = X1.data.cpu().numpy()

	X0 = np.transpose(255.0 * X0.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
	y_ = np.transpose(255.0 * y_.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
	offset = [np.transpose(offset_i[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0)) for offset_i in offset]
	filter = [np.transpose(
		filter_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
		(1, 2, 0)) for filter_i in filter]  if filter is not None else None
	X1 = np.transpose(255.0 * X1.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))

	imwrite(arguments_strOut, np.round(y_).astype(np.uint8))

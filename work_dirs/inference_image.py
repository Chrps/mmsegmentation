from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv
import numpy as np
from PIL import Image
config_file = '/mmsegmentation/work_dirs/pascal_mos/config.py'
checkpoint_file = '/mmsegmentation/work_dirs/pascal_mos/iter_1000.pth'

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = '/mmsegmentation/data/JPEGImages/frame_0.jpg'  # or img = mmcv.imread(img), which will only load it once
img = mmcv.imread(img)
result = inference_model(model, img)
print(result)
# Assuming pred_sem_seg is your tensor
# Assuming result is your SegDataSample object
pred_sem_seg_tensor = result.pred_sem_seg.data.cpu().numpy()  # Accessing pred_sem_seg attribute and converting to numpy array

# Replace values
pred_sem_seg_tensor[pred_sem_seg_tensor == 2] = 75
pred_sem_seg_tensor[pred_sem_seg_tensor == 1] = 38

# Convert back to PIL Image
pred_sem_seg_img = Image.fromarray(pred_sem_seg_tensor[0].astype(np.uint8))

# Save the image
pred_sem_seg_img.save("/mmsegmentation/work_dirs/out/modified_pred_sem_seg.png")

# visualize the results in a new window
#show_result_pyplot(model, img, result, show=True)
# or save the visualization results to image files
# you can change the opacity of the painted segmentation map in (0, 1].
#show_result_pyplot(model, img, result, show=True, out_file='result.jpg', opacity=0.5)
# test a video and show the results
#video = mmcv.VideoReader('video.mp4')
#for frame in video:
#   result = inference_model(model, frame)
#   show_result_pyplot(model, frame, result, wait_time=1)
from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv
import numpy as np
from PIL import Image
import cv2
def overlay_mask(frame, mask, alpha=0.5):
    # Convert mask to grayscale
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # Create a copy of the frame to preserve the original
    result = frame.copy()
    
    # Use bitwise_and to copy the red and green channels from the mask where the mask is not black
    result = cv2.bitwise_or(result, mask, mask=cv2.bitwise_not(mask_gray))
    
    # Perform weighted add to overlay mask onto frame
    overlay = cv2.addWeighted(result, alpha, frame, 1 - alpha, 0)
    
    return overlay



config_file = '/mmsegmentation/work_dirs/pascal_mos/config.py'
checkpoint_file = '/mmsegmentation/work_dirs/pascal_mos/iter_1000.pth'

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:0')

# Video file
video_path = '/mmsegmentation/work_dirs/video.MOV'

output_video_path = '/mmsegmentation/work_dirs/out/new_model/video_output.mp4'

# Open the video file
video = mmcv.VideoReader(video_path)

# Define color mapping
color_mapping = {
    1: (0, 255, 0),  # Green for class 1 (0, 255, 0)
    2: (0, 0, 255)   # Red for class 2
}

fps = video.fps
size = (int(video.width), int(video.height))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, size)

for idx, frame in enumerate(video):
    # Perform inference
    result = inference_model(model, frame)
    
    # Access segmentation result
    pred_sem_seg_tensor = result.pred_sem_seg.data.cpu().numpy()
    
    # Replace values with color
    colored_mask = np.zeros((pred_sem_seg_tensor.shape[1], pred_sem_seg_tensor.shape[2], 3), dtype=np.uint8)
    for class_id, color in color_mapping.items():
        colored_mask[pred_sem_seg_tensor[0] == class_id] = color
    
    overlayed_frame = overlay_mask(frame, colored_mask)
    video_writer.write(overlayed_frame)

    # save image
    #cv2.imwrite(f'/mmsegmentation/work_dirs/out/frame_{idx}.png', overlayed_frame)
    
video_writer.release()
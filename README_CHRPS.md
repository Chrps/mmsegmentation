* Build: `./docker_build.sh`
* Run: `./docker_run.sh`
* In the container:
    * If not enough GPU memory, run: `export CUDA_VISIBLE_DEVICES=-1`
    * Train: `python3 /mmsegmentation/tools/train.py <CONFIG_FILE>`
        * e.g. `/mmsegmentation/work_dirs/pascal_mos/pascal_mos.py`
    * Restart training: `python3 /mmsegmentation/tools/train.py <CONFIG_FILE>  --resume --cfg-options load_from=${CHECKPOINT}`
        * e.g. `/mmsegmentation/work_dirs/pascal_mos/iter_500.pth`
    * Inference image: `python3 /mmsegmentation/work_dirs/inference_image.py` 
        * Update file to change input image and model used
    * Inference video: `python3 /mmsegmentation/work_dirs/inference_video.py` 
        * Update file to change input video and model used
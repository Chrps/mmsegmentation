# TODO: update to make paths more general
docker run --gpus all --shm-size=8g -it -v /media/chrps/wd_external/ambolt/PotentialProjects/Mos/mos_pascal_voc_final:/mmsegmentation/data -v /media/chrps/wd_external/ambolt/PotentialProjects/Mos/mmsegmentation/work_dirs:/mmsegmentation/work_dirs mmsegmentation

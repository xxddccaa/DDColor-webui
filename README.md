
https://www.dong-blog.fun/post/1931



webui 7862
```bash
cd /your/path/to/image_color

docker run --gpus device=2 \
--shm-size=32g \
-it \
--net host \
-v /ssd/xiedong/image_color:/ddcolor \
-v ./DDColormodel:/DDColormodel/ \
kevinchina/deeplearning:2.5.1-cuda12.1-cudnn9-devel-ddcolor-webui-metric bash


python ddcolor_app_lab.py --model_path /DDColormodel --port 7862
```



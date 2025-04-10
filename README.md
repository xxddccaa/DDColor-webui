
https://www.dong-blog.fun/post/1931


webui 7861
```bash
cd /your/path/to/image_color

docker run --gpus device=2 \
--shm-size=32g \
-it \
--net host \
-v ./ddcolor_app.py:/ddcolor/ddcolor_app.py \
-v ./DDColormodel:/DDColormodel/ \
kevinchina/deeplearning:2.5.1-cuda12.1-cudnn9-devel-ddcolor-webui-metric bash


python /ddcolor/ddcolor_app.py --model_path /DDColormodel --port 7861
```

webui 7862
```bash
cd /your/path/to/image_color

docker run --gpus device=2 \
--shm-size=32g \
-it \
--net host \
-v ./ddcolor_app.py:/ddcolor/ddcolor_app.py \
-v ./DDColormodel:/DDColormodel/ \
kevinchina/deeplearning:2.5.1-cuda12.1-cudnn9-devel-ddcolor-webui-metric bash


python /ddcolor/ddcolor_app.py --model_path /DDColormodel --port 7862
```


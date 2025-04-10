```bash
docker run --gpus device=2 \
--shm-size=32g \
-it \
--net host \
-v /ssd/xiedong/image_color:/ssd/xiedong/image_color \
kevinchina/deeplearning:2.5.1-cuda12.1-cudnn9-devel-ddcolor-webui bash
```

```bash
python ddcolor_inference.py \
--model_path /ssd/xiedong/image_color/DDColormodel/ \
--src_dir /ssd/xiedong/image_color/pytorch-CycleGAN-and-pix2pix/results/tongyong_l2ab_4/testA_35/images \
--dst_dir /ssd/xiedong/image_color/ddcolor_test
```

会生成图到/ssd/xiedong/image_color/ddcolor_test



python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ssd_mobilenet_v1_stamp_qr.config \
    --trained_checkpoint_prefix training/model.ckpt-30 \
    --output_directory stamp_graph

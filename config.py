class Config :
    img_dir = "/kaggle/input/datasets/jeffaudi/coco-2014-dataset-for-yolov3/coco2014/images/train2014"
    ann_file = "/kaggle/input/datasets/minhkhoai/seqtr-annotations-weights/annotations/refcoco-unc/instances.json"

    imsize = 640

    max_query_len = 15

    # Tham số cho BERT
    bert_model = "bert-base-uncased"
    bert_enc_num = 12

    backbone = "resnet50"

    dilation = False

    position_embedding = "sine" # sine hoặc learned

    # Tham số cho DETR
    detr_enc_num = 6
    hidden_dim = 256
    nheads = 8
    dim_feedforward = 2048
    dropout = 0.1
    pre_norm = False

    # Tham số cho Vison Language Transformer (Fusion)
    vl_hidden_dim = 256
    vl_nheads = 8
    vl_dim_feedforward = 2048
    vl_dropout = 0.1
    vl_enc_layers = 6

    # Tham số training 
    lr = 1e-4               # VL Transformer + MLP head (train mạnh)
    lr_bert = 1e-5          # BERT (fine-tune nhẹ)
    lr_visu_cnn = 1e-5      # ResNet backbone (fine-tune nhẹ)
    lr_visu_tra = 1e-5      # DETR Transformer Encoder (fine-tune nhẹ)

    # Optimizer
    optimizer = "adamw" # adamw, adam, sgd
    weight_decay = 1e-4

    # Traing schedule 
    batch_size = 8
    epochs = 30
    lr_scheduler = "step" # step, cosine, poly
    lr_drop = 60 # Epoch giảm lr

    clip_max_norm = 0.01    # Gradient clipping


    # Data augmentation
    aug_crop = True # Random crop
    aug_scale = True # Multi-scale resize
    aug_translate = True # Random translate
    avg_blur = True # Gaussian blur

    # Đường dẫn checkpoint và log
    detr_model = None
    output_dir = "/kaggle/working/transvg_outputs"

    # Resume training 
    resume = ""

    # MISC
    seed = 13
    num_workers = 2
    device = "cuda"

    # ImageNet normalization (chuẩn cho ResNet pretrained)
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
class DefaultConfig():
    #backbone
    pretrained=True
    freeze_stage_1=True
    freeze_bn=True

    #fpn
    use_yolof = False
    fpn_out_channels=256
    use_p5=True
    
    #head
    class_num=80
    use_GN_head=True
    prior=0.01
    add_centerness=True
    cnt_on_reg=True

    #training
    strides=[8,16,32,64,128]
    limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]]

    #inference
    score_threshold=0.05
    nms_iou_threshold=0.6
    max_detection_boxes_num=1000

class YolofDC5Config():
        #backbone
    pretrained=False
    freeze_stage_1=True
    freeze_bn=True

    #fpn
    use_yolof = True
    C5_inplanes = 512 * 4
    fpn_out_channels=512
    use_p5=True
    
    #head
    class_num=80
    use_GN_head=True
    prior=0.01
    add_centerness=True
    cnt_on_reg=True

    #training
    strides=[16]
    limit_range=[[-1,512]]
    scales = [1.0]

    #inference
    score_threshold=0.05
    nms_iou_threshold=0.6
    max_detection_boxes_num=1000
python test_models.py somethingv2 \
--VAP --batch_size=64 -j=4 --test_crops=1 --test_segments=8,8,8 \
--weights=pretrained/PAN_Lite_somethingv2_resnet50_shift8_blockres_avg_segment8_e80.pth.tar,pretrained/PAN_RGB_somethingv2_resnet50_shift8_blockres_avg_segment8_e50.pth.tar,pretrained/PAN_PA_somethingv2_resnet50_shift8_blockres_avg_segment8_e80.pth.tar \
--full_res --twice_sample

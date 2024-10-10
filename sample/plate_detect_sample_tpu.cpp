

#include "VP.h"

#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_plate_detect_node_tpu.h"
#include "../nodes/infers/vp_plate_ocr_node_tpu.h"
#include "../nodes/osd/vp_face_osd_node.h"
#include "../nodes/vp_screen_des_node.h"
#include "../nodes/vp_rtmp_des_node.h"

#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## 1-1-1 sample ##
* 1 video input, 1 infer task, and 1 output.
*/

#if _1_1_1_sample

int main() {
    VP_SET_LOG_INCLUDE_CODE_LOCATION(false);
    VP_SET_LOG_INCLUDE_THREAD_ID(false);
    VP_LOGGER_INIT();

    // create nodes
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "/dev/video0", 0.6);
    auto yunet_face_detector_0 = std::make_shared<vp_nodes::vp_plate_detect_node_tpu>("yunet_face_detector_0", "/home/zyw/ssdlite_ocr_edgetpu.tflite", "");
    auto sface_face_encoder_0 = std::make_shared<vp_nodes::vp_plate_ocr_node_tpu>("sface_face_encoder_0", "/home/zyw/lprnet_edgetpu.tflite", "");
    auto osd_0 = std::make_shared<vp_nodes::vp_face_osd_node>("osd_0");
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);

    // construct pipeline
    yunet_face_detector_0->attach_to({file_src_0});
    sface_face_encoder_0->attach_to({yunet_face_detector_0});
    osd_0->attach_to({sface_face_encoder_0});
    screen_des_0->attach_to({osd_0});

    file_src_0->start();

    // for debug purpose
    vp_utils::vp_analysis_board board({file_src_0});
    board.display();
}

#endif

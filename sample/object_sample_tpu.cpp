#include <python3.9/Python.h>
#include <iostream>
#include "VP.h"

#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_mobilenet_edgetpu_object_node.h"
#include "../nodes/infers/vp_sface_feature_encoder_node.h"
#include "../nodes/osd/vp_osd_node.h"
#include "../nodes/vp_screen_des_node.h"
#include "../nodes/vp_sever_control_node.h"
#include "../nodes/vp_split_node.h"
#include "../nodes/vp_rtmp_des_node.h"

#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## 1-1-1 sample ##
* 1 video input, 1 infer task, and 1 output.
*/

#if _1_1_1_sample

using namespace std;

int main() {
    VP_SET_LOG_INCLUDE_CODE_LOCATION(false);
    VP_SET_LOG_INCLUDE_THREAD_ID(false);
    VP_LOGGER_INIT();
    
    //Py_SetPythonHome(L"/usr/include/python3.9");
    Py_Initialize();
    if(!Py_IsInitialized()){
        cout << "python init fail" << endl;
        return 0;
    }
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(r'/home/zyw/Desktop/SERVO_DRIVER_WITH_ESP32/examples/SCServo_Python/sms_sts')");
    PyRun_SimpleString("sys.path.append(r'/home/zyw/Desktop/SERVO_DRIVER_WITH_ESP32/examples/SCServo_Python/scservo_sdk')");
    PyRun_SimpleString("sys.path.append(r'/home/zyw/Desktop/SERVO_DRIVER_WITH_ESP32/examples/SCServo_Python')");
    PyObject *pModule = NULL;
    //PyObject *pFunc = NULL;
    PyRun_SimpleString("import sync_write");
    pModule = PyImport_ImportModule("sync_write.initial_position()");
    Py_Finalize();

    // create nodes
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "/dev/video0", 0.6);
    auto split = std::make_shared<vp_nodes::vp_split_node>("split", false, true);  // split by deep-copy not by channel!
    
    auto yunet_face_detector_0 = std::make_shared<vp_nodes::vp_mobilenet_edgetpu_object_node>("yunet_face_detector_0", "/home/zyw/edgetpu-sample-master/edgetpu-object/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite", "", "", "/home/zyw/edgetpu-sample-master/edgetpu-object/coco_label.txt");
    //auto sface_face_encoder_0 = std::make_shared<vp_nodes::vp_sface_feature_encoder_node>("sface_face_encoder_0", "/home/zyw/models/face/face_recognition_sface_2021dec.onnx");
    auto osd_0 = std::make_shared<vp_nodes::vp_osd_node>("osd_0");
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);
    auto sever_control_0 = std::make_shared<vp_nodes::vp_sever_control_node>("sever_control_0", 0);

    // construct pipeline
    yunet_face_detector_0->attach_to({file_src_0});
    //sface_face_encoder_0->attach_to({yunet_face_detector_0});
    osd_0->attach_to({yunet_face_detector_0});
    split->attach_to({osd_0});
    sever_control_0->attach_to({split});
    screen_des_0->attach_to({split});
    

    file_src_0->start();

    // for debug purpose
    vp_utils::vp_analysis_board board({file_src_0});
    board.display();
}

#endif

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include <cmath>

#include <python3.9/Python.h>
#include <iostream>

#include "edgetpu_c.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include <core/session/onnxruntime_cxx_api.h>
#include <core/session/onnxruntime_c_api.h>

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

#include "vp_mobilenet_edgetpu_face_node.h"

using namespace cv;
using namespace cv::face;

namespace vp_nodes {
        
    vp_mobilenet_edgetpu_face_node::vp_mobilenet_edgetpu_face_node(std::string node_name, 
                                                            std::string model_path, 
                                                            std::string convert_type,
                                                            float score_threshold,
                                                            float nms_threshold, 
                                                            int top_k
                                                            ):
                                                            vp_primary_infer_node(node_name, model_path, convert_type),
                                                            scoreThreshold(score_threshold),
                                                            nmsThreshold(nms_threshold),
                                                            topK(top_k)
                                                            {
        this->initialized();
    }
    
    vp_mobilenet_edgetpu_face_node::~vp_mobilenet_edgetpu_face_node() {

    }
    
    //void vp_mobilenet_edgetpu_face_node::btm_angle(int img_width, float x, float width){
        //float face_x = x + width/2.0;
        //float offset_x = (face_x / img_width - 0.5) * 2;
        //if (std::fabs(offset_x) < 0.5) {
                //offset_x = 0;
        //}
        
    //}
    
    int vp_mobilenet_edgetpu_face_node::get_color(int c, int classNum){
        float colors[6][3] = {{1, 0, 1}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}};
        int offset = classNum * 123457 % 80;
        float ratio = ((float)classNum / 80) * 5;
        int i = floor(ratio);
        int j = ceil(ratio);
        ratio -= i;
        float ret = (1 - ratio) * colors[i][c] + ratio * colors[j][c];
        return ret * 255;
    }
            
   
}

    

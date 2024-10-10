
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>


#include "edgetpu_c.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#include <opencv2/opencv.hpp>

#include "vp_mobilenet_edgetpu_object_node.h"

#define INPUT_WIDTH 300
#define INPUT_HEIGHT 300
#define INPUT_CHANNEL 3

namespace vp_nodes {
        
    vp_mobilenet_edgetpu_object_node::vp_mobilenet_edgetpu_object_node(std::string node_name, 
                                                            std::string model_path,
							                                 std::string convert_type,
                                                            std::string model_config_path,
                                                            std::string labels_path,
                                                            float score_threshold,
                                                            float nms_threshold, 
                                                            int top_k
                                                            ):
                                                            vp_primary_infer_node(node_name, model_path, convert_type, model_config_path, labels_path),
                                                            scoreThreshold(score_threshold),
                                                            nmsThreshold(nms_threshold),
                                                            topK(top_k)
                                                            {
        this->initialized();
    }
    
    vp_mobilenet_edgetpu_object_node::~vp_mobilenet_edgetpu_object_node() {

    }
    
    int vp_mobilenet_edgetpu_object_node::get_color(int c, int classNum){
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

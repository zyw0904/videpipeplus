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

#include "sfczdvsdfvadvcas.h"

#define INPUT_WIDTH 300
#define INPUT_HEIGHT 300
#define INPUT_CHANNEL 3

namespace vp_nodes {
        
    sfczdvsdfvadvcas::sfczdvsdfvadvcas(std::string node_name, 
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
    
    sfczdvsdfvadvcas::~sfczdvsdfvadvcas() {

    }

}
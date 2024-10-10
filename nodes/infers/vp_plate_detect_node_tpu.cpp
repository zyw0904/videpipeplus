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

#include "vp_plate_detect_node_tpu.h"

namespace vp_nodes {
        
    vp_plate_detect_node_tpu::vp_plate_detect_node_tpu(std::string node_name, 
                                                            std::string model_path, 
							    std::string convert_type,
                                                            float score_threshold
                                                            ):
                                                            vp_primary_infer_node(node_name, model_path, convert_type),
                                                            scoreThreshold(score_threshold)
                                                            {
        this->initialized();
    }
    
    vp_plate_detect_node_tpu::~vp_plate_detect_node_tpu() {

    }
    
    
        
    void vp_plate_detect_node_tpu::postprocess(std::vector<std::vector<int>>& outputs_shape, std::vector<std::vector<float>>& tensor_vector, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) { 
        
        //int outputsize = interpreter->outputs().size();
        
       // detection_boxes = interpreter->tensor(interpreter->outputs()[0])->data.f;
        
        //const float* detection_class = interpreter->tensor(interpreter->outputs()[1])->data.f;
        //detection_score = interpreter->tensor(interpreter->outputs()[2])->data.f;
        //detection_number = interpreter->tensor(interpreter->outputs()[3])->data.f;
        
       // cout << detection_number[0] << endl;   
        //for(int i=0; i<4*detection_number[0]; i++){
            //cout << detection_boxes[i] << "," << detection_class[i] << "," << detection_score[i] << endl;
        //}
        
        detection_boxes = interpreter->tensor(interpreter->outputs()[0])->data.f;
        //detection_class = interpreter->tensor(interpreter->outputs()[1])->data.f;
        detection_score = interpreter->tensor(interpreter->outputs()[2])->data.f;
        detection_number = *interpreter->tensor(interpreter->outputs()[3])->data.f;
        assert(frame_meta_with_batch.size() == 1);
        auto& frame_meta = frame_meta_with_batch[0];
        inputW = frame_meta->frame.cols;
        inputH = frame_meta->frame.rows;
        cv::Mat faces;
        cv::Mat face(1, 5, CV_32FC1);
        for(int i = 0; i < detection_number; i++)
        {
                if(detection_score[i] > scoreThreshold)
                {
                        
                        float score = detection_score[i];
                        face.at<float>(0, 4) = score;
                        //auto det_index = static_cast<int>(raw_outputs[1].at<float>(0, i)) + 1;
                        //auto label = labels[det_index].c_str();
                        float y1 = detection_boxes[4*i+0] * inputH;
                        float x1 = detection_boxes[4*i+1] * inputW;
                        float y2 = detection_boxes[4*i+2] * inputH;
                        float x2 = detection_boxes[4*i+3] * inputW;
                        float width = x2 - x1;
                        float height = y2 - y1;
                        face.at<float>(0, 0) = x1;
                        face.at<float>(0, 1) = y1;
                        face.at<float>(0, 2) = width;
                        face.at<float>(0, 3) = height;
                        auto face_target = std::make_shared<vp_objects::vp_frame_face_target>(x1, y1, width, height, score);
                
                        frame_meta->face_targets.push_back(face_target);
                        //faces.push_back(face);
                }
        }
        
    }    
   
}

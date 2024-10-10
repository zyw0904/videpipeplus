
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

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

#include "vp_fire_dtect_node_tpu.h"

using namespace cv;
using namespace cv::face;

namespace vp_nodes {
        
    vp_fire_dtect_node_tpu::vp_fire_dtect_node_tpu(std::string node_name, 
                                               std::string model_path,
                                               std::string convert_type,
                                               std::string model_config_path,
                                               std::string labels_path,
                                               float score_threshold, 
                                               float nms_threshold, 
                                               int top_k):
                                                vp_primary_infer_node(node_name, model_path, convert_type, model_config_path, labels_path),
                                                scoreThreshold(score_threshold),
                                                nmsThreshold(nms_threshold),
                                                topK(top_k){
        this->initialized();
    }
    
    vp_fire_dtect_node_tpu::~vp_fire_dtect_node_tpu() {

    }
    
    //void vp_mobilenet_edgetpu_face_node::btm_angle(int img_width, float x, float width){
        //float face_x = x + width/2.0;
        //float offset_x = (face_x / img_width - 0.5) * 2;
        //if (std::fabs(offset_x) < 0.5) {
                //offset_x = 0;
        //}
        
    //}
    
    int vp_fire_dtect_node_tpu::get_color(int c, int classNum){
        float colors[6][3] = {{1, 0, 1}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}};
        int offset = classNum * 123457 % 80;
        float ratio = ((float)classNum / 80) * 5;
        int i = floor(ratio);
        int j = ceil(ratio);
        ratio -= i;
        float ret = (1 - ratio) * colors[i][c] + ratio * colors[j][c];
        return ret * 255;
    }
        
    void vp_fire_dtect_node_tpu::postprocess(std::vector<std::vector<int>>& outputs_shape, std::vector<std::vector<float>>& tensor_vector, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) { 
        
        float* data = tensor_vector[0].data();        
        assert(frame_meta_with_batch.size() == 1);
        auto& frame_meta = frame_meta_with_batch[0];
        inputW = frame_meta->frame.cols;
        inputH = frame_meta->frame.rows;
        
        Mat faces;
        Mat face(1, 7, CV_32FC1);
        
        for (size_t i = 0; i < outputs_shape[0][1]; ++i) {
            if (data[i*6+4] > scoreThreshold) {
                face.at<float>(0, 4) = data[i*6+4];
                float x = data[i*6+0]*640/inputW;
                float y = data[i*6+1]*640/inputH;
                float w = data[i*6+2]*640/inputW;
                float h = data[i*6+3]*640/inputH;
                float label = data[i*6+5];
                float x1 = x - w / 2;
                float y1 = y - h / 2;
                face.at<float>(0, 0) = x1;
                face.at<float>(0, 1) = y1;
                face.at<float>(0, 2) = w;
                face.at<float>(0, 3) = h;
                faces.push_back(face); 
                float score = face.at<float>(0, 4);
                cout << data[i*6+0]  << "," << data[i*6+1] << "," << data[i*6+2] << "," << data[i*6+3] << endl;
                auto face_target = std::make_shared<vp_objects::vp_frame_face_target>(x1, y1, w, h, score);

                frame_meta->face_targets.push_back(face_target);
            }
        }
        
        //if (faces.rows > 1)
        //{
            // Retrieve boxes and scores
            //std::vector<Rect2i> faceBoxes;
            //std::vector<float> faceScores;
            //for (int rIdx = 0; rIdx < faces.rows; rIdx++)
            //{
                //faceBoxes.push_back(Rect2i(int(faces.at<float>(rIdx, 0)),
                                           //int(faces.at<float>(rIdx, 1)),
                                           //int(faces.at<float>(rIdx, 2)),
                                           //int(faces.at<float>(rIdx, 3))));
                //faceScores.push_back(faces.at<float>(rIdx, 4));
            //}

            //std::vector<int> keepIdx;
            //dnn::NMSBoxes(faceBoxes, faceScores, scoreThreshold, nmsThreshold, keepIdx, 1.f, topK);

            // Get NMS results
            //Mat nms_faces;
            //for (int idx: keepIdx)
            //{
                //nms_faces.push_back(faces.row(idx));
            //}

            // insert face target back to frame meta
            //for (int i = 0; i < nms_faces.rows; i++) {
                //auto x = int(nms_faces.at<float>(i, 0));
                //auto y = int(nms_faces.at<float>(i, 1));
                //auto w = int(nms_faces.at<float>(i, 2));
                //auto h = int(nms_faces.at<float>(i, 3));
                
                // check value range
                //x = std::max(x, 0);
                //y = std::max(y, 0);
                //w = std::min(w, frame_meta->frame.cols - x);
                //h = std::min(h, frame_meta->frame.rows - y);

                //auto score = nms_faces.at<float>(i, 4);

                
            //}
            
        //}
        
        
    }    
   
}

    

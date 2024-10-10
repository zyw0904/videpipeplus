
#pragma once

#include "../vp_primary_infer_node.h"
#include "../../objects/vp_frame_face_target.h"

namespace vp_nodes {
    
    extern cv::Mat FRAME;
    
    class vp_plate_detect_node_tpu: public vp_primary_infer_node
    {
    private:
        // names of output layers in yunet
        
        float scoreThreshold = 0.85;
        float nmsThreshold = 0.6;
        int topK = 50;
        std::vector<cv::Mat> Mats;
        int inputW;
        int inputH;
        float* detection_boxes;
        float* detection_score;
        int detection_number;
        
    protected:
        // override infer and preprocess as yunet has a different logic
        //virtual void infer(const cv::Mat& blob_to_infer, std::vector<cv::Mat>& raw_outputs) override;
        //virtual void preprocess(const std::vector<cv::Mat>& mats_to_infer, cv::Mat& blob_to_infer) override;

        virtual void postprocess(std::vector<std::vector<int>>& outputs_shape, std::vector<std::vector<float>>& tensor_vector, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;
        
    public:
        vp_plate_detect_node_tpu(std::string node_name, std::string model_path, std::string convert_type, float score_threshold = 0.85);
        ~vp_plate_detect_node_tpu();
    };
    
}

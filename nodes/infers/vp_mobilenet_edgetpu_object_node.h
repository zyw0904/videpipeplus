#pragma once

#include "../vp_primary_infer_node.h"
#include "../../objects/vp_frame_target.h"

namespace vp_nodes {
    
    extern cv::Mat FRAME;
    
    class vp_mobilenet_edgetpu_object_node: public vp_primary_infer_node
    {
    private:
        // names of output layers in yunet
        
        float scoreThreshold = 0.7;
        float nmsThreshold = 0.6;
        int topK = 50;
        std::vector<cv::Mat> Mats;
        int inputW;
        int inputH;
        float* detection_boxes;;
        float* detection_score;
        float* detection_class;
        int detection_number;
        std::vector<std::string> Labels;
        int get_color(int c, int classNum);
        bool code = false;
        
    protected:
        // override infer and preprocess as yunet has a different logic
        //virtual void infer(const cv::Mat& blob_to_infer, std::vector<cv::Mat>& raw_outputs) override;
        //virtual void preprocess(const std::vector<cv::Mat>& mats_to_infer, cv::Mat& blob_to_infer) override;

        //virtual void postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;
        
    public:
        vp_mobilenet_edgetpu_object_node(std::string node_name, std::string model_path, std::string convert_type = "", std::string model_config_path = "", std::string labels_path = "", float score_threshold = 0.7, float nms_threshold = 0.6, int top_k = 50);
        ~vp_mobilenet_edgetpu_object_node();
    };
    
}

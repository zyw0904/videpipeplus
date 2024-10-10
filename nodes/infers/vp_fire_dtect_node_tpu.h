#pragma once

#include "../vp_primary_infer_node.h"
#include "../../objects/vp_frame_face_target.h"

namespace vp_nodes {
    
    extern cv::Mat FRAME;
    
    class vp_fire_dtect_node_tpu: public vp_primary_infer_node
    {
    private:
        // names of output layers in yunet
        
        float scoreThreshold = 0.2;
        float nmsThreshold = 0.5;
        int topK = 50;
        int inputW;
        int inputH;

        //std::vector<std::vector<Point2f>> landmarks;
        int get_color(int c, int classNum);
        std::vector<string> ReadLabels(const string& filename);
        //string GetLabel(const vector<string>& labels, int label);
        
    protected:
        // override infer and preprocess as yunet has a different logic
        //virtual void infer(const cv::Mat& blob_to_infer, std::vector<cv::Mat>& raw_outputs) override;
        //virtual void preprocess(const std::vector<cv::Mat>& mats_to_infer, cv::Mat& blob_to_infer) override;
        virtual void postprocess(std::vector<std::vector<int>>& outputs_shape, std::vector<std::vector<float>>& tensor_vector, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;
        
    public:
        vp_fire_dtect_node_tpu(std::string node_name, std::string model_path, std::string convert_type = "", std::string model_config_path = "", std::string labels_path = "", float score_threshold = 0.3, float nms_threshold = 0.5, int top_k = 50);
        ~vp_fire_dtect_node_tpu();
    };
    
}

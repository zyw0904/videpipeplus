
#pragma once

#include "../vp_secondary_infer_node.h"
#include "../../objects/vp_sub_target.h"

namespace vp_nodes {
    // vehicle plate detector based on tensorrt
    // source code: ../../third_party/trt_vehicle
    // note:
    // this class is not based on opencv::dnn module but tensorrt, a few data members declared in base class are not usable any more(just ignore), such as vp_infer_node::net.
    class vp_plate_ocr_node_tpu: public vp_secondary_infer_node
    {
    private:
        /* data */
        std::vector<cv::Mat> Mats;

        
    protected:
        // we need a totally new logic for the whole infer combinations
        // no separate step pre-defined needed in base class
        //virtual void infer(const cv::Mat& blob_to_infer, std::vector<cv::Mat>& raw_outputs) override;
        //virtual void preprocess(const std::vector<cv::Mat>& mats_to_infer, cv::Mat& blob_to_infer) override;
        virtual void run_infer_combinations(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;
        // override pure virtual method, for compile pass
        virtual void postprocess(std::vector<std::vector<int>>& outputs_shape, std::vector<std::vector<float>>& tensor_vector, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;
    public:
        vp_plate_ocr_node_tpu(std::string node_name, std::string model_path, std::string convert_type);
        ~vp_plate_ocr_node_tpu();
    };

}

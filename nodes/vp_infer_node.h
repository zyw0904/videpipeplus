

#pragma once
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

#include "edgetpu_c.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include <core/session/onnxruntime_cxx_api.h>
#include <core/session/onnxruntime_c_api.h>

#include "openvino/openvino.hpp"

#include "/home/zyw/openvino/install/samples/cpp/common/utils/include/samples/common.hpp"
//#include "format_reader_ptr.h"

#include "vp_node.h"

using namespace cv;
using namespace cv::face;

namespace vp_nodes {

    extern cv::Mat FRAME;
    // infer type
    // infer on the whole frame or small cropped image?
    enum vp_infer_type {
        PRIMARY,      // infer on the whole frame, like detector, pose estimatation
        SECONDARY     // infer on small cropped image, like classifier, feature extractor and secondary detector which need detect on small cropped images.
    };

    // base class for infer node, can't be instanstiated directly. 
    // note: 
    // the class is based on opencv::dnn module which is the default way for all deep learning inference in code, 
    // we can implement it using other backends such as tensorrt with cuda acceleration, see vp_ppocr_text_detector_node which is based on PaddlePaddle dl framework from BaiDu corporation.
    class vp_infer_node: public vp_node {
    private:
        // load labels if need
        void load_labels();
        std::vector<cv::Mat> Mats;

    protected:
        vp_infer_type infer_type;
        std::string model_path;
        std::string model_config_path;
	    std::string convert_type;
        std::string labels_path;
        int input_width;
        int input_height;
        int batch_size;
        cv::Scalar mean;
        cv::Scalar std;
        float scale;
        bool swap_rb;

        // protected as it can't be instanstiated directly.
        vp_infer_node(std::string node_name, 
                    vp_infer_type infer_type, 
                    std::string model_path, 
		            std::string convert_type = "",
                    std::string model_config_path = "", 
                    std::string labels_path = "", 
                    int input_width = 128, 
                    int input_height = 128, 
                    int batch_size = 1,
                    float scale = 1.0,
                    cv::Scalar mean = cv::Scalar(123.675, 116.28, 103.53),  // imagenet dataset
                    cv::Scalar std = cv::Scalar(1),
                    bool swap_rb = true);
        
        // the 1st step, MUST implement in specific derived class.
        // prepare data for infer, fetch frames from frame meta.
        virtual void prepare(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch, std::vector<cv::Mat>& mats_to_infer) = 0;
        
        // the 2nd step, has a default implementation.
        // preprocess data, such as normalization, mean substract.
        virtual void preprocess(const std::vector<cv::Mat>& mats_to_infer, cv::Mat& blob_to_infer);

        // the 3rd step, has a default implementation.
        // infer and retrive raw outputs.
        virtual void infer(const cv::Mat& blob_to_infer, std::vector<std::vector<int>>& outputs_shape, std::vector<std::vector<float>>& tensor_vector);

        // the 4th step, MUST implement in specific derived class.
        // postprocess on raw outputs and create/update something back to frame meta again.
        virtual void postprocess(std::vector<std::vector<int>>& outputs_shape, std::vector<std::vector<float>>& tensor_vector, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch);

        // debug purpose(ms)
        virtual void infer_combinations_time_cost(int data_size, int prepare_time, int preprocess_time, int infer_time, int postprocess_time);

        // infer operations(call prepare/preprocess/infer/postprocess by default)
        // we can define new logic for infer operations by overriding it.
        virtual void run_infer_combinations(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch);

	    const char* res;
	    std::string model_file;

        // labels as text format
        std::vector<std::string> labels;
        
	//tensorflowlite
        unique_ptr<tflite::Interpreter> interpreter;
        size_t num_devices;
        //unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices;
        std::unique_ptr<tflite::FlatBufferModel> model;
        tflite::ops::builtin::BuiltinOpResolver resolver;
        
        //openvino
        //ov::CompiledModel compiled_model;
        //ov::InferRequest infer_request;
        //size_t ov_num_outputs;
        
        int num_outputs;
        
	// opencv::dnn as backend 
        cv::dnn::Net net;
	
	//onnxruntime
	    std::unique_ptr<Ort::Session> session;//创建一个空会话
    	size_t num_input_nodes;	
    	size_t num_output_nodes;
    	std::vector<std::string> input_node_names;
    	std::vector<int64_t> input_node_dims;
    	std::vector<std::string> output_node_names;
    	std::vector<Ort::Value> output_tensors;
    	ONNXTensorElementDataType input_type;

        //std::vector<std::vector<float>> tensor_vector;
        //std::vector<std::vector<int>> outputs_shape;

        cv::Mat create_pascal_label_colormap();
        cv::Mat label_to_color_image(const cv::Mat& label, const cv::Mat& colormap);

        // re-implementation for one by one mode, marked as 'final' as we need not override any more in specific derived classes.
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override final; 
        // re-implementation for batch by batch mode, marked as 'final' as we need not override any more in specific derived classes.
        virtual void handle_frame_meta(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& meta_with_batch) override final; 
    public:
        ~vp_infer_node();
    };
}

#include <cstring>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include <future>
#include <limits>

#include <python3.9/Python.h>

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


#include "vp_infer_node.h"

using namespace cv;
using namespace cv::face;

namespace vp_nodes {
    vp_infer_node::vp_infer_node(std::string node_name, 
                            vp_infer_type infer_type, 
                            std::string model_path, 
			    std::string convert_type,
                            std::string model_config_path, 
                            std::string labels_path, 
                            int input_width, 
                            int input_height, 
                            int batch_size,
                            float scale,
                            cv::Scalar mean,
                            cv::Scalar std,
                            bool swap_rb):
                            vp_node(node_name),
                            infer_type(infer_type),
                            model_path(model_path),
			    convert_type(convert_type),
                            model_config_path(model_config_path),
                            labels_path(labels_path),
                            input_width(input_width),
                            input_height(input_height),
                            batch_size(batch_size),
                            scale(scale),
                            mean(mean),
                            std(std),
                            swap_rb(swap_rb) {
        // try to load network from file,
        // failing means maybe it has a custom implementation for model loading in derived class such as using other backends other than opencv::dnn.
    	if(!convert_type.empty()){
    	    const char* a = model_path.c_str();
    	    const char* b = model_config_path.c_str();
    	    const char* c = convert_type.c_str();
            std::string type;
            if (convert_type == "GPU"){
                type = "onnx";
            }
            else if(convert_type == "TPU"){
                type = "edgetpu";
            }
            else if(convert_type == "VPU"){
                type = "xml";
            }
    	    size_t model_suffix = model_path.find_last_of(".");
	        size_t suffix = model_path.find("edgetpu");
            if (model_path.substr(model_suffix + 1) == type || suffix < model_path.length()) {
                model_file = model_path;
            }
            else{
                    Py_Initialize();
                    if(!Py_IsInitialized()){
                        cout << "python init fail" << endl;
                        return;
                    }
                    PyRun_SimpleString("import sys");
                    PyRun_SimpleString("sys.path.append(r'/home/zyw/video_pipe_c-pure_cpu')");
                    PyObject *pModule;
                    PyObject *pFunc;
                    pModule = PyImport_ImportModule("convert");
                    pFunc = PyObject_GetAttrString(pModule, "run_convert");
                    PyObject *pArgs = PyTuple_New(3); 
                    PyTuple_SetItem(pArgs, 0, Py_BuildValue("s", a));
                    PyTuple_SetItem(pArgs, 1, Py_BuildValue("s", b));
                    PyTuple_SetItem(pArgs, 2, Py_BuildValue("s", type.c_str()));
                    PyObject *pRetValue = PyObject_CallObject(pFunc, pArgs);
                    PyArg_Parse(pRetValue, "s", &res);
                    model_file = res; 
                    cout << model_file << endl;
                    //cout << res << endl;
                    Py_Finalize();
            }
    	}
    	else {
    		model_file = model_path;
		    cout << model_file << endl;
    	}

    	size_t pos = model_file.find_last_of(".");
    	if (model_file.substr(pos + 1) == "onnx"){
    		static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "default");
            Ort::SessionOptions sessionOptions;
            //Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
        	session = std::make_unique<Ort::Session>(env, model_file.c_str(), sessionOptions);
    		    	
    		//get input node informarion
    		num_input_nodes = session->GetInputCount();
    		Ort::AllocatorWithDefaultOptions allocator;
    		Ort::AllocatedStringPtr input_name = session->GetInputNameAllocated(0, allocator);
    		input_node_names.push_back(input_name.get());
    		Ort::TypeInfo type_info = session->GetInputTypeInfo(0);
    		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    		input_node_dims = tensor_info.GetShape();
    		input_type = tensor_info.GetElementType();


    		//get output node information
    		num_output_nodes = session->GetOutputCount();
    		for (size_t i = 0; i < num_output_nodes; i++) {   
    			Ort::AllocatedStringPtr output_name = session->GetOutputNameAllocated(i, allocator);
    			output_node_names.push_back(output_name.get());
    		}
    	}
      	    
      else if (model_file.substr(pos + 1) == "tflite") {
            VP_WARN(vp_utils::string_format("[%s] cv::dnn::readNet load network failed!", node_name.c_str()));
            //size_t num_devices;
            unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(edgetpu_list_devices(&num_devices), &edgetpu_free_devices);

            if (num_devices == 0) 
            {
                cerr << "No connected TPU found" << endl;
                return;
            }
            const auto& device = devices.get()[0];

            // Load model.
            model = tflite::FlatBufferModel::BuildFromFile(model_file.c_str());
            if (!model) 
            {
                cerr << "Cannot read model from " << res << endl;
                return;
            }
                                
            // Create interpreter.
            //tflite::ops::builtin::BuiltinOpResolver resolver;
            //unique_ptr<tflite::Interpreter> interpreter;                
            if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk) 
            {
                cerr << "Cannot create interpreter" << endl;
                return;
            }
                                
            auto* delegate = edgetpu_create_delegate(device.type, device.path, nullptr, 0);
            interpreter->ModifyGraphWithDelegate(delegate);

            // Allocate tensors.
            if (interpreter->AllocateTensors() != kTfLiteOk) 
            {
                cerr << "Cannot allocate interpreter tensors" << endl;
                return;
            } 

      }
      //else if (model_file.substr(pos + 1) == "xml") {
        //ov::Core core;
        // Step 2. Read a model
        //std::shared_ptr<ov::Model> model = core.read_model(model_file.c_str());
                                        
        // Step 4. Inizialize Preprocessing for the model
        //ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
        // Specify input image format
        //ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
        // Specify preprocess pipeline to input image without resizing
        //ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({255., 255., 255.});
        //  Specify model's input layout
        //ppp.input().model().set_layout("NCHW");
        // Specify output results format
        //ppp.output().tensor().set_element_type(ov::element::f32);
        // Embed above steps in the graph
        //model = ppp.build();
        //compiled_model = core.compile_model(model, "MYRIAD");
        // 获取模型的输出节点信息
        //std::vector<ov::Output<const ov::Node>> outputs = compiled_model.outputs();
        //ov_num_outputs = outputs.size();
        //infer_request = compiled_model.create_infer_request();
      //}

        // load labels if labels_path is specified
        if (labels_path != "") {
            load_labels();
        }

        assert(batch_size > 0);
        // primary infer nodes can handle frame meta batch by batch(whole frame), 
        // others can handle multi batchs ONLY inside a single frame(small croped image).
        if (infer_type == vp_infer_type::PRIMARY && batch_size > 1) {
            frame_meta_handle_batch = batch_size;
        }
    }
    
    vp_infer_node::~vp_infer_node() {

    } 

    // handle frame meta one by one
    std::shared_ptr<vp_objects::vp_meta> vp_infer_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        std::vector<std::shared_ptr<vp_objects::vp_frame_meta>> frame_meta_with_batch {meta};
        run_infer_combinations(frame_meta_with_batch);    
        return meta;
    }

    // handle frame meta batch by batch
    void vp_infer_node::handle_frame_meta(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& meta_with_batch) {
        const auto& frame_meta_with_batch = meta_with_batch;
        run_infer_combinations(frame_meta_with_batch);
        // no return
    }

    // default implementation
    // infer batch by batch
    void vp_infer_node::infer(const cv::Mat& blob_to_infer, std::vector<std::vector<int>>& outputs_shape, std::vector<std::vector<float>>& tensor_vector) {
        if(interpreter) {

            int INPUT_HEIGHT;
            int INPUT_WIDTH;
            // Set interpreter input.
            const auto* input_tensor = interpreter->input_tensor(0);
            TfLiteType input_type = input_tensor->type;
            num_outputs = interpreter->outputs().size();
            if(input_tensor->dims->data[1] != 3) {
                INPUT_HEIGHT = input_tensor->dims->data[1];
                INPUT_WIDTH = input_tensor->dims->data[2]; 
            }
            else{
                INPUT_HEIGHT = input_tensor->dims->data[2];
                INPUT_WIDTH = input_tensor->dims->data[3]; 
            }

            cv::Mat TFLite_workingFrame;

            //time_t start, end;

            //workingFrame = vp_nodes::FRAME; 
            //cvtColor(cameraFrame, workingFrame, COLOR_BGR2RGB);
            //cout << "detection_number" << endl;
            cv::Size targetSize(INPUT_WIDTH,INPUT_HEIGHT);
            cv::resize(blob_to_infer, TFLite_workingFrame, targetSize, 0, 0);
            if (input_type == kTfLiteUInt8) {
                uint8_t* input_data = interpreter->typed_input_tensor<uint8_t>(0);
                memcpy(input_data, TFLite_workingFrame.data, TFLite_workingFrame.total() * TFLite_workingFrame.elemSize());
            }
            else if (input_type == kTfLiteFloat32) {
                float* input_data = interpreter->typed_input_tensor<float>(0);
                memcpy(input_data, TFLite_workingFrame.data, TFLite_workingFrame.total() * TFLite_workingFrame.elemSize());
            }
            
            // Run inference.
            if (interpreter->Invoke() != kTfLiteOk) 
            {
                cerr << "Cannot invoke interpreter" << endl;
                return;
            }
            const std::vector<int>& output_indices = interpreter->outputs();
            for (int i = 0; i < num_outputs; ++i) {
                int output_index = output_indices[i];
                TfLiteTensor* tensor = interpreter->tensor(output_index);
                std::vector<int> tensor_shape(tensor->dims->size);
                for (int i = 0; i < tensor->dims->size; ++i) {
                    tensor_shape[i] = tensor->dims->data[i];
                }
                outputs_shape.push_back(tensor_shape);
                // 计算张量的总元素数
                int num_elements = 1;
                for (int dim : tensor_shape) {
                    num_elements *= dim;
                }
                if (input_type == kTfLiteUInt8){
                    auto* tensor_data = interpreter->typed_output_tensor<int64_t>(0);
                    tensor_vector.push_back(std::vector<float>(tensor_data, tensor_data + num_elements));
                }
                else if (input_type == kTfLiteFloat32) {
                    auto* tensor_data = interpreter->typed_output_tensor<float>(0);
                    tensor_vector.push_back(std::vector<float>(tensor_data, tensor_data + num_elements));
                }
                
            }

        }
        //else if (infer_request) {
            //cout << "ov_num_outputs:" << ov_num_outputs << endl;
            //cv::Mat WorkingFrame = blob_to_infer;
            //ov::Tensor input_tensor = infer_request.get_input_tensor(0);
            //ov::Tensor input_tensor_1 = infer_request.get_input_tensor(1);
            //ov::Shape tensor_shape_1 = input_tensor_1.get_shape();
            //ov::Shape tensor_shape = input_tensor.get_shape();
            //bool is_NHWC = (tensor_shape[1] != 3);
            //size_t channel = 3;
            //size_t height = is_NHWC ? tensor_shape[1] : tensor_shape[2];
            //size_t width = is_NHWC ? tensor_shape[2] : tensor_shape[3];
            //cv::Mat blob_image;
            //cv::resize(WorkingFrame, blob_image, cv::Size(width, height));
            //float* image_data = input_tensor.data<float>();
            //if (is_NHWC) {
              //for (size_t h = 0; h < height; h++) {
                //for (size_t w = 0; w < width; w++) {
                  //for (size_t c = 0; c < channel; c++) {
                    //size_t index = h * width * channel + w * channel +c;
                      //image_data[index] =  blob_image.at<cv::Vec3b>(h, w)[c];
                  //}
                //}
              //}
            //}else{
              //for (size_t c = 0; c < channel; c++) {
                //for (size_t h = 0; h < height; h++) {
                  //for (size_t w = 0; w < width; w++) {
                    //size_t index = c * width * height + h * width +w;
                      //image_data[index] =  blob_image.at<cv::Vec3b>(h, w)[c];
                  //}
                //}
              //}
            //}
            // Step 6. Create an infer request for model inference 
            //infer_request.infer();
            //for (size_t i = 0; i < ov_num_outputs; i++) {
              //const ov::Tensor& output_tensor = infer_request.get_output_tensor(i);
                  
              // 获取输出张量的形状
              //std::vector<int> tensors_shape;
              //ov::Shape tensor_shape = output_tensor.get_shape();
              //for (size_t j = 0; j < tensor_shape.size(); j++) {
                //tensors_shape.push_back(tensor_shape[j]);
              //}
              //outputs_shape.push_back(tensors_shape);
              // 获取输出张量的内容
              //const float* tensor_data = output_tensor.data<float>();
              //size_t tensor_size = output_tensor.get_byte_size();
              //tensor_vector.push_back(std::vector<float>(tensor_data, tensor_data + (tensor_size / sizeof(float))));
            //}
        //}
        else {
            Ort::TypeInfo type_info = session->GetInputTypeInfo(0);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            ONNXTensorElementDataType input_types = tensor_info.GetElementType();
        		std::vector<const char*> input_node_names_cstr;
        		for (const auto& name : input_node_names) {
        		    input_node_names_cstr.push_back(name.c_str());
        		}
        		std::vector<const char*> output_node_names_cstr;
        	    for (const auto& name : output_node_names) {
        		    output_node_names_cstr.push_back(name.c_str());
        	    }              
        		bool is_NHWC = (input_node_dims[1] != 3);
        		int64_t batch_size = input_node_dims[0];
        		int64_t height = is_NHWC ? input_node_dims[1] : input_node_dims[2];
        		int64_t width = is_NHWC ? input_node_dims[2] : input_node_dims[3];
        		int64_t channels = 3;
        		cv::Mat ONNX_workingFrame;
        		cv::resize(blob_to_infer, ONNX_workingFrame, cv::Size(width, height));
            if (input_types == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8){
                std::vector<uint8_t> input_data;
                if (is_NHWC) {
                // NHWC 格式
                    input_data.resize(batch_size * height * width * channels);
                    for (int b = 0; b < batch_size; ++b) {
                        for (int i = 0; i < height; ++i) {
                            for (int j = 0; j < width; ++j) {
                                for (int c = 0; c < channels; ++c) {
                                    input_data[b * height * width * channels + i * width * channels + j * channels + c] = ONNX_workingFrame.at<cv::Vec3b>(i, j)[c];
                                }
                            }
                        }
                    }
                } else {
                // NCHW 格式
                    input_data.resize(batch_size * channels * height * width);
                    for (int b = 0; b < batch_size; ++b) {
                        for (int c = 0; c < channels; ++c) {
                            for (int i = 0; i < height; ++i) {
                                for (int j = 0; j < width; ++j) {
                                    input_data[b * channels * height * width + c * height * width + i * width + j] = ONNX_workingFrame.at<cv::Vec3b>(i, j)[c];
                                }
                            }
                        }
                    }
                }
                Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                Ort::Value input_tensors = Ort::Value::CreateTensor<uint8_t>(memory_info, input_data.data(), input_data.size(), input_node_dims.data(), input_node_dims.size());
                output_tensors = session->Run(Ort::RunOptions{nullptr}, input_node_names_cstr.data(), &input_tensors, num_input_nodes, output_node_names_cstr.data(), num_output_nodes);
            }
            else if (input_types == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT){
                std::vector<float> input_data;
                if (is_NHWC) {
                // NHWC 格式
                    input_data.resize(batch_size * height * width * channels);
                    for (int b = 0; b < batch_size; ++b) {
                        for (int i = 0; i < height; ++i) {
                            for (int j = 0; j < width; ++j) {
                                for (int c = 0; c < channels; ++c) {
                                    input_data[b * height * width * channels + i * width * channels + j * channels + c] = ONNX_workingFrame.at<cv::Vec3b>(i, j)[c];
                                }
                            }
                        }
                    }
                } else {
                // NCHW 格式
                    input_data.resize(batch_size * channels * height * width);
                    for (int b = 0; b < batch_size; ++b) {
                        for (int c = 0; c < channels; ++c) {
                            for (int i = 0; i < height; ++i) {
                                for (int j = 0; j < width; ++j) {
                                    input_data[b * channels * height * width + c * height * width + i * width + j] = ONNX_workingFrame.at<cv::Vec3b>(i, j)[c];
                                }
                            }
                        }
                    }
                }
                Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                Ort::Value input_tensors = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(), input_node_dims.data(), input_node_dims.size());
                output_tensors = session->Run(Ort::RunOptions{nullptr}, input_node_names_cstr.data(), &input_tensors, num_input_nodes, output_node_names_cstr.data(), num_output_nodes);
            }
		
            for(int i = 0; i < num_output_nodes; i++) {
                Ort::TypeInfo type_info = output_tensors[i].GetTypeInfo();
                ONNXTensorElementDataType output_type = type_info.GetTensorTypeAndShapeInfo().GetElementType();
                Ort::TensorTypeAndShapeInfo tensor_info = output_tensors[i].GetTensorTypeAndShapeInfo();
                size_t num_elements = tensor_info.GetElementCount();
                size_t num_dims = tensor_info.GetDimensionsCount();
                std::vector<long int> outputs_dims = tensor_info.GetShape();
                std::vector<int> tensor_shape;
                for (auto dim : outputs_dims) {
                    tensor_shape.push_back(dim);
                }
                outputs_shape.push_back(tensor_shape);
                if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                    auto* float_array = output_tensors[i].GetTensorMutableData<int64_t>();
                    tensor_vector.push_back(std::vector<float>(float_array, float_array + num_elements));
                }
                else if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                    auto* float_array = output_tensors[i].GetTensorMutableData<float>();
                    tensor_vector.push_back(std::vector<float>(float_array, float_array + num_elements));
                }
                
                //tensor_vector.push_back(std::vector<float>(float_array, float_array + num_elements));
                //auto* maxElement = std::max_element(float_array, float_array + num_elements);
                //scout << *maxElement << endl;

            } 
        }      
    }

    // 创建PASCAL VOC标签颜色映射
    cv::Mat vp_infer_node::create_pascal_label_colormap() {
        cv::Mat colormap(256, 1, CV_8UC3);
        for (int i = 0; i < 256; ++i) {
            int r = 0, g = 0, b = 0;
            int cid = i;
            for (int j = 0; j < 8; ++j) {
                r |= ((cid >> 0) & 1) << (7 - j);
                g |= ((cid >> 1) & 1) << (7 - j);
                b |= ((cid >> 2) & 1) << (7 - j);
                cid >>= 3;
            }
            colormap.at<cv::Vec3b>(i) = cv::Vec3b(b, g, r);
        }
        return colormap;
    }

    // 将标签转换为彩色图像
    cv::Mat vp_infer_node::label_to_color_image(const cv::Mat& label, const cv::Mat& colormap) {
        if (label.type() != CV_32SC1) {
            throw std::invalid_argument("Expect 2-D input label of type CV_32SC1");
        }

        cv::Mat color_image(label.size(), CV_8UC3);
        for (int r = 0; r < label.rows; ++r) {
            for (int c = 0; c < label.cols; ++c) {
                auto label_value = label.at<int>(r, c);
                if (label_value >= colormap.rows) {
                    cout << label_value << endl;
                    cout << r << "," << c <<endl;
                    throw std::invalid_argument("Label value too large.");
                }
                color_image.at<cv::Vec3b>(r, c) = colormap.at<cv::Vec3b>(label_value);
            }
        }
        return color_image;
    }

    void vp_infer_node::postprocess(std::vector<std::vector<int>>& outputs_shape, std::vector<std::vector<float>>& tensor_vector, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {  

        if (outputs_shape.size() == 1 && outputs_shape[0] == std::vector<int>({1, outputs_shape[0][1], outputs_shape[0][2]})) {
            float* data = tensor_vector[0].data();
            assert(frame_meta_with_batch.size() == 1);
            auto& frame_meta = frame_meta_with_batch[0];
            int inputW = frame_meta->frame.cols;
            int inputH = frame_meta->frame.rows;
            cv::Mat result(outputs_shape[0][1], outputs_shape[0][2], CV_32SC1);
            for (int i = 0; i < outputs_shape[0][1]; ++i) {
                for (int j = 0; j < outputs_shape[0][2]; ++j) {
                    result.at<int>(i, j) = data[i * outputs_shape[0][1] + j];
                }
            }
            cv::Mat colormap = create_pascal_label_colormap();
            cv::Mat mask_img = label_to_color_image(result, colormap);
            auto face_target = std::make_shared<vp_objects::vp_frame_face_target>(0, 0, 0, 0, 0, "", mask_img);
            frame_meta->face_targets.push_back(face_target);
            outputs_shape.clear();
            tensor_vector.clear();
        }
        else if (outputs_shape.size() == 4 && outputs_shape[0] == std::vector<int>({1, outputs_shape[0][1], 4}) && outputs_shape[1] == std::vector<int>({1, outputs_shape[1][1]}) && outputs_shape[2] == std::vector<int>({1, outputs_shape[2][1]})) {
            float* detection_boxes = tensor_vector[0].data();
            float* detection_label = tensor_vector[1].data();
            float* detection_score = tensor_vector[2].data();
            float* detection_number = tensor_vector[3].data();
            assert(frame_meta_with_batch.size() == 1);
            auto& frame_meta = frame_meta_with_batch[0];
            int inputW = frame_meta->frame.cols;
            int inputH = frame_meta->frame.rows;
            for(int i = 0; i < *detection_number; i++)
            {
                    if(detection_score[i] > 0.7)
                    {
                                
                        float score = detection_score[i];
                        float y1 = detection_boxes[4*i+0] * inputH;
                        float x1 = detection_boxes[4*i+1] * inputW;
                        float y2 = detection_boxes[4*i+2] * inputH;
                        float x2 = detection_boxes[4*i+3] * inputW;
                        float width = x2 - x1;
                        float height = y2 - y1;
                        auto det_index = detection_label[i];
                        if (labels_path == "") {
                            auto face_target = std::make_shared<vp_objects::vp_frame_face_target>(x1, y1, width, height, score);
                            frame_meta->face_targets.push_back(face_target);
                        }
                        else{
                            ifstream file(labels_path);
                            if(!file)
                            {
                                return; // Open failed.
                            }  
                            std::vector<string> labels;
                            for (string line; getline(file, line);)
                            {
                                labels.emplace_back(line);
                            }
                            auto label = labels[det_index].c_str();
                            auto target = std::make_shared<vp_objects::vp_frame_target>(x1, y1, width, height, det_index, score, frame_meta->frame_index, frame_meta->channel_index, label);
                            frame_meta->targets.push_back(target);
                        } 
                    
                    }
            }
            outputs_shape.clear();
            tensor_vector.clear();
        } 
        else if (outputs_shape.size() == 1 && outputs_shape[0] == std::vector<int>({1, 1, outputs_shape[0][2], 7})) {
            std::vector<int> shape = outputs_shape[0];
            int maxProposalCount = shape[2];
            int objectSize = shape[3];
            float* detections = tensor_vector[0].data();
            assert(frame_meta_with_batch.size() == 1);
            auto& frame_meta = frame_meta_with_batch[0];
            int inputW = frame_meta->frame.cols;
            int inputH = frame_meta->frame.rows;
            //inputW = 640;
            //inputH = 480;
            for (unsigned int i = 0; i < maxProposalCount; i++) 
            {
                float image_id = detections[i * objectSize + 0];
                // exit early if the detection is not a face
                if (image_id < 0) 
                {
                    //std::cout << "Only " << i << " proposals found" << std::endl;
                    break;
                }
                    
                // Calculate and save the values that we need
                // These values are the confidence scores and bounding box coordinates
                float confidence = detections[i * objectSize + 2];
                int xmin = (int)(detections[i * objectSize + 3] * inputW);
                int ymin = (int)(detections[i * objectSize + 4] * inputH);
                int xmax = (int)(detections[i * objectSize + 5] * inputW);
                int ymax = (int)(detections[i * objectSize + 6] * inputH);  

                // filter out low scores
                if (confidence > 0.7) {
                    // Make sure coordinates are do not exceed image dimensions
                    xmin = std::max(0, xmin);
                    ymin = std::max(0, ymin);
                    xmax = std::min(inputW, xmax);
                    ymax = std::min(inputH, ymax);
                    int width = xmax - xmin;
                    int height = ymax - ymin;
                    // Helper for current detection
                    //detectionResults currentDetection;
                        
                    // Put the cropped face and bounding box coordinates into the detected faces vector
                    if (0 <= xmin && 0 <= width && xmin + width <= inputW && 0 <= ymin && 0 <= height && ymin + height <= inputH) {
                        auto face_target = std::make_shared<vp_objects::vp_frame_face_target>(xmin, ymin, width, height, confidence);
                        frame_meta->face_targets.push_back(face_target);
                    }        
                }
            }
            outputs_shape.clear();
            tensor_vector.clear();
        }
        else if (outputs_shape.size() == 1 && outputs_shape[0] == std::vector<int>({outputs_shape[0][0], 5})) {
            std::vector<int> shape = outputs_shape[0];
            int maxProposalCount = shape[0];
            int objectSize = shape[1];
            float* detections = tensor_vector[0].data();
            assert(frame_meta_with_batch.size() == 1);
            auto& frame_meta = frame_meta_with_batch[0];
            int inputW = frame_meta->frame.cols;
            int inputH = frame_meta->frame.rows;
            //inputW = 640;
            //inputH = 480;
            for (unsigned int i = 0; i < maxProposalCount; i++) 
            {
                //float image_id = detections[i * objectSize + 0];
                // exit early if the detection is not a face
                //if (image_id < 0) 
                //{
                    //std::cout << "Only " << i << " proposals found" << std::endl;
                    //break;
                //}
                    
                // Calculate and save the values that we need
                // These values are the confidence scores and bounding box coordinates
                float confidence = detections[i * objectSize + 4];
                int xmin = (int)(detections[i * objectSize + 0] * inputW);
                int ymin = (int)(detections[i * objectSize + 1] * inputH);
                int xmax = (int)(detections[i * objectSize + 2] * inputW);
                int ymax = (int)(detections[i * objectSize + 3] * inputH);  

                // filter out low scores
                if (confidence > 0.7) {
                    // Make sure coordinates are do not exceed image dimensions
                    xmin = std::max(0, xmin);
                    ymin = std::max(0, ymin);
                    xmax = std::min(inputW, xmax);
                    ymax = std::min(inputH, ymax);
                    int width = xmax - xmin;
                    int height = ymax - ymin;
                    // Helper for current detection
                    //detectionResults currentDetection;
                        
                    // Put the cropped face and bounding box coordinates into the detected faces vector
                    if (0 <= xmin && 0 <= width && xmin + width <= inputW && 0 <= ymin && 0 <= height && ymin + height <= inputH) {
                        auto face_target = std::make_shared<vp_objects::vp_frame_face_target>(xmin, ymin, width, height, confidence);
                        frame_meta->face_targets.push_back(face_target);
                    }        
                }
            }
            outputs_shape.clear();
            tensor_vector.clear();
        }
        else if (outputs_shape.size() == 2 && outputs_shape[0] == std::vector<int>({1, 1, 1, 1}) && outputs_shape[0] == std::vector<int>({1, 2, 1, 1})) {
            assert(frame_meta_with_batch.size() == 1);
            auto& frame_meta = frame_meta_with_batch[0];
            float* age = tensor_vector[0].data();
            float* gender = tensor_vector[1].data();
            float Age = *age / 100;
            std::string Gender;
            if (gender[0] > gender[1]) {
                Gender = "female";
            }else {
                Gender = "male";
            }
            auto face_target = std::make_shared<vp_objects::vp_frame_face_target>(0, 0, 0, 0, Age, Gender);
            frame_meta->face_targets.push_back(face_target);

        }    
    }

    // default implementation
    // create a 4D matrix(n, c, h, w)
    void vp_infer_node::preprocess(const std::vector<cv::Mat>& mats_to_infer, cv::Mat& blob_to_infer) {
        if(!net.empty()) {
                this->Mats = mats_to_infer;
                cv::dnn::blobFromImages(mats_to_infer, blob_to_infer, scale, cv::Size(input_width, input_height), mean, swap_rb);
                if (std != cv::Scalar(1)) {
                    // divide by std
                }
        }
        else {
                blob_to_infer = mats_to_infer[0].clone();
        }
    }

    void vp_infer_node::run_infer_combinations(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
        /*
        * call logic by default:
        * frame_meta_with_batch -> mats_to_infer -> blob_to_infer -> raw_outputs -> frame_meta_with_batch
        */
        std::vector<cv::Mat> mats_to_infer;
        // 4D matrix
        cv::Mat blob_to_infer;
        // multi heads of output in network, raw matrix output which need to be parsed by users.
        std::vector<std::vector<int>> outputs_shape;
        std::vector<std::vector<float>> tensor_vector;

        // start
        auto start_time = std::chrono::system_clock::now();
        // 1st step, prepare
        prepare(frame_meta_with_batch, mats_to_infer);
        auto prepare_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

        // nothing to infer
        if (mats_to_infer.size() == 0) {
            return;
        }

        start_time = std::chrono::system_clock::now();
        // 2nd step, preprocess
        preprocess(mats_to_infer, blob_to_infer);
        auto preprocess_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

        start_time = std::chrono::system_clock::now();
        // 3rd step, infer
        infer(blob_to_infer, outputs_shape, tensor_vector);
        auto infer_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

        start_time = std::chrono::system_clock::now();
        // 4th step, postprocess
        postprocess(outputs_shape, tensor_vector, frame_meta_with_batch);   
        auto postprocess_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

        // end
        infer_combinations_time_cost(mats_to_infer.size(), prepare_time.count(), preprocess_time.count(), infer_time.count(), postprocess_time.count());
    }

    // print all by default
    void vp_infer_node::infer_combinations_time_cost(int data_size, int prepare_time, int preprocess_time, int infer_time, int postprocess_time) {
        /*
        std::cout << "########## infer combinations summary ##########" << std::endl;
        std::cout << " node_name:" << node_name << std::endl;
        std::cout << " data_size:" << data_size << std::endl;
        std::cout << " prepare_time:" << prepare_time << "ms" << std::endl;
        std::cout << " preprocess_time:" << preprocess_time << "ms" << std::endl;
        std::cout << " infer_time:" << infer_time << "ms" << std::endl;
        std::cout << " postprocess_time:" << postprocess_time << "ms" << std::endl;
        std::cout << "########## infer combinations summary ##########" << std::endl;
        */

        std::ostringstream s_stream;
        s_stream << "\n########## infer combinations summary ##########\n";
        s_stream << " node_name:" << node_name << "\n";
        s_stream << " data_size:" << data_size << "\n";
        s_stream << " prepare_time:" << prepare_time << "ms\n";
        s_stream << " preprocess_time:" << preprocess_time << "ms\n";
        s_stream << " infer_time:" << infer_time << "ms\n";
        s_stream << " postprocess_time:" << postprocess_time << "ms\n";
        s_stream << "########## infer combinations summary ##########\n";     

        // to log
        VP_DEBUG(s_stream.str());
    }

    void vp_infer_node::load_labels() {
        try {
            std::ifstream label_stream(labels_path);
            for (std::string line; std::getline(label_stream, line); ) {
                if (!line.empty() && line[line.length() - 1] == '\r') {
                    line.erase(line.length() - 1);
                }
                labels.push_back(line);
            }
        }
        catch(const std::exception& e) {
            
        }  
    }
}

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include <variant>
#include <list>
#include <iterator>


#include "edgetpu_c.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#include <opencv2/opencv.hpp>

#include "vp_plate_ocr_node_tpu.h"

namespace vp_nodes {
        
    vp_plate_ocr_node_tpu::vp_plate_ocr_node_tpu(std::string node_name, 
                                                 std::string model_path,
 						 std::string convert_type
                                                 ):
                                                 vp_secondary_infer_node(node_name, model_path, convert_type)
                                                 {
        this->initialized();
    }
    
    vp_plate_ocr_node_tpu::~vp_plate_ocr_node_tpu() {

    }
    void vp_plate_ocr_node_tpu::postprocess(std::vector<std::vector<int>>& outputs_shape, std::vector<std::vector<float>>& tensor_vector, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {

    }
        
    void vp_plate_ocr_node_tpu::run_infer_combinations(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
        
        std::unordered_map<std::string, int> char2value = {{"0", 0}, {"1", 1}, {"2", 2}, {"3", 3}, {"4", 4}, {"5", 5}, {"6", 6}, {"7", 7}, {"8", 8}, {"9", 9}, {"<Anhui>", 10}, {"<Beijing>", 11}, {"<Chongqing>", 12}, {"<Fujian>", 13}, {"<Gansu>", 14}, {"<Guangdong>", 15}, {"<Guangxi>", 16}, {"<Guizhou>", 17}, {"<Hainan>", 18}, {"<Hebei>", 19}, {"<Heilongjiang>", 20}, {"<Henan>", 21}, {"<HongKong>", 22}, {"<Hubei>", 23}, {"<Hunan>", 24}, {"<InnerMongolia>", 25}, {"<Jiangsu>", 26}, {"<Jiangxi>", 27}, {"<Jilin>", 28}, {"<Liaoning>", 29}, {"<Macau>", 30}, {"<Ningxia>", 31}, {"<Qinghai>", 32}, {"<Shaanxi>", 33}, {"<Shandong>", 34}, {"<Shanghai>", 35}, {"<Shanxi>", 36}, {"<Sichuan>", 37}, {"<Tianjin>", 38}, {"<Tibet>", 39}, {"<Xinjiang>", 40}, {"<Yunnan>", 41}, {"<Zhejiang>", 42}, {"<police>", 43}, {"A", 44}, {"B", 45}, {"C", 46}, {"D", 47}, {"E", 48}, {"F", 49}, {"G", 50}, {"H", 51}, {"I", 52}, {"J", 53}, {"K", 54}, {"L", 55}, {"M", 56}, {"N", 57}, {"O", 58}, {"P", 59}, {"Q", 60}, {"R", 61}, {"S", 62}, {"T", 63}, {"U", 64}, {"V", 65}, {"W", 66}, {"X", 67}, {"Y", 68}, {"Z", 69}, {"_", 70}};
        std::unordered_map<int, std::string> value2char;
        for (const auto& pair : char2value) {
                value2char[pair.second] = pair.first;
        }
        assert(frame_meta_with_batch.size() == 1);
        auto& frame_meta = frame_meta_with_batch[0];
        cv::Rect box;
        for (const auto& i : frame_meta->face_targets) {
                box = cv::Rect(i->x, i->y, i->width, i->height);
                box.x = std::max(box.x, 0);
                box.y = std::max(box.y, 0);
                box.width = std::min(box.width, frame_meta->frame.cols - box.x);
                box.height = std::min(box.height, frame_meta->frame.rows - box.y);
        }
        
        std::vector<cv::Mat> mats_to_infer;
        cv::Mat blob_to_infer;
        std::vector<std::vector<int>> outputs_shape;
        std::vector<std::vector<float>> tensor_vector;

        // start
        auto start_time = std::chrono::system_clock::now();

        // prepare data, as same as base class
        vp_secondary_infer_node::prepare(frame_meta_with_batch, mats_to_infer);

        //vp_infer_node::preprocess(mats_to_infer, blob_to_infer);
        if(!mats_to_infer.empty()) {
                blob_to_infer = mats_to_infer[0].clone();
                auto prepare_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

                start_time = std::chrono::system_clock::now();
                // here we detect vehicle plate for all target(like car/bus) in frame
                vp_infer_node::infer(blob_to_infer, outputs_shape, tensor_vector);
                TfLiteTensor* outputTensor = interpreter->tensor(interpreter->outputs()[0]);
                const float* detection = interpreter->tensor(interpreter->outputs()[0])->data.f;
                const int* dims = outputTensor->dims->data;
                std::string plate_characters = "";
                std::list<std::string> output_characters = {"_"};
                std::list<int> List;
                int index = 7 * dims[2] - 1;
                int a;
                for(int i = 0; i < dims[1]; i++) {
                        float temp = -2e20;
                        for(int j = 0; j < dims[2]; j++) {
                                if(temp < detection[index]) {
                                        temp = detection[index];
                                        a = j;
                                }
                                index++;
                        }
                        List.push_back(a);
                }
                for(const auto& a : List) {
                      output_characters.push_back(value2char[a]);
                      std::string lastElement = output_characters.back();
                      auto it = output_characters.end();
                      std::advance(it, -2);
                      std::string secondlastElement = *it;
                      if (lastElement == secondlastElement || lastElement == "_") {
                              continue;
                      }
                      plate_characters += lastElement;
                }
                auto target = std::make_shared<vp_objects::vp_frame_target>(box.x, box.y, box.width, box.height, 
                                                    -1, 0, frame_meta->frame_index, frame_meta->channel_index, plate_characters);      
                frame_meta->targets.push_back(target);
                
                auto infer_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

                // can not calculate preprocess time and postprocess time, set 0 by default.
                vp_infer_node::infer_combinations_time_cost(mats_to_infer.size(), prepare_time.count(), 0, infer_time.count(), 0);
        }
    }
        
        
     
}

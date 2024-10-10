
#include "vp_yunet_face_detector_node.h"


namespace vp_nodes {
        
    vp_yunet_face_detector_node::vp_yunet_face_detector_node(std::string node_name, 
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
                                                            topK(top_k) {
        this->initialized();
    }
    
    vp_yunet_face_detector_node::~vp_yunet_face_detector_node() {

    }
    
    void vp_yunet_face_detector_node::postprocess(std::vector<std::vector<int>>& outputs_shape, std::vector<std::vector<float>>& tensor_vector, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
        using namespace cv;
        // 3 heads of output
        //assert(raw_outputs.size() == 3);
        assert(frame_meta_with_batch.size() == 1);
        auto& frame_meta = frame_meta_with_batch[0];

        // Extract from output_blobs
        //Mat loc = raw_outputs[0];
        //Mat conf = raw_outputs[1];
        //Mat iou = raw_outputs[2];

        // we need generate priors if input size changed or priors is not initialized
        if (outputs_shape[0][0] != priors.size()) {
            inputW = frame_meta->frame.cols;
            inputH = frame_meta->frame.rows;
            generatePriors();
        }
        
        //assert(outputs_shape[0][0] == priors.size());
        assert(outputs_shape[0][0] == outputs_shape[1][0]);
        assert(outputs_shape[0][0] == outputs_shape[2][0]);
        cout << inputW << endl;

        // Decode from deltas and priors
        const std::vector<float> variance = {0.1f, 0.2f};
        float* loc_v = tensor_vector[0].data();
        float* conf_v = tensor_vector[1].data();
        float* iou_v = tensor_vector[2].data();
        Mat faces;
        Mat face(1, 15, CV_32FC1);
        for (size_t i = 0; i < outputs_shape[0][0]; ++i) {
            
            // Get score
            float clsScore = conf_v[i*2+1];
            float iouScore = iou_v[i];
            // Clamp
            if (iouScore < 0.f) {
                iouScore = 0.f;
            }
            else if (iouScore > 1.f) {
                iouScore = 1.f;
            }
            float score = std::sqrt(clsScore * iouScore);
            face.at<float>(0, 14) = score;

            // Get bounding box
            float cx = (priors[i].x + loc_v[i*14+0] * variance[0] * priors[i].width)  * inputW;
            float cy = (priors[i].y + loc_v[i*14+1] * variance[0] * priors[i].height) * inputH;
            float w  = priors[i].width  * exp(loc_v[i*14+2] * variance[0]) * inputW;
            float h  = priors[i].height * exp(loc_v[i*14+3] * variance[1]) * inputH;
            float x1 = cx - w / 2;
            float y1 = cy - h / 2;
            face.at<float>(0, 0) = x1;
            face.at<float>(0, 1) = y1;
            face.at<float>(0, 2) = w;
            face.at<float>(0, 3) = h;
            faces.push_back(face);
        }

        if (faces.rows > 1)
        {
            // Retrieve boxes and scores
            std::vector<Rect2i> faceBoxes;
            std::vector<float> faceScores;
            for (int rIdx = 0; rIdx < faces.rows; rIdx++)
            {
                faceBoxes.push_back(Rect2i(int(faces.at<float>(rIdx, 0)),
                                           int(faces.at<float>(rIdx, 1)),
                                           int(faces.at<float>(rIdx, 2)),
                                           int(faces.at<float>(rIdx, 3))));
                faceScores.push_back(faces.at<float>(rIdx, 14));
            }

            std::vector<int> keepIdx;
            dnn::NMSBoxes(faceBoxes, faceScores, scoreThreshold, nmsThreshold, keepIdx, 1.f, topK);

            // Get NMS results
            Mat nms_faces;
            for (int idx: keepIdx)
            {
                nms_faces.push_back(faces.row(idx));
            }

            // insert face target back to frame meta
            for (int i = 0; i < nms_faces.rows; i++) {
                auto x = int(nms_faces.at<float>(i, 0));
                auto y = int(nms_faces.at<float>(i, 1));
                auto w = int(nms_faces.at<float>(i, 2));
                auto h = int(nms_faces.at<float>(i, 3));
                
                // check value range
                x = std::max(x, 0);
                y = std::max(y, 0);
                w = std::min(w, frame_meta->frame.cols - x);
                h = std::min(h, frame_meta->frame.rows - y);

                auto score = nms_faces.at<float>(i, 14);

                auto face_target = std::make_shared<vp_objects::vp_frame_face_target>(x, y, w, h, score);

                frame_meta->face_targets.push_back(face_target);
            }
            
        }
    }

    // // refer to vp_infer_node::preprocess
    // void vp_yunet_face_detector_node::preprocess(const std::vector<cv::Mat>& mats_to_infer, cv::Mat& blob_to_infer) {
    //     cv::dnn::blobFromImages(mats_to_infer, blob_to_infer);
    // }
    
    // // refer to vp_infer_node::infer
    // void vp_yunet_face_detector_node::infer(const cv::Mat& blob_to_infer, std::vector<cv::Mat>& raw_outputs) {
    //     // blob_to_infer is a 4D matrix
    //     // the first dim is number of batch, MUST be 1
    //     assert(blob_to_infer.dims == 4);
    //     assert(blob_to_infer.size[0] == 1);
    //     assert(!net.empty());

    //     net.setInput(blob_to_infer);
    //     net.forward(raw_outputs, out_names);
    // }

    void vp_yunet_face_detector_node::generatePriors() {
        using namespace cv;
        // Calculate shapes of different scales according to the shape of input image
        Size feature_map_2nd = {
            int(int((inputW+1)/2)/2), int(int((inputH+1)/2)/2)
        };
        Size feature_map_3rd = {
            int(feature_map_2nd.width/2), int(feature_map_2nd.height/2)
        };
        Size feature_map_4th = {
            int(feature_map_3rd.width/2), int(feature_map_3rd.height/2)
        };
        Size feature_map_5th = {
            int(feature_map_4th.width/2), int(feature_map_4th.height/2)
        };
        Size feature_map_6th = {
            int(feature_map_5th.width/2), int(feature_map_5th.height/2)
        };

        std::vector<Size> feature_map_sizes;
        feature_map_sizes.push_back(feature_map_3rd);
        feature_map_sizes.push_back(feature_map_4th);
        feature_map_sizes.push_back(feature_map_5th);
        feature_map_sizes.push_back(feature_map_6th);

        // Fixed params for generating priors
        const std::vector<std::vector<float>> min_sizes = {
            {10.0f,  16.0f,  24.0f},
            {32.0f,  48.0f},
            {64.0f,  96.0f},
            {128.0f, 192.0f, 256.0f}
        };
        CV_Assert(min_sizes.size() == feature_map_sizes.size()); // just to keep vectors in sync
        const std::vector<int> steps = { 8, 16, 32, 64 };

        // Generate priors
        priors.clear();
        for (size_t i = 0; i < feature_map_sizes.size(); ++i)
        {
            Size feature_map_size = feature_map_sizes[i];
            std::vector<float> min_size = min_sizes[i];

            for (int _h = 0; _h < feature_map_size.height; ++_h)
            {
                for (int _w = 0; _w < feature_map_size.width; ++_w)
                {
                    for (size_t j = 0; j < min_size.size(); ++j)
                    {
                        float s_kx = min_size[j] / inputW;
                        float s_ky = min_size[j] / inputH;

                        float cx = (_w + 0.5f) * steps[i] / inputW;
                        float cy = (_h + 0.5f) * steps[i] / inputH;

                        Rect2f prior = { cx, cy, s_kx, s_ky };
                        priors.push_back(prior);
                    }
                }
            }
        }
    }
}


#pragma once
#include <vector>
#include <memory>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include "shapes/vp_rect.h"

using namespace cv;

namespace vp_objects {
    // target in frame detected by face detectors such as yunet.
    // note: we can define new target type like vp_frame_xxx_target... if need (see vp_frame_pose_target also)
    class vp_frame_face_target
    {
    private:
        /* data */
    public:
        vp_frame_face_target(int x, 
                            int y, 
                            int width, 
                            int height, 
                            float score,
                            std::string labels = "",
                            cv::Mat result = cv::Mat(),
                            //std::vector<std::pair<int, int>> key_points = std::vector<std::pair<int, int>>(), 
                            std::vector<float> embeddings = std::vector<float>(),
                            std::vector<std::string> Label = std::vector<std::string>(),
                            std::vector<float> Scores = std::vector<float>());
        ~vp_frame_face_target();

        // x of top left
        int x;
        // y of top left
        int y;
        // width of rect
        int width;
        // height of rect
        int height;


        // confidence
        float score;
        std::string labels;
        cv::Mat result;

        // feature vector created by infer nodes such as vp_sface_feature_encoder_node.
        // embeddings can be used for face recognize or other reid works.
        std::vector<float> embeddings;
        
        std::vector<std::string> Label;
        std::vector<float> Scores;

        // key points (5 points or more)
        //std::vector<std::pair<int, int>> key_points;

        // track id filled by vp_track_node (child class) if it exists.
        int track_id = -1;

        // clone myself
        std::shared_ptr<vp_frame_face_target> clone();

        // rect area of target
        vp_rect get_rect() const;
    };

}

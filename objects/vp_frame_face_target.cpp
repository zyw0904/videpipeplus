#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include "vp_frame_face_target.h"

using namespace cv;

namespace vp_objects {
        
    vp_frame_face_target::vp_frame_face_target(int x, 
                                                int y, 
                                                int width, 
                                                int height, 
                                                float score,
                                                std::string labels,
                                                cv::Mat result, 
                                                //std::vector<std::pair<int, int>> key_points, 
                                                std::vector<float> embeddings,
                                                std::vector<std::string> Label,
                                                std::vector<float> Scores):
                                                x(x),
                                                y(y),
                                                width(width),
                                                height(height),
                                                score(score),
                                                labels(labels),
                                                result(result),
                                                //key_points(key_points),
                                                embeddings(embeddings),
                                                Label(Label),
                                                Scores(Scores) {
        
    }
    
    vp_frame_face_target::~vp_frame_face_target() {
    }

    std::shared_ptr<vp_frame_face_target> vp_frame_face_target::clone() {
        return std::make_shared<vp_frame_face_target>(*this);
    }

    vp_rect vp_frame_face_target::get_rect() const{
        return vp_rect(x, y, width, height);
    }
}

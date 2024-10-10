
#include <opencv2/imgproc.hpp>
#include "vp_face_osd_node.h"
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

namespace vp_nodes {
        
    vp_face_osd_node::vp_face_osd_node(std::string node_name): vp_node(node_name) {
        this->initialized();
    }
    
    vp_face_osd_node::~vp_face_osd_node()
    {
    }

    std::shared_ptr<vp_objects::vp_meta> vp_face_osd_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        // operations on osd_frame
        if (meta->osd_frame.empty()) {
            meta->osd_frame = meta->frame.clone();
        }
        auto& canvas = meta->osd_frame;
        
        // scan face targets
        for(auto& i : meta->face_targets) {
            if (i->x==0 && i->y==0 && i->width==0 && i->height==0) {
                if (i->score != 0) {
                    cv::putText(canvas, cv::format("Probability: %.2f", i->score), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1, 8, 0);
                }
                if (!i->labels.empty()) {
                    cv::putText(canvas, cv::format("Probability: %.2f", i->labels), cv::Point(10, 35), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1, 8, 0);
                }
                if(!i->result.empty()) {
                    //cv::Mat overlay;
                    //double alpha = 0.5; // 透明度
                    //cv::addWeighted(canvas, alpha, i->result, 1 - alpha, 0, overlay);
                    //canvas = overlay.clone();
                    cv::Mat output_img(canvas.rows, canvas.cols * 2, CV_8UC3);
                    cv::resize(i->result, i->result, cv::Size(canvas.cols, canvas.rows), 0, 0);
                    cv::hconcat(canvas, i->result, output_img);
                    canvas = output_img.clone();    
                }

            }
            else {
                cv::rectangle(canvas, cv::Rect(i->x, i->y, i->width, i->height), cv::Scalar(0, 255, 0), 2);
        
                // track_id
                if (i->track_id != -1) {
                    auto id = std::to_string(i->track_id);
                    cv::putText(canvas, id, cv::Point(i->x, i->y), 1, 1.5, cv::Scalar(0, 0, 255));
                }
                if(!i->Label.empty()) {
                    auto labels = i->Label;
                    cv::putText(canvas, cv::format("%s", labels[0].c_str()), cv::Point(i->x, i->y-5), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2, 8, 0);
                }
                if(!i->Scores.empty()) {
                    auto scores = i->Scores;
                    cv::putText(canvas, cv::format("%.2f", scores), cv::Point(i->x + i->width - 6, i->y-5), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2, 8, 0);
                }else{
                    cv::putText(canvas, cv::format("%.2f", i->score), cv::Point(i->x + i->width - 6, i->y-5), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2, 8, 0);
                }
                
            }

            // just handle 5 keypoints
            //if (i->key_points.size() >= 5) {
                //cv::circle(canvas, cv::Point(i->key_points[0].first, i->key_points[0].second), 2, cv::Scalar(255, 0, 0), 2);
                //cv::circle(canvas, cv::Point(i->key_points[1].first, i->key_points[1].second), 2, cv::Scalar(0, 0, 255), 2);
                //cv::circle(canvas, cv::Point(i->key_points[2].first, i->key_points[2].second), 2, cv::Scalar(0, 255, 0), 2);
                //cv::circle(canvas, cv::Point(i->key_points[3].first, i->key_points[3].second), 2, cv::Scalar(255, 0, 255), 2);
                //cv::circle(canvas, cv::Point(i->key_points[4].first, i->key_points[4].second), 2, cv::Scalar(0, 255, 255), 2);
            //}
        }

        return meta;
    }

    std::shared_ptr<vp_objects::vp_meta> vp_face_osd_node::handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) {
        return meta;
    }
}



#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include "vp_des_node.h"

namespace vp_nodes {
    // screen des node, display video on local window.
    class vp_sever_control_node: public vp_des_node
    {
    private:
        /* data */
        
        int last_btm_degree = 1024;
        int last_top_degree = 1024;
        float offset_dead_block = 0.5;
        int tempframe = 0;
        int tempjuli = 0;
        static void pythonTask(int next_btm_degree, int next_top_degree);
        
    protected:
        // re-implementation, return nullptr.
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override; 
        // re-implementation, return nullptr.
        virtual std::shared_ptr<vp_objects::vp_meta> handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) override;
    public:
        vp_sever_control_node(std::string node_name, 
                            int channel_index, 
                            bool osd = true,
                            vp_objects::vp_size display_w_h = {});
        ~vp_sever_control_node();

        // for osd frame
        bool osd;
        // display size
        vp_objects::vp_size display_w_h;

    };
}



#include "VP.h"

#include "../nodes/vp_src_node.h"
#include "../nodes/infers/vp_fire_detect_node_tpu.h"
#include "../nodes/osd/vp_face_osd_node.h"
#include "../nodes/vp_screen_des_node.h"

#include "../utils/analysis_board/vp_analysis_board.h"

#if _1_1_1_sample

int main() {
    
    
    VP_SET_LOG_INCLUDE_CODE_LOCATION(false);
    VP_SET_LOG_INCLUDE_THREAD_ID(false);
    VP_LOGGER_INIT();
    

    auto vp_screen_des_node = std::make_shared<vp_nodes::vp_screen_des_node>("vp_screen_des_node", 0);

    auto vp_face_osd_node = std::make_shared<vp_nodes::vp_face_osd_node>("vp_face_osd_node", 0);

    auto vp_fire_detect_node_tpu = std::make_shared<vp_nodes::vp_fire_detect_node_tpu>("vp_fire_detect_node_tpu", "222", "", "", "");

    auto vp_src_node = std::make_shared<vp_nodes::vp_src_node>("vp_src_node", 0, "222", 0.6);


    
    
    vp_fire_detect_node_tpu->attach_to({vp_src_node});
    
    vp_face_osd_node->attach_to({vp_fire_detect_node_tpu});
    
    
    
    
    vp_screen_des_node->attach_to({vp_face_osd_node});
    vp_src_node->start();
    
    vp_utils::vp_analysis_board board({vp_src_node});
    board.display();
}

#endif

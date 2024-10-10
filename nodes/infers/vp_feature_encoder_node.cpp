
#include "vp_feature_encoder_node.h"


namespace vp_nodes {
    
    vp_feature_encoder_node::vp_feature_encoder_node(std::string node_name, std::string model_path, std::string convert_type):
                                                    vp_secondary_infer_node(node_name, model_path, convert_type) {
        this->initialized();
    }
    
    vp_feature_encoder_node::~vp_feature_encoder_node() {
        
    } 
}

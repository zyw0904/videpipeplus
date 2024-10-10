
#include <python3.9/Python.h>
#include <iostream>
#include <future>
#include "vp_sever_control_node.h"
#include "../utils/vp_utils.h"

namespace vp_nodes {
    vp_sever_control_node::vp_sever_control_node(std::string node_name, 
                                            int channel_index, 
                                            bool osd,
                                            vp_objects::vp_size display_w_h):
                                            vp_des_node(node_name, channel_index),
                                            osd(osd),
                                            display_w_h(display_w_h) {
        
        this->initialized();
    }
    
    vp_sever_control_node::~vp_sever_control_node() {

    }
    
    void vp_sever_control_node::pythonTask(int next_btm_degree, int next_top_degree) {
        Py_Initialize();
        if(!Py_IsInitialized()){
            cout << "python init fail" << endl;
            return;
        }
        PyRun_SimpleString("import sys");
        PyRun_SimpleString("sys.path.append(r'/home/zyw/Desktop/SERVO_DRIVER_WITH_ESP32/examples/SCServo_Python/sms_sts')");
        PyRun_SimpleString("sys.path.append(r'/home/zyw/Desktop/SERVO_DRIVER_WITH_ESP32/examples/SCServo_Python/scservo_sdk')");
        PyRun_SimpleString("sys.path.append(r'/home/zyw/Desktop/SERVO_DRIVER_WITH_ESP32/examples/SCServo_Python')");
        PyObject *pModule_btm = NULL;
        PyObject *pFunc_btm = NULL;
        //PyRun_SimpleString("import btm_write");
        pModule_btm = PyImport_ImportModule("btm_write");
        pFunc_btm = PyObject_GetAttrString(pModule_btm, "btm_position");
        PyObject *pArgs_btm = PyTuple_New(1);
        PyTuple_SetItem(pArgs_btm, 0, Py_BuildValue("i", next_btm_degree));
        PyEval_CallObject(pFunc_btm, pArgs_btm);
        
        PyObject *pModule_top = NULL;
        PyObject *pFunc_top = NULL;
        //PyRun_SimpleString("import top_write");
        pModule_top = PyImport_ImportModule("top_write");
        pFunc_top = PyObject_GetAttrString(pModule_top, "top_position");
        PyObject *pArgs_top = PyTuple_New(1);
        PyTuple_SetItem(pArgs_top, 0, Py_BuildValue("i", next_top_degree));
        PyEval_CallObject(pFunc_top, pArgs_top);
        Py_Finalize();
    }

    // re-implementation, return nullptr.
    std::shared_ptr<vp_objects::vp_meta> 
        vp_sever_control_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
            
            cv::Mat resize_frame;
            if (this->display_w_h.width != 0 && this->display_w_h.height != 0) {                 
                cv::resize((osd && !meta->osd_frame.empty()) ? meta->osd_frame : meta->frame, resize_frame, cv::Size(display_w_h.width, display_w_h.height));
            }
            else {
                resize_frame = (osd && !meta->osd_frame.empty()) ? meta->osd_frame : meta->frame;
            }
            if(!meta->face_targets.empty()) {
                for(auto& i : meta->face_targets){
                    
                    if(i->x < 0){
                        i->x = 0;
                    }
                    if(i->x + i->width > 640){
                        i->width = 640 - i->x;
                    }
                    if(i->y < 0){
                        i->y = 0;
                    }
                    if(i->y + i->height > 640){
                        i->height = 640 - i->y;
                    }
                    float face_x = i->x + i->width / 2.0;
                    float face_y = i->y + i->height;
                    if(abs(tempjuli - face_x) < 50) {
                        break;
                    }
                    else {
                        //人脸在画面中心X轴上的偏移量
                        float offset_x = (face_x / resize_frame.cols - 0.5) * 2;
                        //人脸在画面中心Y轴上的偏移量
                        float offset_y = (face_y / resize_frame.rows - 0.5) * 2;
                        
                        //水平旋转
                        //设置最小阈值
                        if (abs(offset_x) < offset_dead_block){
                            offset_x = 0;
                        }
                        //offset范围在-50到50左右
                        int btm_delta_degree = offset_x * 20;
                        int Btm_delta_degree = (btm_delta_degree * 4096) / 360;
                        //计算得到新的底部舵机角度
                        int next_btm_degree = last_btm_degree + Btm_delta_degree;
                        if (next_btm_degree < 0){
                            next_btm_degree = 0;
                        }
                        else if(next_btm_degree > 2048){
                            next_btm_degree = 2048;
                        }
                        
                        
                        //垂直旋转
                        //设置最小阈值
                        if (abs(offset_y) < offset_dead_block){
                            offset_y = 0;
                        }
                        //offset范围在-50到50左右
                        int top_delta_degree = offset_y * 20;
                        int Top_delta_degree = (top_delta_degree * 4096) / 360;
                        //计算得到新的底部舵机角度
                        int next_top_degree = last_top_degree + Top_delta_degree;
                        if (next_top_degree < 0){
                            next_top_degree = 0;
                        }
                        else if(next_top_degree > 2048){
                            next_top_degree = 2048;
                        }
                                           
                        
                        //std::future<void> result = std::async(std::launch::async, pythonTask, next_btm_degree, next_top_degree);
                        //if (tempframe % 30 == 0) {
                        //if(last_btm_degree != next_btm_degree){
                        pythonTask(next_btm_degree, next_top_degree);
                        //}
                        last_btm_degree = next_btm_degree;
                        last_top_degree = next_top_degree; 

                        //}
                        //tempframe++;
                        //if (tempframe == 31){
                            //tempframe = 0;
                        //}
                        tempjuli = face_x;
                    }
                
                }
            }
            
            // for general works defined in base class
            return vp_des_node::handle_frame_meta(meta);
    }

    // re-implementation, return nullptr.
    std::shared_ptr<vp_objects::vp_meta> 
        vp_sever_control_node::handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) {
            // for general works defined in base class
            return vp_des_node::handle_control_meta(meta);
    }
}

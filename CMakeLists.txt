cmake_minimum_required(VERSION 3.12)

project(video_pipe)


add_definitions(-DAPI_EXPORTS)
add_definitions(-O2 -pthread)

# CUDA & OpenCV
# find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Python COMPONENTS Interpreter Development)

#onnxruntime
set(ONNXRUNTIME_ROOT_PATH "/usr/local/include/onnxruntime")
set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime
                             ${ONNXRUNTIME_ROOT_PATH}/onnxruntime
                             ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime/core/session)
set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_PATH}/build/Linux/Release/libonnxruntime.so)

set(InferenceEngine_DIR "/home/zyw/openvino/install/runtime/cmake")
#find_package(InferenceEngine REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Python_VERSION: ${Python_INCLUDE_DIRS}")
include_directories( ${PROJECT_SOURCE_DIR}/include ${Python_INCLUDE_DIRS} )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${Tensorflow_INCLUDE_DIRS} )
include_directories( ${CMAKE_SOURCE_DIR} )
include_directories("/home/zyw/openvino/install/runtime/include")
include_directories("/home/zyw/openvino/install/runtime/include/ie")
include_directories("/home/zyw/openvino/install/runtime/include/ngraph")
include_directories("/home/zyw/openvino/install/runtime/include/openvino")
include_directories("/home/zyw/openvino/install/samples/cpp/common/utils/include/samples")
include_directories("/home/zyw/openvino/install/samples/cpp/common/format_reader")
include_directories("/home/zyw/openvino/install/samples/cpp/common/utils/src")
include_directories("/home/zyw/dlib")
link_directories("/home/zyw/dlib/build/dlib")
#include_directories(${InferenceEngine_INCLUDE_DIRS})
include_directories( ${ONNXRUNTIME_INCLUDE_DIRS} )

# message(STATUS "CUDA library status:")
# message(STATUS "    version: ${CUDA_VERSION}")
# message(STATUS "    libraries: ${CUDA_LIBS}")
# message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lpthread -lstdc++ -lrt")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAB9 -DSTC_SERVER")

# trt_vehicle
# file(GLOB TRT_VEHICLE_SOURCES "third_party/trt_vehicle/**/*.cpp")
# cuda_add_library(trt_vehicle SHARED ${TRT_VEHICLE_SOURCES})
# target_link_libraries(trt_vehicle nvinfer cudart ${OpenCV_LIBS})

# paddle_ocr
# file(GLOB PADDLE_OCR_SOURCES "third_party/paddle_ocr/**/*.cpp")
# add_library(paddle_ocr SHARED ${PADDLE_OCR_SOURCES})
# target_link_libraries(paddle_ocr paddle_inference paddle2onnx)


file(GLOB NODES "nodes/*.cpp")
file(GLOB INFERS_NODES "nodes/infers/*.cpp")
file(GLOB OSD_NODES "nodes/osd/*.cpp")
file(GLOB TRACK "nodes/track/*.cpp")
file(GLOB SORT "nodes/track/sort/*.cpp")
file(GLOB OBJECTS "objects/*.cpp")
file(GLOB ELEMENTS_OBJECTS "objects/elements/*.cpp")
file(GLOB SHAPES_OBJECTS "objects/shapes/*.cpp")
file(GLOB BA "ba/*.cpp")
file(GLOB UTILS_ANALYSIS_BOARD "utils/analysis_board/*.cpp")
file(GLOB LOGGER "utils/logger/*.cpp")


add_library(video_pipe SHARED ${NODES} ${INFERS_NODES} ${OSD_NODES} ${TRACK} ${SORT} ${OBJECTS} ${ELEMENTS_OBJECTS} ${SHAPES_OBJECTS} ${BA} ${UTILS_ANALYSIS_BOARD} ${LOGGER})
target_link_libraries(video_pipe ${OpenCV_LIBS} -lstdc++fs)

target_link_libraries(video_pipe tensorflow-lite libpthread.so.0 -ldl)
target_link_libraries(video_pipe ${CMAKE_SOURCE_DIR}/libedgetpu.so.1.0 )
#target_link_libraries(video_pipe ${InferenceEngine_LIBRARIES} dl)
target_link_libraries(video_pipe ${Python_LIBRARIES})
target_link_libraries(video_pipe ${ONNXRUNTIME_LIB})

install(TARGETS video_pipe DESTINATION bin)

add_executable(1-1-1_sample sample/1-1-1_sample.cpp)
target_link_libraries(1-1-1_sample video_pipe ${OpenCV_LIBS})

add_executable(1-1-1_sample_tpu sample/1-1-1_sample_tpu.cpp)
target_link_libraries(1-1-1_sample_tpu video_pipe ${OpenCV_LIBS})

#add_executable(1-1-1_sample_vpu sample/1-1-1_sample_vpu.cpp)
#target_link_libraries(1-1-1_sample_vpu video_pipe ${OpenCV_LIBS})

add_executable(face_tracking_sample_tpu sample/face_tracking_sample_tpu.cpp)
target_link_libraries(face_tracking_sample_tpu video_pipe ${OpenCV_LIBS})

add_executable(object_sample_tpu sample/object_sample_tpu.cpp)
target_link_libraries(object_sample_tpu video_pipe ${OpenCV_LIBS})

add_executable(plate_detect_sample_tpu sample/plate_detect_sample_tpu.cpp)
target_link_libraries(plate_detect_sample_tpu video_pipe ${OpenCV_LIBS})

#add_executable(1-1-N_sample sample/1-1-N_sample.cpp)
#target_link_libraries(1-1-N_sample video_pipe ${OpenCV_LIBS})

#add_executable(1-N-N_sample sample/1-N-N_sample.cpp)
#target_link_libraries(1-N-N_sample video_pipe ${OpenCV_LIBS})

#add_executable(N-1-N_sample sample/N-1-N_sample.cpp)
#target_link_libraries(N-1-N_sample video_pipe ${OpenCV_LIBS})

#add_executable(N-N_sample sample/N-N_sample.cpp)
#target_link_libraries(N-N_sample video_pipe ${OpenCV_LIBS})

#add_executable(paddle_infer_sample sample/paddle_infer_sample.cpp)
#target_link_libraries(paddle_infer_sample video_pipe ${OpenCV_LIBS})

#add_executable(src_des_sample sample/src_des_sample.cpp)
#target_link_libraries(src_des_sample video_pipe ${OpenCV_LIBS})

#add_executable(trt_infer_sample sample/trt_infer_sample.cpp)
#target_link_libraries(trt_infer_sample video_pipe ${OpenCV_LIBS})

#add_executable(vp_logger_sample sample/vp_logger_sample.cpp)
#target_link_libraries(vp_logger_sample video_pipe ${OpenCV_LIBS})

add_executable(face_tracking_sample sample/face_tracking_sample.cpp)
target_link_libraries(face_tracking_sample video_pipe ${OpenCV_LIBS})

#add_executable(interaction_with_pipe_sample sample/interaction_with_pipe_sample.cpp)
#target_link_libraries(interaction_with_pipe_sample video_pipe ${OpenCV_LIBS})

#add_executable(message_broker_sample sample/message_broker_sample.cpp)
#target_link_libraries(message_broker_sample video_pipe ${OpenCV_LIBS})

#add_executable(message_broker_sample2 sample/message_broker_sample2.cpp)
#target_link_libraries(message_broker_sample2 video_pipe ${OpenCV_LIBS})

#add_executable(vehicle_tracking_sample sample/vehicle_tracking_sample.cpp)
#target_link_libraries(vehicle_tracking_sample video_pipe ${OpenCV_LIBS})

#add_executable(record_sample sample/record_sample.cpp)
#target_link_libraries(record_sample video_pipe ${OpenCV_LIBS})

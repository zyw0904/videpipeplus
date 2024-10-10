import subprocess
import onnx
import tensorflow as tf
import torch
from ultralytics import YOLO

def pt_to_edgetpu(models):
    import os
    try:
        try:
            model = YOLO(models)
            results = model.export(format='tflite', imgsz=224, int8=True)
            c = f'edgetpu_compiler {results}'
            subprocess.run(c, shell=True, check=True)
            edgetpu_model = str(results).rsplit("/", 1)[1].rsplit('.', 1)[0] + "_edgetpu.tflite"
            edgetpu_path = os.path.abspath('.') + "/" + edgetpu_model
            return edgetpu_path
        except Exception as e:
            try:
                import sys
                sys.path.append(f"{os.path.abspath('.')}/onnx2tflite")
                from converter import onnx_converter
                c = f"python3 {os.path.abspath('.')}/yolov7/export.py --weights {models} --img-size 224 --int8"
                subprocess.run(c, shell=True, check=True)
                onnx_path = models.rsplit(".", 1)[0] + ".onnx"
                onnx_converter(
                    onnx_model_path=onnx_path,
                    need_simplify=True,
                    output_path="",  # 输出的tflite存储路径
                    target_formats=['tflite'],  # or ['keras'], ['keras', 'tflite']
                    weight_quant=False,
                    int8_model=False,
                    int8_mean=None,
                    int8_std=None,
                    image_root=None
                )
                try:
                    tflite_file = onnx_path.rsplit("/", 1)[1].rsplit('.', 1)[0] + ".tflite"
                except Exception as e:
                    tflite_file = onnx_path.rsplit('.', 1)[0] + ".tflite"
                tflite_path = os.path.abspath('.') + "/" + tflite_file
                c = f'edgetpu_compiler {tflite_path}'
                subprocess.run(c, shell=True, check=True)
                edgetpu_model = tflite_path.rsplit("/", 1)[1].rsplit('.', 1)[0] + "_edgetpu.tflite"
                edgetpu_path = os.path.abspath('.') + "/" + edgetpu_model
                return edgetpu_path
            except Exception as e:
                c = f"python3 {os.path.abspath('.')}/yolov5/export.py --weights {models} --imgsz 224 --int8 --include tflite"
                subprocess.run(c, shell=True, check=True)
                tflite_path = models.rsplit(".", 1)[0] + "-int8.tflite"
                c = f'edgetpu_compiler {tflite_path}'
                subprocess.run(c, shell=True, check=True)
                try:
                    edgetpu_model = tflite_path.rsplit("/", 1)[1].rsplit('.', 1)[0] + "_edgetpu.tflite"
                except Exception as e:
                    edgetpu_model = tflite_path.rsplit('.', 1)[0] + "_edgetpu.tflite"
                edgetpu_path = os.path.abspath('.') + "/" + edgetpu_model
                return edgetpu_path
    except Exception as e:
        import torch
        import torch.nn
        import onnx
        import sys
        sys.path.append(f"{os.path.abspath('.')}/onnx2tflite")
        from converter import onnx_converter

        model = torch.load(models)
        model.eval()
        input_names = ['input']
        output_names = ['output']
        x = torch.randn(1, 3, 224, 224, requires_grad=True)
        torch.onnx.export(model, x, 'best.onnx', input_names=input_names, output_names=output_names, verbose='True')
        onnx_converter(
                    onnx_model_path='best.onnx',
                    need_simplify=True,
                    output_path="",  # 输出的tflite存储路径
                    target_formats=['tflite'],  # or ['keras'], ['keras', 'tflite']
                    weight_quant=False,
                    int8_model=False,
                    int8_mean=None,
                    int8_std=None,
                    image_root=None
        )
        c = f'edgetpu_compiler best.tflite'
        subprocess.run(c, shell=True, check=True)
        edgetpu_path = os.path.abspath('.') + "//best_edgetpu.tflite"
        return edgetpu_path

def onnx_to_edgetpu(models):
    import os
    import sys
    sys.path.append(f"{os.path.abspath('.')}/onnx2tflite")
    from converter import onnx_converter
    import onnx
    from onnx import shape_inference

    try:
        onnx_model = onnx.load(models)
        # Determine the fixed input shape
        input_shape = (1, 3, 224, 224)  # Example input shape (batch_size, channels, height, width)
        # Modify the input shapes of the model to the fixed shape
        for input in onnx_model.graph.input:
            input.type.tensor_type.shape.dim[0].dim_value = input_shape[0]
            input.type.tensor_type.shape.dim[1].dim_value = input_shape[1]
            input.type.tensor_type.shape.dim[2].dim_value = input_shape[2]
            input.type.tensor_type.shape.dim[3].dim_value = input_shape[3]

        # Optionally, run shape inference to update other shapes in the model
        onnx_model = shape_inference.infer_shapes(onnx_model)
        # Save the modified model
        output_model_path = 'modified_model.onnx'
        onnx.save(onnx_model, output_model_path)
        onnx_converter(
            onnx_model_path=output_model_path,
            need_simplify=True,
            output_path="",  # 输出的tflite存储路径
            target_formats=['tflite'],  # or ['keras'], ['keras', 'tflite']
            weight_quant=False,
            int8_model=True,
            int8_mean=None,
            int8_std=None,
            image_root=None
        )
        tflite_path = output_model_path.rsplit(".", 1)[0] + ".tflite"
        c = f'edgetpu_compiler {tflite_path}'
        subprocess.run(c, shell=True, check=True)
        edgetpu_model = tflite_path.rsplit('.', 1)[0] + "_edgetpu.tflite"
        edgetpu_path = os.path.abspath('.') + "/" + edgetpu_model
        return edgetpu_path
    except Exception as e:
        import os
        c2 = f'onnx2tf -i {models} -oiqt'
        subprocess.run(c2, shell=True, check=True)
        saved_model_dir = os.path.abspath('.') + '/saved_model'
        save_model_to_tflite(saved_model_dir)
        tflite_path = os.path.abspath('.') + '/saved_model.tflite'
        c = f'edgetpu_compiler {tflite_path}'
        subprocess.run(c, shell=True, check=True)
        edgetpu_model = tflite_path.rsplit("/", 1)[1].rsplit('.', 1)[0] + "_edgetpu.tflite"
        edgetpu_path = os.path.abspath('.') + "/" + edgetpu_model
        return edgetpu_path

def caffe_to_edgetpu(prototxt_file, caffe_model):#pip install caffe2onnx
    import os
    import glob
    onnx_model = caffe_model.rsplit(".", 1)[0] + ".onnx"
    c1 = f'python3 -m caffe2onnx.convert --prototxt {prototxt_file} --caffemodel {caffe_model} --onnx {onnx_model}'
    subprocess.run(c1, shell=True, check=True)
    c2 = f'onnx2tf -i {onnx_model} -oiqt'
    subprocess.run(c2, shell=True, check=True)
    saved_model_dir1 = os.path.abspath('.') + '/saved_model'
    save_model_to_tflite(saved_model_dir1)
    tflite_path = os.path.abspath('.') + '/saved_model.tflite'
    c = f'edgetpu_compiler {tflite_path}'
    subprocess.run(c, shell=True, check=True)
    edgetpu_model = tflite_path.rsplit("/", 1)[1].rsplit('.', 1)[0] + "_edgetpu.tflite"
    edgetpu_path = os.path.abspath('.') + "/" + edgetpu_model
    return edgetpu_path

def edgetpu_to_onnx(models):#pip install tf2onnx
    import os
    try:
        onnx_file = models.rsplit("/", 1)[1].rsplit('.', 1)[0] + ".onnx"
    except Exception as e:
        onnx_file = models.rsplit('.', 1)[0] + ".onnx"
    onnx_path = os.path.abspath('.') + "/" + onnx_file
    c = f'python3 -m tf2onnx.convert --tflite {models} --output {onnx_path}'
    subprocess.run(c, shell=True, check=True)
    return onnx_path

def edgetpu_to_trt(models, flop=16):
    import os
    import tensorrt as trt
    onnx_path = edgetpu_to_onnx(models)
    try:
        engine_model = onnx_path.rsplit("/", 1)[1].rsplit('.', 1)[0] + ".engine"
    except Exception as e:
        engine_model = onnx_path.rsplit('.', 1)[0] + ".engine"
    engine_file_path = os.path.abspath('.') + "/" + engine_model
    trt_logger = trt.Logger(trt.Logger.VERBOSE)  # trt.Logger.ERROR
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )

    parser = trt.OnnxParser(network, trt_logger)
    # parse ONNX
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    print("Completed parsing ONNX file")
    builder.max_workspace_size = 2 << 30
    # default = 1 for fixed batch size
    builder.max_batch_size = 1
    # set mixed flop computation for the best performance
    if builder.platform_has_fast_fp16 and flop == 16:
        builder.fp16_mode = True

    if os.path.isfile(engine_file_path):
        try:
            os.remove(engine_file_path)
        except Exception:
            print("Cannot remove existing file: ",
                  engine_file_path)
    print("Creating Tensorrt Engine")

    config = builder.create_builder_config()
    config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS))
    config.max_workspace_size = 2 << 30
    config.set_flag(trt.BuilderFlag.FP16)

    engine = builder.build_engine(network, config)
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())
    print("Serialized Engine Saved at: ", engine_file_path)
    return engine_file_path

def edgetpu_to_pt(models):
    import onnx
    import torch
    from onnx2torch import convert
    onnx_path = edgetpu_to_onnx(models)
    torch_model = convert(onnx_path)
    return torch_model

def xml_to_edgetpu(models):#pips install openvino2tensorflow
    import os
    c1 = f'openvino2tensorflow   --model_path {models}  --output_saved_model --output_pb  --output_weight_quant_tflite --output_float16_quant_tflite'
    subprocess.run(c1, shell=True, check=True)
    c2 = f'edgetpu_compiler saved_model/model_float16_quant.tflite'
    subprocess.run(c2, shell=True, check=True)
    edgetpu_path = os.path.abspath('.') + "/model_float16_quant_edgetpu.tflite"
    return edgetpu_path

def edgetpu_to_xml(models):#pips install openvino
    import openvino
    import os
    try:
        xml_file = models.rsplit("/", 1)[1].rsplit('.', 1)[0] + ".xml"
    except Exception as e:
        xml_file = models.rsplit('.', 1)[0] + ".xml"
    c = f'ovc {models}'
    subprocess.run(c, shell=True, check=True)
    xml_path = os.path.abspath('.') + "/" + xml_file
    return xml_path

def xml_to_onnx(models):#pips install openvino2tensorflow
    import os
    c1 = f'openvino2tensorflow   --model_path {models}  --output_saved_model --output_pb --output_onnx'
    subprocess.run(c1, shell=True, check=True)
    onnx_path = os.path.abspath('.') + "/saved_model/model_float32.onnx"
    return onnx_path

def onnx_to_xml(models):#pips install openvino
    import openvino
    import os
    try:
        xml_file = models.rsplit("/", 1)[1].rsplit('.', 1)[0] + ".xml"
    except Exception as e:
        xml_file = models.rsplit('.', 1)[0] + ".xml"
    c = f'ovc {models}'
    subprocess.run(c, shell=True, check=True)
    xml_path = os.path.abspath('.') + "/" + xml_file
    return xml_path

def save_model_to_tflite(models):
    import tensorflow as tf
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(models)  # path to the SavedModel directory
    tflite_model = converter.convert()
    # Save the model.
    with open('saved_model.tflite', 'wb') as f:
        f.write(tflite_model)

def save_model_to_edgetpu(models):
    import os
    import tensorflow as tf
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(models)  # path to the SavedModel directory
    tflite_model = converter.convert()
    # Save the model.
    with open('Saved_Model.tflite', 'wb') as f:
        f.write(tflite_model)
    c = f'edgetpu_compiler Saved_Model.tflite'
    subprocess.run(c, shell=True, check=True)
    edgetpu_path = os.path.abspath('.') + "/Saved_Model_edgetpu.tflite"
    return edgetpu_path


def kreas_to_edgetpu(models):
    import os
    import tensorflow as tf
    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(models)
    tflite_model = converter.convert()
    # Save the model.
    with open('kreas_model.tflite', 'wb') as f:
        f.write(tflite_model)
    c = f'edgetpu_compiler kreas_model.tflite'
    subprocess.run(c, shell=True, check=True)
    edgetpu_path = os.path.abspath('.') + "/kreas_model_edgetpu.tflite"
    return edgetpu_path

def tflite_to_edgetpu(models):
    import os
    try:
        edgetpu_file = models.rsplit("/", 1)[1].rsplit('.', 1)[0] + "_edgetpu.tflite"
    except Exception as e:
        edgetpu_file = models.rsplit('.', 1)[0] + "_edgetpu.tflite"
    c = f'edgetpu_compiler {models}'
    subprocess.run(c, shell=True, check=True)
    edgetpu_path = os.path.abspath('.') + "/" + edgetpu_file
    return edgetpu_path


def run_convert(models, config_file, convert_models):
    try:
        suffix = models.rsplit(".", 1)[1]
        if suffix == 'pt':
            return pt_to_edgetpu(models)
        elif suffix == 'caffemodel':
            return caffe_to_edgetpu(config_file, models)
        elif suffix == 'onnx':
            if convert_models == 'edgetpu':
                return onnx_to_edgetpu(models)
            elif convert_models == 'xml':
                return onnx_to_xml(models)
        elif suffix == 'tflite':
            if convert_models == 'onnx':
                return edgetpu_to_onnx(models)
            elif convert_models == 'pt':
                return edgetpu_to_pt(models)
            elif convert_models == 'trt':
                return edgetpu_to_trt(models)
            elif convert_models == 'xml':
                return edgetpu_to_xml(models)
            elif convert_models == 'edgetpu':
                return tflite_to_edgetpu(models)
        elif suffix == 'xml':
            if convert_models == 'edgetpu':
                return xml_to_edgetpu(models)
            elif convert_models == 'onnx':
                return xml_to_onnx(models)
        elif suffix == 'h5':
            return kreas_to_edgetpu(models)
    except Exception as e:
        suffix = models.rsplit("/", 1)[1]
        if suffix == 'saved_model':
            return save_model_to_edgetpu(models)

#if __name__ == "__main__":
#     config_file = r'/home/zyw/桌面/caffe2onnx-main/test/caffemodel/mobilenet/mobilenet.prototxt'
#     models = r'/home/zyw/桌面/firesmoke-fp16.tflite'
#     saved_model_dir = '/home/zyw/PycharmProjects/pythonProject/saved_model'
#     convert_models = 'edgetpu'
#     a = run_convert(models, config_file, convert_models)
#     print(a)



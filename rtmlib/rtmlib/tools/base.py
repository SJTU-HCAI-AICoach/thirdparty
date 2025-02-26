import os
from abc import ABCMeta, abstractmethod
from typing import Any

import cv2
import numpy as np

from .file import download_checkpoint
import tensorrt as trt
import pycuda.driver as cuda


def check_mps_support():
    try:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        return 'MPSExecutionProvider' in providers or 'CoreMLExecutionProvider' in providers
    except ImportError:
        return False

RTMLIB_SETTINGS = {
    'opencv': {
        'cpu': (cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU),

        # You need to manually build OpenCV through cmake
        'cuda': (cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA)
    },
    'onnxruntime': {
        'cpu': 'CPUExecutionProvider',
        'cuda': 'CUDAExecutionProvider',
        'mps': 'CoreMLExecutionProvider' if check_mps_support() else 'CPUExecutionProvider'
    },
}


class BaseTool(metaclass=ABCMeta):

    def __init__(self,
                 onnx_model: str = None,
                 model_input_size: tuple = None,
                 mean: tuple = None,
                 std: tuple = None,
                 backend: str = 'opencv',
                 device: str = 'cpu'):

        # if not os.path.exists(onnx_model):
        #     onnx_model = download_checkpoint(onnx_model)

        if backend == 'opencv':
            try:
                providers = RTMLIB_SETTINGS[backend][device]

                session = cv2.dnn.readNetFromONNX(onnx_model)
                session.setPreferableBackend(providers[0])
                session.setPreferableTarget(providers[1])
                self.session = session
            except Exception:
                raise RuntimeError(
                    'This model is not supported by OpenCV'
                    ' backend, please use `pip install'
                    ' onnxruntime` or `pip install'
                    ' onnxruntime-gpu` to install onnxruntime'
                    ' backend. Then specify `backend=onnxruntime`.')  # noqa

        elif backend == 'onnxruntime':
            import onnxruntime as ort
            providers = RTMLIB_SETTINGS[backend][device]
            self.session = ort.InferenceSession(path_or_bytes=onnx_model,
                                                providers=[providers])
        elif backend == 'openvino':
            from openvino.runtime import Core
            core = Core()
            model_onnx = core.read_model(model=onnx_model)

            if device != 'cpu':
                print('OpenVINO only supports CPU backend, automatically'
                      ' switched to CPU backend.')
            self.compiled_model = core.compile_model(
                model=model_onnx,
                device_name='CPU',
                config={'PERFORMANCE_HINT': 'LATENCY'})
            self.input_layer = self.compiled_model.input(0)
            self.output_layer0 = self.compiled_model.output(0)
            self.output_layer1 = self.compiled_model.output(1)
        elif backend == 'tensorrt':
    
            
            engine_path = onnx_model.replace('.onnx', '.engine')
            assert os.path.exists(engine_path)
            logger = trt.Logger(trt.Logger.WARNING)
            logger.min_severity = trt.Logger.Severity.ERROR
            runtime = trt.Runtime(logger)
            trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins
            with open(engine_path, "rb") as f:
                serialized_engine = f.read()

            self.engine = runtime.deserialize_cuda_engine(serialized_engine)
            self.context = self.engine.create_execution_context()
            self.inputs, self.outputs, self.bindings = [], [], []
            self.stream = cuda.Stream()
            for idx,binding in enumerate(self.engine):
                size = trt.volume(self.engine.get_tensor_shape(binding))
                dims = self.engine.get_tensor_shape(binding)
                if dims[1] < 0:
                    size *= -1
                dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                self.bindings.append(int(device_mem))
                name = self.engine.get_tensor_name(idx)
                if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                    self.inputs.append({'host': host_mem, 'device': device_mem, "name": name})
                else :
                    host_mem = cuda.pagelocked_empty(size * 10, dtype)
                    device_mem = cuda.mem_alloc(host_mem.nbytes)
                    self.outputs.append({'host': host_mem, 'device': device_mem, "name": name})
        else:
            raise NotImplementedError

        print(f'load {onnx_model} with {backend} backend')

        self.onnx_model = onnx_model
        self.model_input_size = model_input_size
        self.mean = mean
        self.std = std
        self.backend = backend
        self.device = device

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Implement the actual function here."""
        raise NotImplementedError

    def inference(self, img: np.ndarray):
        """Inference model.

        Args:
            img (np.ndarray): Input image in shape.

        Returns:
            outputs (np.ndarray): Output of RTMPose model.
        """
        # build input to (1, 3, H, W)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        input = img[None, :, :, :]

        # run model
        if self.backend == 'opencv':
            outNames = self.session.getUnconnectedOutLayersNames()
            self.session.setInput(input)
            outputs = self.session.forward(outNames)
        elif self.backend == 'onnxruntime':
            sess_input = {self.session.get_inputs()[0].name: input}
            sess_output = []
            for out in self.session.get_outputs():
                sess_output.append(out.name)

            outputs = self.session.run(sess_output, sess_input)
        elif self.backend == 'openvino':
            results = self.compiled_model(input)
            output0 = results[self.output_layer0]
            output1 = results[self.output_layer1]
            outputs = [output0, output1]
        # elif self.backend == 'tensorrt':
        #     import pycuda.driver as cuda
        #     self.inputs[0]['host'] = np.ravel(img)
        #     # transfer data to the gpu
        #     for inp in self.inputs:
        #         cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        #     # run inference
        #     #  |      execute_async_v3(self: tensorrt_bindings.tensorrt.IExecutionContext, stream_handle: int) -> bool
        #     #  |      execute_v2(self: tensorrt_bindings.tensorrt.IExecutionContext, bindings: List[int]) -> bool
        #     for binding_index, input_dict in enumerate(self.inputs):
        #         binding_name = input_dict["name"]
        #         self.context.set_input_shape(binding_name, input.shape)
        #         self.context.set_tensor_address(binding_name, input_dict['device'])
        #     for binding_index, output_dict in enumerate(self.outputs):
        #         binding_name = output_dict["name"]
        #         #self.context.set_input_shape(binding_name, [1] + list(img.shape))
        #         self.context.set_tensor_address(binding_name, output_dict['device'])
        #     self.context.execute_async_v3(
        #         #bindings=self.bindings,
        #         stream_handle=self.stream.handle)
        #     # fetch outputs from gpu
        #     for out in self.outputs:
        #         cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        #     # synchronize stream
        #     self.stream.synchronize()
        #     outputs = [np.array(out['host'])[None].reshape(1, 10,-1) for out in self.outputs]
        elif self.backend == 'tensorrt':
          
            self.inputs[0]['host'] = np.ravel(img)
            # transfer data to the gpu
            for inp in self.inputs:
                cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
            # run inference
            #  |      execute_async_v3(self: tensorrt_bindings.tensorrt.IExecutionContext, stream_handle: int) -> bool
            #  |      execute_v2(self: tensorrt_bindings.tensorrt.IExecutionContext, bindings: List[int]) -> bool
            for binding_index, input_dict in enumerate(self.inputs):
                binding_name = input_dict["name"]
                self.context.set_input_shape(binding_name, input.shape)
                self.context.set_tensor_address(binding_name, input_dict['device'])
            for binding_index, output_dict in enumerate(self.outputs):
                binding_name = output_dict["name"]
                #self.context.set_input_shape(binding_name, [1] + list(img.shape))
                self.context.set_tensor_address(binding_name, output_dict['device'])
            self.context.execute_async_v3(
                #bindings=self.bindings,
                stream_handle=self.stream.handle)
            # fetch outputs from gpu
            for out in self.outputs:
                cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
            # synchronize stream
            self.stream.synchronize()
            outputs = [np.array(out['host'])[None].reshape(1, 10,-1) for out in self.outputs]
        return outputs
    def batch_inference(self, img: np.ndarray):
        # build input to (1, 3, H, W)
        img = np.ascontiguousarray(img, dtype=np.float32) 
        input = img.transpose(0,2,3,1)
       
        self.inputs[0]['host'] = np.ravel(img)
        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        #  |      execute_async_v3(self: tensorrt_bindings.tensorrt.IExecutionContext, stream_handle: int) -> bool
        #  |      execute_v2(self: tensorrt_bindings.tensorrt.IExecutionContext, bindings: List[int]) -> bool
        for binding_index, input_dict in enumerate(self.inputs):
            binding_name = input_dict["name"]
            self.context.set_input_shape(binding_name, input.shape)
            self.context.set_tensor_address(binding_name, input_dict['device'])
        for binding_index, output_dict in enumerate(self.outputs):
            binding_name = output_dict["name"]
            #self.context.set_input_shape(binding_name, [1] + list(img.shape))
            self.context.set_tensor_address(binding_name, output_dict['device'])
        self.context.execute_async_v3(
            #bindings=self.bindings,
            stream_handle=self.stream.handle)
        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()

        outputs = [np.array(out['host'])[None].reshape(1, 10,-1) for out in self.outputs]
        return outputs

class YOLOXTensorRT(metaclass=ABCMeta):

    def __init__(self,
                 onnx_model: str = None,
                 model_input_size: tuple = None,
                 mean: tuple = None,
                 std: tuple = None,
                 backend: str = 'opencv',
                 device: str = 'cpu'):

        if backend == 'tensorrt':
        
            
            engine_path = onnx_model.replace('.onnx', '.engine')
            assert os.path.exists(engine_path)
            logger = trt.Logger(trt.Logger.WARNING)
            logger.min_severity = trt.Logger.Severity.ERROR
            runtime = trt.Runtime(logger)
            trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins
            with open(engine_path, "rb") as f:
                serialized_engine = f.read()

            self.engine = runtime.deserialize_cuda_engine(serialized_engine)
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()
        else:
            raise NotImplementedError

        print(f'load {onnx_model} with {backend} backend')

        self.onnx_model = onnx_model
        self.model_input_size = model_input_size
        self.mean = mean
        self.std = std
        self.backend = backend
        self.device = device

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Implement the actual function here."""
        raise NotImplementedError

    def inference(self, img):
        # build input to (1, 3, H, W)
        n = len(img)
        img = np.stack(img, axis=0).transpose(0,3,1,2)
        img = np.ascontiguousarray(img, dtype=np.float32)
        input = img

      

        inputs_host = np.ravel(np.ascontiguousarray(img))
        inputs_device = cuda.mem_alloc(inputs_host.nbytes)


        per_batch = 100

        dets_host = cuda.pagelocked_empty(n*5 * per_batch, np.float32)
        labels_host = cuda.pagelocked_empty(n*1 * per_batch, np.int32)

        dets_device = cuda.mem_alloc(dets_host.nbytes)
        labels_device = cuda.mem_alloc(labels_host.nbytes)

        cuda.memcpy_htod_async(inputs_device, inputs_host, self.stream)

        self.context.set_binding_shape(0, input.shape)
        self.context.set_input_shape("input", input.shape)
        self.context.set_tensor_address("input", inputs_device)
        self.context.set_tensor_address("dets", dets_device)
        self.context.set_tensor_address("labels", labels_device)

        self.context.execute_async_v3(
            stream_handle=self.stream.handle)

        cuda.memcpy_dtoh_async(dets_host, dets_device, self.stream)
        cuda.memcpy_dtoh_async(labels_host, labels_device, self.stream)

        # synchronize stream
        self.stream.synchronize()
        output_dets = np.array(dets_host).reshape(n, 100, 5)
        output_labels = np.array(labels_host)
        return output_dets



class VitPoseTensorRT(metaclass=ABCMeta):

    def __init__(self,
                 onnx_model: str = None,
                 model_input_size: tuple = None,
                 mean: tuple = None,
                 std: tuple = None,
                 backend: str = 'opencv',
                 device: str = 'cpu'):

        if backend == 'tensorrt':
            
            
            engine_path = onnx_model.replace('.onnx', '.engine')
            assert os.path.exists(engine_path)
            logger = trt.Logger(trt.Logger.WARNING)
            logger.min_severity = trt.Logger.Severity.ERROR
            runtime = trt.Runtime(logger)
            trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins
            with open(engine_path, "rb") as f:
                serialized_engine = f.read()

            self.engine = runtime.deserialize_cuda_engine(serialized_engine)
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()
        else:
            raise NotImplementedError

        print(f'load {onnx_model} with {backend} backend')

        self.onnx_model = onnx_model
        self.model_input_size = model_input_size
        self.mean = mean
        self.std = std
        self.backend = backend
        self.device = device

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Implement the actual function here."""
        raise NotImplementedError

    def inference(self, img):
        # build input to (1, 3, H, W)
        n = len(img)
        img = np.stack(img, axis=0).transpose(0,3,1,2)
        img = np.ascontiguousarray(img, dtype=np.float32)
        input = img


        inputs_host = np.ravel(np.ascontiguousarray(img))
        inputs_device = cuda.mem_alloc(inputs_host.nbytes)

        outputs_host = cuda.pagelocked_empty(n*17*64*48, np.float32)
        outputs_device = cuda.mem_alloc(outputs_host.nbytes)

        cuda.memcpy_htod_async(inputs_device, inputs_host, self.stream)

        self.context.set_binding_shape(0, input.shape)
        self.context.set_input_shape("input", input.shape)
        self.context.set_tensor_address("input", inputs_device)
        self.context.set_tensor_address("output", outputs_device)

        self.context.execute_async_v3(
            stream_handle=self.stream.handle)

        cuda.memcpy_dtoh_async(outputs_host, outputs_device, self.stream)

        # synchronize stream
        self.stream.synchronize()
        output_feats = np.array(outputs_host).reshape(n, 17, 64, 48)
        return output_feats


class OSNetTensorRT(metaclass=ABCMeta):

    def __init__(self,
                 onnx_model: str = None,
                 model_input_size: tuple = None,
                 mean: tuple = None,
                 std: tuple = None,
                 backend: str = 'opencv',
                 device: str = 'cpu'):

        if backend == 'tensorrt':
       
            
            engine_path = onnx_model.replace('.onnx', '.engine')
            assert os.path.exists(engine_path)
            logger = trt.Logger(trt.Logger.WARNING)
            logger.min_severity = trt.Logger.Severity.ERROR
            runtime = trt.Runtime(logger)
            trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins
            with open(engine_path, "rb") as f:
                serialized_engine = f.read()

            self.engine = runtime.deserialize_cuda_engine(serialized_engine)
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()
        else:
            raise NotImplementedError

        print(f'load {onnx_model} with {backend} backend')

        self.onnx_model = onnx_model
        self.model_input_size = model_input_size
        self.mean = mean
        self.std = std
        self.backend = backend
        self.device = device

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Implement the actual function here."""
        raise NotImplementedError

    def inference(self, img):
        # build input to (1, 3, H, W)
        n = len(img)
        img = np.stack(img, axis=0).transpose(0,3,1,2)
        img = np.ascontiguousarray(img, dtype=np.float32)
        input = img

     

        inputs_host = np.ravel(np.ascontiguousarray(img))
        inputs_device = cuda.mem_alloc(inputs_host.nbytes)

        outputs_host = cuda.pagelocked_empty(n*512, np.float32)
        outputs_device = cuda.mem_alloc(outputs_host.nbytes)

        cuda.memcpy_htod_async(inputs_device, inputs_host, self.stream)

        self.context.set_binding_shape(0, input.shape)
        self.context.set_input_shape("base_images", input.shape)
        self.context.set_tensor_address("base_images", inputs_device)
        self.context.set_tensor_address("features", outputs_device)

        self.context.execute_async_v3(
            stream_handle=self.stream.handle)

        cuda.memcpy_dtoh_async(outputs_host, outputs_device, self.stream)

        # synchronize stream
        self.stream.synchronize()
        output_feats = np.array(outputs_host).reshape(n, 512)
        return output_feats
   
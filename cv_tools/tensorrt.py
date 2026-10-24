import tempfile
import time

import torch


class Output(torch.nn.Module):
    def __init__(self, model, name):
        super().__init__()
        self.model, self.name = model, name

    def forward(self, pixel_values):
        return getattr(self.model(pixel_values=pixel_values), self.name)


class Engine:
    def __init__(self, path):
        import tensorrt as trt

        dtypes = {trt.float16: torch.float16, trt.float32: torch.float32}
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Cannot deserialize TensorRT engine: {path}; delete it to rebuild")
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        output = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        self.shape = tuple(self.engine.get_tensor_shape(self.input))
        self.dtype = dtypes[self.engine.get_tensor_dtype(self.input)]
        self.output = torch.empty(tuple(self.engine.get_tensor_shape(output)), device="cuda",
                                  dtype=dtypes[self.engine.get_tensor_dtype(output)])
        self.context.set_tensor_address(output, self.output.data_ptr())

    def __call__(self, pixels):
        if tuple(pixels.shape) != self.shape or pixels.dtype != self.dtype:
            raise ValueError(f"Engine requires {self.shape} {self.dtype}; got {tuple(pixels.shape)} {pixels.dtype}")
        self.context.set_tensor_address(self.input, pixels.data_ptr())
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        # [Agent] The output buffer is reused; callers consume it before the next call.
        return self.output


def create_engine(model, output, size, path):
    import tensorrt as trt

    print(f"Building TensorRT engine {path.name}, once per model and input size", flush=True)
    started = time.perf_counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    # [Agent] The ONNX export is only a build input and weighs as much as the model; keep it in a temp dir.
    with tempfile.TemporaryDirectory(dir=path.parent) as directory:
        onnx = f"{directory}/model.onnx"
        example = torch.randn(1, 3, *size, device="cuda", dtype=torch.float16)
        with torch.inference_mode():
            torch.onnx.export(Output(model, output).eval(), (example,), onnx, input_names=["pixel_values"],
                              output_names=[output], opset_version=18, dynamo=True, external_data=True)
        del model, example
        torch.cuda.empty_cache()
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(onnx):
            raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        # [Agent] HF accumulates normalization and softmax in float32. Weakly typed FP16 builds may otherwise
        # [Agent] downcast those reductions and visibly move outputs even though inference stays finite.
        for index in range(network.num_layers):
            layer = network.get_layer(index)
            if layer.type in {trt.LayerType.CONSTANT, trt.LayerType.SHAPE, trt.LayerType.CAST}:
                continue
            float_outputs = [i for i in range(layer.num_outputs) if layer.get_output(i).dtype == trt.float32]
            if float_outputs or layer.type in {trt.LayerType.NORMALIZATION, trt.LayerType.SOFTMAX}:
                layer.precision = trt.float32
                for i in float_outputs:
                    layer.set_output_type(i, trt.float32)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 2**30)
        config.builder_optimization_level = 3
        engine = builder.build_serialized_network(network, config)
        if engine is None:
            raise RuntimeError("TensorRT engine build failed")
    path.write_bytes(engine)
    print(f"Engine built in {time.perf_counter() - started:.0f}s: {path}", flush=True)

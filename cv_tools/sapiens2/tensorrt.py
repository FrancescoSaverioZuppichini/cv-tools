import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForPoseEstimation
from transformers.models.sapiens2.modeling_sapiens2 import Sapiens2PoseEstimatorOutput

from cv_tools.common import ROOT, resolution, resolve_device


class TensorRTPose:
    def __init__(self, engine):
        import tensorrt as trt

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Cannot deserialize TensorRT engine: {engine}")
        self.context = self.engine.create_execution_context()
        self.input_shape = tuple(self.engine.get_tensor_shape("pixel_values"))
        self.input_dtype = torch.from_numpy(np.empty((), dtype=trt.nptype(self.engine.get_tensor_dtype("pixel_values")))).dtype
        output_dtype = torch.from_numpy(np.empty((), dtype=trt.nptype(self.engine.get_tensor_dtype("heatmaps")))).dtype
        self.output = torch.empty(tuple(self.engine.get_tensor_shape("heatmaps")), device="cuda", dtype=output_dtype)
        self.context.set_tensor_address("heatmaps", self.output.data_ptr())

    def __call__(self, pixel_values):
        if tuple(pixel_values.shape) != self.input_shape or pixel_values.dtype != self.input_dtype:
            raise ValueError(f"Engine requires {self.input_shape} {self.input_dtype}; got {tuple(pixel_values.shape)} {pixel_values.dtype}")
        self.context.set_tensor_address("pixel_values", pixel_values.data_ptr())
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        # [Agent] This buffer is reused; callers must consume it before the next forward pass.
        return Sapiens2PoseEstimatorOutput(heatmaps=self.output)


class Heatmaps(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).heatmaps


def build_engine(model_path, size, engine_path, workspace_gib=4):
    import tensorrt as trt

    torch.set_num_threads(4)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path = engine_path.with_suffix(".onnx")
    started = time.perf_counter()
    if not onnx_path.exists():
        model = AutoModelForPoseEstimation.from_pretrained(
            model_path, local_files_only=True, dtype=torch.float16, attn_implementation="sdpa"
        ).cuda().eval()
        example = torch.randn(1, 3, *size, device="cuda", dtype=torch.float16)
        with torch.inference_mode():
            torch.onnx.export(Heatmaps(model).eval(), (example,), str(onnx_path),
                              input_names=["pixel_values"], output_names=["heatmaps"],
                              opset_version=18, dynamo=True, external_data=True)
        del model, example
        torch.cuda.empty_cache()
    print(json.dumps({"export_seconds": time.perf_counter() - started}), flush=True)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    # [Agent] HF accumulates RMS normalization in float32. Weakly typed FP16 builds may otherwise
    # [Agent] downcast those reductions and visibly move landmarks even though inference stays finite.
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        if layer.type in {trt.LayerType.CONSTANT, trt.LayerType.SHAPE, trt.LayerType.CAST}:
            continue
        float_outputs = [i for i in range(layer.num_outputs) if layer.get_output(i).dtype == trt.float32]
        if float_outputs or layer.type in {trt.LayerType.NORMALIZATION, trt.LayerType.SOFTMAX}:
            layer.precision = trt.float32
            for i in float_outputs:
                layer.set_output_type(i, trt.float32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gib * 2**30)
    config.builder_optimization_level = 3
    started = time.perf_counter()
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT engine build failed")
    engine_path.write_bytes(engine)
    print(json.dumps({"build_seconds": time.perf_counter() - started,
                      "engine": str(engine_path), "tensorrt": trt.__version__}), flush=True)


def build():
    p = argparse.ArgumentParser(description="Export a static Sapiens2 FP16 TensorRT engine on the target GPU")
    p.add_argument("--model", type=Path, default=ROOT / ".models" / "sapiens2-pose-0.4b")
    p.add_argument("--resolution", type=resolution, default=(1024, 768))
    p.add_argument("--engine", type=Path)
    p.add_argument("--workspace-gib", type=int, default=4)
    args = p.parse_args()
    args.device = "cuda"
    resolve_device(args, p)
    engine = args.engine or ROOT / "engines" / f"sapiens2-fp16-b1-{args.resolution[0]}x{args.resolution[1]}.plan"
    build_engine(args.model, args.resolution, engine, args.workspace_gib)


if __name__ == "__main__":
    build()

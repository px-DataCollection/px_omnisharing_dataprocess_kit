import tensorrt as trt

def build_engine(onnx_path, engine_path, shapes):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # 加载 ONNX 模型
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # 配置构建器
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB
    config.flags = 1 << int(trt.BuilderFlag.FP16)  # 启用 FP16
    
    # 设置输入形状
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        if input_tensor.name in shapes:
            input_tensor.shape = shapes[input_tensor.name]
            print(f"Set shape for {input_tensor.name}: {input_tensor.shape}")
    
    # 构建引擎
    serialized_engine = builder.build_serialized_network(network, config)
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    print(f"Engine saved to {engine_path}")
    
    return serialized_engine

# 使用示例
# 输入尺寸为32的倍数，720x1280需要填充到736x1280
build_engine(
    onnx_path="iinet.onnx",
    engine_path="checkpoints/iinet.engine",
    shapes={
        'left': (1, 3, 736, 1280),
        'right': (1, 3, 736, 1280)
    }
)

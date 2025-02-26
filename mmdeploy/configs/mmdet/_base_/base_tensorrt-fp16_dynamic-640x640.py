_base_ = ['./base_dynamic.py', '../../_base_/backends/tensorrt-fp16.py']

backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 640, 640],
                    opt_shape=[4, 3, 640, 640],
                    max_shape=[8, 3, 640, 640])))
    ])

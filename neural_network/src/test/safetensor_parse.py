from safetensors import safe_open

with safe_open("../first_model.safetensors", framework="pytorch") as f:
    # print(f.get_tensor_names())
    print(f.keys())
    '''
    conv是卷积层
    bn是批归一化层
    fc是全连接层
    '''
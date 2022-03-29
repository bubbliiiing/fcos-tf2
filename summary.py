#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
from nets.fcos import FCOS

if __name__ == "__main__":
    input_shape     = [600, 600, 3]
    
    model           = FCOS(input_shape, 80, backbone="resnet50")
    model.summary()

    # for i, layer in enumerate(model.layers):
    #     print(i, layer.name)
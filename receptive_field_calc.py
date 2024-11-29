import math


class ReceptiveFieldCalculator():
    def calculate(self, architecture, input_image_size):
        input_layer = ('input_layer', input_image_size, 1, 1, 0.5)
        self._print_layer_info(input_layer)

        for key in architecture:
            current_layer = self._calculate_layer_info(architecture[key],
                                                       input_layer, key)
            self._print_layer_info(current_layer)
            input_layer = current_layer

    def _print_layer_info(self, layer):
        print(f'------')
        print(
            f'{layer[0]}: n = {layer[1]}; j = {layer[2]}; r = {layer[3]}; '
            f'start = {layer[4]}')
        print(f'------')

    def _calculate_layer_info(self, current_layer, input_layer, layer_name):
        n_in = input_layer[1]
        j_in = input_layer[2]
        r_in = input_layer[3]
        start_in = input_layer[4]

        k = current_layer[0]
        s = current_layer[1]
        p = current_layer[2]

        n_out = math.floor((n_in - k + 2 * p) / s) + 1
        padding = (n_out - 1) * s - n_in + k
        p_right = math.ceil(padding / 2)
        p_left = math.floor(padding / 2)

        j_out = j_in * s
        r_out = r_in + (k - 1) * j_in
        start_out = start_in + ((k - 1) / 2 - p_left) * j_in
        return layer_name, n_out, j_out, r_out, start_out

alex_net = {
    'conv1': [11, 4, 0],
    'pool1': [3, 2, 0],
    'conv2': [5, 1, 2],
    'pool2': [3, 2, 0],
    'conv3': [3, 1, 1],
    'conv4': [3, 1, 1],
    'conv5': [3, 1, 1],
    'pool5': [3, 2, 0],
    'fc6-conv': [6, 1, 0],
    'fc7-conv': [1, 1, 0]
}

obj_det_net = {
    'conv1': [5, 1, 2],
    'pool1': [2, 2, 0],
    'conv2': [7, 1, 3],
    'pool2': [2, 2, 0],
    'conv3': [5, 1, 2],
    'pool3': [2, 2, 0],
    'conv4': [7, 1, 3],
    'pool4': [2, 2, 0],
    'conv5': [5, 1, 2],
    'pool5': [2, 2, 0],
    'conv6': [7, 1, 3],
    'pool6': [2, 2, 0],
    'conv7': [5, 1, 2],
    'pool7': [2, 2, 0],
}
calculator = ReceptiveFieldCalculator()
calculator.calculate(obj_det_net, 320)
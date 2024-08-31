import os
from typing import Dict, List, Tuple

import joblib
import timeloopfe.v4 as tl
import torch
from cimloop.workspace import get_run_dir, results2ppa
from torch import Tensor, nn

THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOP_PATH = f"{THIS_SCRIPT_DIR}/top.yaml.jinja"


def update_spec_problem(dims: List[Tuple[str, int]], spec: tl.Specification):
    COEFFECIENT_DIMS = ["Wdilation", "Hdilation", "Wstride", "Hstride"]
    coefficients: Dict[str, int] = {}

    for dim, val in dims:
        spec.problem.instance[dim] = val
        if dim in COEFFECIENT_DIMS:
            coefficients[dim] = val

    for coeff in spec.problem.shape.coefficients:
        coeff_name = coeff["name"]
        if coeff_name in coefficients:
            coeff["default"] = coefficients[coeff_name]


def update_spec_problem_instance_linear(
    layer: nn.Linear, input: Tensor, output: Tensor
) -> List[Tuple[str, int]]:
    return [
        ("C", layer.in_features),
        ("M", layer.out_features),
    ]


def conv2d_dims(
    layer: nn.Conv2d, input: Tensor, output: Tensor
) -> List[Tuple[str, int]]:

    return [
        ("C", layer.in_channels),
        ("H", input[0].shape[2]),
        ("W", input[0].shape[3]),
        ("R", layer.kernel_size[0]),
        ("S", layer.kernel_size[1]),
        ("Hdilation", layer.dilation[0]),
        ("Wdilation", layer.dilation[1]),
        ("Hstride", layer.stride[0]),
        ("Wstride", layer.stride[1]),
        ("M", layer.out_channels),
        ("P", output.shape[2]),
        ("Q", output.shape[3]),
    ]


def get_layer_data(model: nn.Module, input_tensor: Tensor):
    """
    Given a model and an input tensor, this function will return a dictionary that describes the problem in timeloop
    """
    layer_data: List[List[Tuple[str, int]]] = []

    def hook_fn(module: nn.Module, input: Tensor, output: Tensor):
        """
        Given a layer and an input tensor, this function will return a dictionary that describes the problem in timeloop
        """
        if isinstance(module, nn.Conv2d):
            layer_data.append(conv2d_dims(module, input, output))
        elif isinstance(module, nn.Linear):
            layer_data.append(
                update_spec_problem_instance_linear(module, input, output)
            )
        else:
            return

    hooks = []

    def register_hooks(module: nn.Module):
        if len(list(module.children())) == 0:  # It's a bottom-level layer
            hooks.append(module.register_forward_hook(hook_fn))  # type: ignore
        else:
            for submodule in module.children():
                register_hooks(submodule)

    register_hooks(model)

    with torch.no_grad():
        model(input_tensor)

    for hook in hooks:
        hook.remove()

    return layer_data


def run_layer(
    layer_data: List[Tuple[str, int]], x_dim: int, y_dim: int, frequency: int
):
    spec = tl.Specification.from_yaml_files(TOP_PATH)
    update_spec_problem(layer_data, spec)
    spec.architecture.find("PE").spatial.meshX = x_dim
    spec.architecture.find("PE").spatial.meshY = y_dim
    spec.variables.global_cycle_seconds = 1 / frequency
    spec.mapper.diagnostics = True

    output_dir = get_run_dir()
    run_prefix = f"{output_dir}/timeloop-mapper"
    result = tl.call_mapper(
        specification=spec,
        output_dir=output_dir,
        log_to=os.path.join(output_dir, f"{run_prefix}.log"),
    )

    return result


def timeloop_ppa(
    model: nn.Module,
    x_test: Tensor,
    x_dim_list: List[int],
    y_dim_list: List[int],
    frequency: int,
    cell_bit: int,
):
    """
    remember to add batch 1 for x_test
    """
    layer_data = get_layer_data(model, x_test)
    assert (
        len(layer_data) == len(x_dim_list) == len(y_dim_list)
    ), f"len(layer_data)={len(layer_data)}, len(x_dim_list)={len(x_dim_list)}, len(y_dim_list)={len(y_dim_list)}"

    results = joblib.Parallel(n_jobs=32)(
        joblib.delayed(run_layer)(layer, x_dim, y_dim, frequency)
        for layer, x_dim, y_dim in zip(layer_data, x_dim_list, y_dim_list)
    )

    # DEBUG
    # results = []
    # for layer, x_dim, y_dim in zip(layer_data, x_dim_list, y_dim_list):
    #     result = run_layer(layer, x_dim, y_dim, frequency)
    #     results.append(result)

    return results2ppa(results)

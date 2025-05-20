import os
from typing import Dict, List, Tuple

import joblib
import timeloopfe.v4 as tl
import torch
from cimloop.workspace import get_run_dir, results2ppa
from torch import Tensor, nn

THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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


def update_spec_arch(
    x_dim: int, y_dim: int, spec: tl.Specification
):
    spec.architecture.find("PE").spatial.meshX = x_dim
    spec.architecture.find("PE").spatial.meshY = y_dim


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
    TOP_PATH = f"{THIS_SCRIPT_DIR}/top.yaml.jinja"
    spec = tl.Specification.from_yaml_files(TOP_PATH)
    update_spec_problem(layer_data, spec)
    update_spec_arch(x_dim, y_dim, spec)
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


def timeloop_ppa_hdnn(
    model: nn.Module,
    x_test: Tensor,
    cnn_x_dim_1: int,
    cnn_y_dim_1: int,
    cnn_x_dim_2: int,
    cnn_y_dim_2: int,
    encoder_x_dim: int,
    encoder_y_dim: int,
    frequency: int,
):
    """
    remember to add batch 1 for x_test
    """
    layer_data = get_layer_data(model, x_test)

    # xy dim
    # x_dim_list = [cnn_x_dim_1, cnn_x_dim_2, encoder_layer_data[1][1]]
    # y_dim_list = [cnn_y_dim_1, cnn_y_dim_2, encoder_layer_data[0][1]]
    x_dim_list = [cnn_x_dim_1, cnn_x_dim_2, encoder_x_dim]
    y_dim_list = [cnn_y_dim_1, cnn_y_dim_2, encoder_y_dim]

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

def timeloop_ppa_layers(
    model: nn.Module,
    x_test: Tensor,
    x_dim_list: List[int],
    y_dim_list: List[int],
    frequency: int,
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

def run_layer_eyeriss(
    layer_data: List[Tuple[str, int]],
    mesh_x: int,
    mesh_y: int,
    glb_depth: int,
    glb_width: int,
    glb_n_banks: int,
    glb_read_bw: int,
    glb_write_bw: int,
    rf_depth: int,
    psum_rf_depth: int,
    rf_width: int,
    rf_read_bw: int,
    rf_write_bw: int,
    mac_mult_width: int,
    mac_adder_width: int,
    frequency: int,
):
    """
    Run Timeloop analysis for a single layer using Eyeriss-like architecture parameters.
    """
    TOP_PATH = f"{THIS_SCRIPT_DIR}/eyeriss.yaml.jinja2"
    spec = tl.Specification.from_yaml_files(TOP_PATH)
    
    # Update problem dimensions
    update_spec_problem(layer_data, spec)
    
    # Update architecture parameters
    # Update PE array dimensions
    spec.architecture.find("PE_column").spatial.meshX = mesh_x
    spec.architecture.find("PE").spatial.meshY = mesh_y
    
    # Update global buffer parameters
    glb = spec.architecture.find("shared_glb")
    glb.attributes.depth = glb_depth
    glb.attributes.width = glb_width
    glb.attributes.n_banks = glb_n_banks
    glb.attributes.read_bandwidth = glb_read_bw
    glb.attributes.write_bandwidth = glb_write_bw
    
    # Update RF parameters
    ifmap_spad = spec.architecture.find("ifmap_spad")
    weights_spad = spec.architecture.find("weights_spad")
    psum_spad = spec.architecture.find("psum_spad")
    
    # Update ifmap and weights RF
    for spad in [ifmap_spad, weights_spad]:
        spad.attributes.depth = rf_depth
        spad.attributes.width = rf_width
        spad.attributes.read_bandwidth = rf_read_bw
        spad.attributes.write_bandwidth = rf_write_bw
    
    # Update psum RF
    psum_spad.attributes.depth = psum_rf_depth
    psum_spad.attributes.width = rf_width
    psum_spad.attributes.read_bandwidth = rf_read_bw
    psum_spad.attributes.write_bandwidth = rf_write_bw
    
    # Update MAC parameters
    mac = spec.architecture.find("mac")
    mac.attributes.multiplier_width = mac_mult_width
    mac.attributes.adder_width = mac_adder_width
    
    # Update technology and frequency
    spec.variables.global_cycle_seconds = 1 / frequency
    
    # Enable mapper diagnostics
    spec.mapper.diagnostics = True
    
    # Run mapper
    output_dir = get_run_dir()
    run_prefix = f"{output_dir}/timeloop-mapper"
    result = tl.call_mapper(
        specification=spec,
        output_dir=output_dir,
        log_to=os.path.join(output_dir, f"{run_prefix}.log"),
    )
    
    return result

def timeloop_ppa_eyeriss(
    model: nn.Module,
    x_test: Tensor,
    mesh_x,
    mesh_y,
    glb_depth,
    glb_width,
    glb_n_banks,
    glb_read_bw,
    glb_write_bw,
    rf_depth,
    psum_rf_depth,
    rf_width,
    rf_read_bw,
    rf_write_bw,
    mac_mult_width,
    mac_adder_width,
    frequency,
):
    """
    Run Timeloop PPA analysis for an Eyeriss-like architecture with customizable parameters.
    
    Args:
        model: PyTorch model to analyze
        x_test: Input tensor for the model
        mesh_x: Number of PE columns (default: 14)
        mesh_y: Number of PEs per column (default: 12)
        glb_depth: Global buffer depth in words (default: 16384)
        glb_width: Global buffer width in words (default: 64)
        glb_n_banks: Number of global buffer banks (default: 32)
        glb_read_bw: Global buffer read bandwidth (default: 16)
        glb_write_bw: Global buffer write bandwidth (default: 16)
        rf_depth: RF depth for ifmap and weights (default: 12)
        psum_rf_depth: RF depth for partial sums (default: 16)
        rf_width: RF width in words (default: 16)
        rf_read_bw: RF read bandwidth (default: 2)
        rf_write_bw: RF write bandwidth (default: 2)
        mac_mult_width: MAC multiplier width in bits (default: 8)
        mac_adder_width: MAC adder width in bits (default: 16)
        technology: Technology node (default: "32nm")
        frequency: Operating frequency in MHz (default: 1000)
    
    Returns:
        PPA metrics (power, performance, area)
    """
    # Get layer data from the model
    layer_data = get_layer_data(model, x_test)
    
    # Run Timeloop analysis for each layer using the Eyeriss-specific parameters
    results = joblib.Parallel(n_jobs=32)(
        joblib.delayed(run_layer_eyeriss)(
            layer,
            mesh_x,
            mesh_y,
            glb_depth,
            glb_width,
            glb_n_banks,
            glb_read_bw,
            glb_write_bw,
            rf_depth,
            psum_rf_depth,
            rf_width,
            rf_read_bw,
            rf_write_bw,
            mac_mult_width,
            mac_adder_width,
            frequency,
        )
        for layer in layer_data
    )
    
    # DEBUG
    # results = []
    # for layer in layer_data:
    #     result = run_layer_eyeriss(
    #         layer,
    #         mesh_x,
    #         mesh_y,
    #         glb_depth,
    #         glb_width,
    #         glb_n_banks,
    #         glb_read_bw,
    #         glb_write_bw,
    #         rf_depth,
    #         psum_rf_depth,
    #         rf_width,
    #         rf_read_bw,
    #         rf_write_bw,
    #         mac_mult_width,
    #         mac_adder_width,
    #         frequency,
    #     )
    #     results.append(result)

    # Convert results to PPA metrics
    return results2ppa(results)
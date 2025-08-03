#!/usr/bin/env python3
"""
Convert LayerNormalization operations to TensorRT-compatible operations
"""

import onnx
import numpy as np
from onnx import helper, numpy_helper
import sys

def convert_layernorm_to_tensorrt_compatible(model_path, output_path):
    """Convert LayerNormalization to TensorRT-compatible operations"""
    try:
        model = onnx.load(model_path)
        print(f"Loading model: {model_path}")
        
        # Find all LayerNormalization nodes
        layernorm_nodes = [node for node in model.graph.node if node.op_type == "LayerNormalization"]
        print(f"Found {len(layernorm_nodes)} LayerNormalization nodes")
        
        if len(layernorm_nodes) == 0:
            print("No LayerNormalization nodes found. Model is already compatible.")
            return True
        
        # Create a new model with converted operations
        new_nodes = []
        new_initializers = []
        new_inputs = []
        new_outputs = []
        
        # Copy existing inputs and outputs
        for input_info in model.graph.input:
            new_inputs.append(input_info)
        for output_info in model.graph.output:
            new_outputs.append(output_info)
        
        # Process each node
        for node in model.graph.node:
            if node.op_type == "LayerNormalization":
                print(f"Converting LayerNormalization node: {node.name}")
                
                # Get the inputs
                input_name = node.input[0]
                weight_name = node.input[1]
                bias_name = node.input[2]
                output_name = node.output[0]
                
                # Get epsilon from attributes
                epsilon = 1e-6
                for attr in node.attribute:
                    if attr.name == "epsilon":
                        epsilon = attr.f
                
                # Create intermediate names
                mean_name = f"{node.name}_mean"
                var_name = f"{node.name}_var"
                std_name = f"{node.name}_std"
                normalized_name = f"{node.name}_normalized"
                
                # Add Mean operation
                mean_node = helper.make_node(
                    "ReduceMean",
                    inputs=[input_name],
                    outputs=[mean_name],
                    name=f"{node.name}_mean",
                    axes=[-1],
                    keepdims=1
                )
                new_nodes.append(mean_node)
                
                # Add Sub operation (x - mean)
                sub_node = helper.make_node(
                    "Sub",
                    inputs=[input_name, mean_name],
                    outputs=[f"{node.name}_centered"],
                    name=f"{node.name}_sub"
                )
                new_nodes.append(sub_node)
                
                # Add Pow operation ((x - mean)^2)
                pow_node = helper.make_node(
                    "Pow",
                    inputs=[f"{node.name}_centered"],
                    outputs=[f"{node.name}_squared"],
                    name=f"{node.name}_pow",
                    exponent=2.0
                )
                new_nodes.append(pow_node)
                
                # Add ReduceMean for variance
                var_node = helper.make_node(
                    "ReduceMean",
                    inputs=[f"{node.name}_squared"],
                    outputs=[var_name],
                    name=f"{node.name}_var",
                    axes=[-1],
                    keepdims=1
                )
                new_nodes.append(var_node)
                
                # Add Add operation (var + epsilon)
                add_eps_node = helper.make_node(
                    "Add",
                    inputs=[var_name],
                    outputs=[f"{node.name}_var_eps"],
                    name=f"{node.name}_add_eps"
                )
                # Add epsilon as initializer
                eps_initializer = numpy_helper.from_array(
                    np.array([epsilon], dtype=np.float32),
                    f"{node.name}_epsilon"
                )
                new_initializers.append(eps_initializer)
                add_eps_node.input.append(f"{node.name}_epsilon")
                new_nodes.append(add_eps_node)
                
                # Add Sqrt operation
                sqrt_node = helper.make_node(
                    "Sqrt",
                    inputs=[f"{node.name}_var_eps"],
                    outputs=[std_name],
                    name=f"{node.name}_sqrt"
                )
                new_nodes.append(sqrt_node)
                
                # Add Div operation ((x - mean) / std)
                div_node = helper.make_node(
                    "Div",
                    inputs=[f"{node.name}_centered", std_name],
                    outputs=[normalized_name],
                    name=f"{node.name}_div"
                )
                new_nodes.append(div_node)
                
                # Add Mul operation (normalized * weight)
                mul_node = helper.make_node(
                    "Mul",
                    inputs=[normalized_name, weight_name],
                    outputs=[f"{node.name}_weighted"],
                    name=f"{node.name}_mul"
                )
                new_nodes.append(mul_node)
                
                # Add Add operation (weighted + bias)
                add_bias_node = helper.make_node(
                    "Add",
                    inputs=[f"{node.name}_weighted", bias_name],
                    outputs=[output_name],
                    name=f"{node.name}_add_bias"
                )
                new_nodes.append(add_bias_node)
                
            else:
                # Keep other nodes as-is
                new_nodes.append(node)
        
        # Copy existing initializers
        for initializer in model.graph.initializer:
            new_initializers.append(initializer)
        
        # Create new graph
        new_graph = helper.make_graph(
            new_nodes,
            model.graph.name,
            new_inputs,
            new_outputs,
            new_initializers
        )
        
        # Create new model
        new_model = helper.make_model(
            new_graph,
            producer_name="RF-DETR TensorRT Compatible",
            opset_imports=model.opset_import
        )
        
        # Save the converted model
        onnx.save(new_model, output_path)
        print(f"Converted model saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error converting model: {e}")
        return False

if __name__ == "__main__":
    input_path = "rf-detr-base.onnx"
    output_path = "rf-detr-base-tensorrt.onnx"
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    success = convert_layernorm_to_tensorrt_compatible(input_path, output_path)
    if success:
        print("Conversion completed successfully!")
    else:
        print("Conversion failed!") 
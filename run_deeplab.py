import xir
import vart
import numpy as np
import time
import sys
import os


def get_child_subgraph_dpu(graph):
   
   
    assert graph is not None, "'graph' should not be None."

    root_subgraph = graph.get_root_subgraph()
    assert root_subgraph is not None, "Failed to get root subgraph."

    if root_subgraph.is_leaf:
        return []

    
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0, \
        "Failed to get child subgraphs."

    dpu_subgraphs = [
        cs for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]

    print(f"Found {len(dpu_subgraphs)} DPU subgraphs.")
    for i, cs in enumerate(dpu_subgraphs):
        print(f"  [{i}] name={cs.get_name()}, device={cs.get_attr('device')}")

    return dpu_subgraphs


def benchmark_xmodel(xmodel_path, num_iters=200):
    print(f"Loading graph from: {xmodel_path}")
    graph = xir.Graph.deserialize(xmodel_path)

    subgraphs = get_child_subgraph_dpu(graph)
    if not subgraphs:
        raise RuntimeError("No DPU subgraphs found in this xmodel.")

   
    dpu_subgraph = subgraphs[0]
    print("Using DPU subgraph:", dpu_subgraph.get_name())

    print("Creating VART runner...")
    runner = vart.Runner.create_runner(dpu_subgraph, "run")

    input_tensors = runner.get_input_tensors()
    output_tensors = runner.get_output_tensors()

    in_tensor = input_tensors[0]
    out_tensor = output_tensors[0]

    in_shape = in_tensor.dims
    out_shape = out_tensor.dims

    print("Input tensor shape:", in_shape)
    print("Output tensor shape:", out_shape)

    
    input_data = np.random.randint(
        low=-128, high=127, size=in_shape, dtype=np.int8
    )
    output_data = np.zeros(out_shape, dtype=np.int8)

    # Warmup
    print("Running warmup...")
    for _ in range(10):
        jid = runner.execute_async([input_data], [output_data])
        runner.wait(jid)

    # Timed run
    print(f"Running benchmark for {num_iters} iterations...")
    start = time.perf_counter()
    for _ in range(num_iters):
        jid = runner.execute_async([input_data], [output_data])
        runner.wait(jid)
    end = time.perf_counter()

    total_time = end - start
    latency_ms = (total_time / num_iters) * 1000.0
    throughput = num_iters / total_time

    print("\n=== V70 DPU Benchmark ===")
    print(f"Latency   : {latency_ms:.3f} ms / image")
    print(f"Throughput: {throughput:.2f} images/s")

    return latency_ms, throughput


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run_deeplab.py <path_to_xmodel>")
        sys.exit(1)

    xmodel_path = sys.argv[1]
    if not os.path.exists(xmodel_path):
        print(f"ERROR: xmodel file not found: {xmodel_path}")
        sys.exit(1)

    benchmark_xmodel(xmodel_path)


if __name__ == "__main__":
    main()


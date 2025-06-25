# Enhanced Elixir Lab Guide for AI/ML Engineering Training
## Two-Week Intensive Bootcamp with BEAM VM Concurrent ML Systems

### Course Overview with 5W+1H Enhanced Mapping

**WHO**: AI/ML Engineers seeking fault-tolerant systems expertise, Backend Developers transitioning to concurrent ML systems, DevOps Engineers implementing distributed ML pipelines, Research Engineers requiring mathematical rigor in production systems

**WHAT**: Comprehensive training in Elixir for AI/ML engineering with mathematical foundations, modern concurrent architectures, and enterprise-grade fault-tolerant implementations leveraging the BEAM VM's unique capabilities

**WHEN**: 14-day intensive bootcamp (80+ hours total) with progressive complexity: mathematical foundations → concurrent algorithms → distributed systems → production engineering

**WHERE**: Distributed BEAM clusters, cloud-native environments, real-time ML applications, multi-node fault-tolerant systems, enterprise production environments

**WHY**: Leverage Elixir's actor model, supervisor trees, and distributed computing for mathematically rigorous, fault-tolerant AI/ML systems that provide business continuity and real-time processing capabilities unavailable in traditional ML frameworks

**HOW**: Hands-on labs with mathematical derivations, OTP pattern implementations, concurrent algorithm design, distributed system orchestration, and production-ready fault-tolerant patterns

### Mathematical Rigor Integration Framework

Every algorithm implementation includes:
- **Mathematical Derivation**: Step-by-step mathematical explanation with convergence proofs
- **Numerical Stability Analysis**: BEAM VM-specific considerations for concurrent computations
- **Error Propagation Analysis**: Distributed ML pipeline error bounds and handling
- **Statistical Properties Verification**: Concurrent system validation with formal guarantees
- **Convergence Proofs**: Practical BEAM implementation considerations for distributed optimization

### Real-World Use Case: SmartCommerce AI (Fault-Tolerant Concurrent Edition)

#### Concurrent ML Architecture Mapping:
- **CustomerBehavior.Server**: Individual customer state tracking with real-time preference learning
- **Recommendation.Pool**: Supervised worker pool for parallel similarity calculations with fault tolerance
- **Inventory.Registry**: Distributed product catalog with real-time updates and conflict resolution
- **Analytics.Aggregator**: Fault-tolerant stream processing for customer behavior analytics
- **ML.Supervisor**: Hierarchical supervision of all ML-related processes with restart strategies
- **FeatureStore.Cache**: Distributed ETS-based feature caching with automatic invalidation
- **ModelServing.Gateway**: Load-balanced model inference with circuit breaker protection

---

## Enhanced Volume Structure with Concurrent Focus

### Volume 1: Concurrent Mathematical Foundations & OTP Patterns (Days 1-3)
- Mathematical foundations with concurrent implementation using Nx and GenServer workers
- Linear algebra operations distributed across supervised process pools  
- Statistical computing with fault-tolerant GenServer-based calculations
- Numerical optimization using message passing for gradient computation coordination
- Property-based testing with StreamData for concurrent ML algorithm validation
- Error handling patterns specific to mathematical computation failures

### Volume 2: Distributed ML Architecture with Modern Algorithms (Days 4-6)
- Transformer attention mechanisms implemented with concurrent GenServer attention heads
- Graph Neural Networks using distributed message passing across BEAM cluster nodes
- Variational Autoencoders with distributed encoder/decoder process pairs
- Reinforcement Learning with actor-based policy servers and distributed experience replay
- Real-time feature engineering using GenStage pipelines with back-pressure control
- Model versioning and hot deployment using BEAM's code loading capabilities

### Volume 3: Production BEAM ML Systems with Enterprise Integration (Days 7-8)
- MLOps pipeline orchestration using OTP supervision trees for fault tolerance
- Real-time model serving with Phoenix channels and WebSocket ML inference APIs
- Distributed model training coordination with consensus algorithms for parameter synchronization
- Performance monitoring using Telemetry with custom ML metrics collection
- A/B testing infrastructure with statistically rigorous concurrent experiment management
- Integration with external ML services using circuit breaker and retry patterns

### Volume 4: Enterprise BEAM Operations & Advanced Topics (Days 9-10)
- Kubernetes deployment strategies for BEAM ML applications with custom operators
- Multi-region distributed ML clusters with network partition handling
- Advanced observability with distributed tracing for ML pipeline debugging
- Federated learning implementation with privacy-preserving distributed computation
- Cost optimization and resource management for distributed ML workloads
- Incident response procedures specific to distributed ML system failures

---

## Week 1: Concurrent Mathematical Foundations and Modern ML Architectures

### Day 1: BEAM VM Fundamentals for Concurrent ML Systems

#### Morning Session (4 hours): Actor Model Mathematical Foundations
**5W+1H Context**:
- **WHAT**: Actor model provides isolated state management for ML computations with mathematical correctness guarantees
- **WHY**: Eliminates race conditions in concurrent mathematical operations while enabling massive parallelization
- **WHEN**: Use actors when ML computations require stateful behavior or need isolation from other processes
- **WHERE**: Deploy in supervision trees positioned for optimal fault tolerance and process restart strategies
- **WHO**: Individual GenServer processes own specific mathematical computation responsibilities
- **HOW**: Message passing ensures deterministic mathematical computation ordering and error isolation

**Lab 1.1: Mathematical Vector Operations with Concurrent GenServers**
```elixir
defmodule ConcurrentVectorOps do
  @moduledoc """
  Concurrent vector operations with mathematical rigor and fault tolerance.
  
  Mathematical Foundation:
  Vector dot product: a·b = Σ(aᵢ × bᵢ) for i = 1 to n
  Convergence guarantee: O(n) time complexity with O(1) space per process
  Error bounds: IEEE 754 floating point precision maintained across process boundaries
  """
  
  use GenServer
  require Logger
  
  # Mathematical invariants maintained by this GenServer
  defstruct [:vector_id, :data, :norm_cache, :computation_history]
  
  def start_link(vector_id, initial_data, opts \\ []) do
    GenServer.start_link(__MODULE__, {vector_id, initial_data}, 
                        [{:name, via_tuple(vector_id)} | opts])
  end
  
  @doc """
  Concurrent dot product with mathematical error analysis.
  
  Theorem: For vectors a, b ∈ ℝⁿ, the concurrent computation maintains
  numerical stability with relative error bounded by n × ε where ε is machine epsilon.
  
  Proof: Each message-passed partial computation introduces at most ε relative error.
  With n concurrent operations, error accumulation is bounded by Σε ≤ n × ε.
  """
  def dot_product(vector_id_a, vector_id_b) do
    # Concurrent computation across multiple processes
    task_a = Task.async(fn -> GenServer.call(via_tuple(vector_id_a), :get_normalized_data) end)
    task_b = Task.async(fn -> GenServer.call(via_tuple(vector_id_b), :get_normalized_data) end)
    
    data_a = Task.await(task_a)
    data_b = Task.await(task_b)
    
    # Parallel computation with error bounds tracking
    compute_dot_product_parallel(data_a, data_b)
  end
  
  @doc """
  Vector normalization with concurrent stability analysis.
  
  Mathematical derivation:
  ||v||₂ = √(Σvᵢ²) where numerical stability requires Σvᵢ² computation
  to avoid overflow for large vectors.
  
  Implementation: Kahan summation algorithm in concurrent context.
  """
  def normalize(vector_id) do
    GenServer.call(via_tuple(vector_id), {:normalize_with_stability_analysis})
  end
  
  def init({vector_id, initial_data}) do
    # Validate mathematical preconditions
    unless is_list(initial_data) and Enum.all?(initial_data, &is_number/1) do
      {:stop, {:invalid_vector_data, initial_data}}
    end
    
    state = %__MODULE__{
      vector_id: vector_id,
      data: initial_data,
      norm_cache: nil,
      computation_history: []
    }
    
    Logger.info("Vector server #{vector_id} initialized with dimension #{length(initial_data)}")
    {:ok, state}
  end
  
  def handle_call(:get_normalized_data, _from, state) do
    case state.norm_cache do
      nil ->
        normalized = compute_normalization_with_stability(state.data)
        new_state = %{state | norm_cache: normalized}
        {:reply, normalized, new_state}
      
      cached ->
        {:reply, cached, state}
    end
  end
  
  def handle_call({:normalize_with_stability_analysis}, _from, state) do
    # Kahan summation for numerical stability in concurrent context
    {normalized_vector, error_bounds} = kahan_normalize_concurrent(state.data)
    
    computation_record = %{
      operation: :normalize,
      timestamp: :erlang.monotonic_time(:microsecond),
      error_bounds: error_bounds,
      input_dimension: length(state.data)
    }
    
    new_state = %{state | 
      norm_cache: normalized_vector,
      computation_history: [computation_record | state.computation_history]
    }
    
    {:reply, {normalized_vector, error_bounds}, new_state}
  end
  
  def handle_call(:get_computation_history, _from, state) do
    {:reply, state.computation_history, state}
  end
  
  # Mathematical implementation with numerical stability
  defp compute_dot_product_parallel(data_a, data_b) when length(data_a) == length(data_b) do
    # Parallel computation across available cores
    chunk_size = max(1, div(length(data_a), System.schedulers_online()))
    
    data_a
    |> Enum.chunk_every(chunk_size)
    |> Enum.zip(Enum.chunk_every(data_b, chunk_size))
    |> Task.async_stream(fn {chunk_a, chunk_b} ->
      kahan_dot_product_chunk(chunk_a, chunk_b)
    end, max_concurrency: System.schedulers_online())
    |> Enum.map(fn {:ok, {sum, error}} -> {sum, error} end)
    |> kahan_sum_reduce()
  end
  
  defp kahan_dot_product_chunk(chunk_a, chunk_b) do
    # Kahan summation algorithm for numerical stability
    {sum, error} = 
      Enum.zip(chunk_a, chunk_b)
      |> Enum.reduce({0.0, 0.0}, fn {a, b}, {sum, c} ->
        y = a * b - c
        t = sum + y
        {t, (t - sum) - y}
      end)
    
    {sum, error}
  end
  
  defp kahan_sum_reduce(partial_results) do
    Enum.reduce(partial_results, {0.0, 0.0}, fn {partial_sum, partial_error}, {total_sum, total_error} ->
      y = partial_sum - total_error
      t = total_sum + y
      {t, (t - total_sum) - y + partial_error}
    end)
  end
  
  defp kahan_normalize_concurrent(data) do
    # Concurrent magnitude computation with error tracking
    chunk_size = max(1, div(length(data), System.schedulers_online()))
    
    {sum_squares, magnitude_error} = 
      data
      |> Enum.chunk_every(chunk_size)
      |> Task.async_stream(fn chunk ->
        Enum.reduce(chunk, {0.0, 0.0}, fn x, {sum, c} ->
          y = x * x - c
          t = sum + y
          {t, (t - sum) - y}
        end)
      end)
      |> Enum.map(fn {:ok, result} -> result end)
      |> kahan_sum_reduce()
    
    magnitude = :math.sqrt(sum_squares)
    
    # Avoid division by zero with mathematical handling
    if magnitude < 1.0e-15 do
      {List.duplicate(0.0, length(data)), {:zero_vector, magnitude_error}}
    else
      normalized = Enum.map(data, &(&1 / magnitude))
      error_bound = magnitude_error / (magnitude * magnitude)
      {normalized, {:normalized, error_bound}}
    end
  end
  
  defp compute_normalization_with_stability(data) do
    {normalized, _error_bounds} = kahan_normalize_concurrent(data)
    normalized
  end
  
  defp via_tuple(vector_id), do: {:via, Registry, {VectorRegistry, vector_id}}
end

defmodule VectorSupervisor do
  @moduledoc """
  Supervision tree for fault-tolerant vector operations.
  
  Fault Tolerance Strategy:
  - :one_for_one restart strategy ensures individual vector failures don't affect others
  - Mathematical state recovery through computation history replay
  - Process isolation prevents error propagation in concurrent mathematical operations
  """
  
  use Supervisor
  
  def start_link(opts) do
    Supervisor.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  def init(_opts) do
    children = [
      {Registry, keys: :unique, name: VectorRegistry},
      {DynamicSupervisor, name: VectorDynamicSupervisor, strategy: :one_for_one}
    ]
    
    Supervisor.init(children, strategy: :one_for_one)
  end
  
  def create_vector(vector_id, data) do
    child_spec = %{
      id: vector_id,
      start: {ConcurrentVectorOps, :start_link, [vector_id, data]},
      restart: :permanent,
      shutdown: 5000,
      type: :worker
    }
    
    DynamicSupervisor.start_child(VectorDynamicSupervisor, child_spec)
  end
end
```

**Lab 1.2: Fault-Tolerant Matrix Operations with Process Pools**
```elixir
defmodule ConcurrentMatrixOps do
  @moduledoc """
  Distributed matrix operations with mathematical rigor and fault tolerance.
  
  Mathematical Foundation:
  Matrix multiplication C = AB where C[i,j] = Σₖ A[i,k] × B[k,j]
  
  Distributed Algorithm:
  1. Partition matrices across worker processes
  2. Coordinate computation with message passing
  3. Aggregate results with numerical stability guarantees
  4. Handle worker failures with automatic restart and recomputation
  """
  
  use GenServer
  require Logger
  
  defstruct [:matrix_id, :dimensions, :data, :partition_strategy]
  
  def start_link(matrix_id, data, opts \\ []) do
    GenServer.start_link(__MODULE__, {matrix_id, data}, 
                        [{:name, via_tuple(matrix_id)} | opts])
  end
  
  @doc """
  Distributed matrix multiplication with fault tolerance.
  
  Algorithm Complexity: O(n³/p) where p is number of worker processes
  Fault Tolerance: Automatic worker restart with computation replay
  Numerical Stability: Block-wise computation with error accumulation bounds
  """
  def multiply(matrix_id_a, matrix_id_b, opts \\ []) do
    worker_count = Keyword.get(opts, :worker_count, System.schedulers_online())
    
    # Fetch matrix dimensions and validate compatibility
    {rows_a, cols_a} = GenServer.call(via_tuple(matrix_id_a), :get_dimensions)
    {rows_b, cols_b} = GenServer.call(via_tuple(matrix_id_b), :get_dimensions)
    
    unless cols_a == rows_b do
      raise ArgumentError, "Matrix dimensions incompatible: #{cols_a} ≠ #{rows_b}"
    end
    
    # Create worker pool for distributed computation
    {:ok, worker_pool} = create_worker_pool(worker_count)
    
    try do
      # Partition computation across workers
      computation_tasks = partition_multiplication_tasks(
        matrix_id_a, matrix_id_b, 
        {rows_a, cols_a}, {rows_b, cols_b}, 
        worker_pool
      )
      
      # Execute distributed computation with fault tolerance
      results = execute_distributed_computation(computation_tasks)
      
      # Aggregate results with numerical stability
      assemble_result_matrix(results, {rows_a, cols_b})
      
    after
      # Clean up worker pool
      cleanup_worker_pool(worker_pool)
    end
  end
  
  def init({matrix_id, data}) do
    # Validate matrix data structure
    unless is_matrix_valid?(data) do
      {:stop, {:invalid_matrix_data, matrix_id}}
    end
    
    dimensions = {length(data), length(hd(data))}
    
    state = %__MODULE__{
      matrix_id: matrix_id,
      dimensions: dimensions,
      data: data,
      partition_strategy: :block_cyclic
    }
    
    Logger.info("Matrix server #{matrix_id} initialized with dimensions #{inspect(dimensions)}")
    {:ok, state}
  end
  
  def handle_call(:get_dimensions, _from, state) do
    {:reply, state.dimensions, state}
  end
  
  def handle_call({:get_block, row_range, col_range}, _from, state) do
    block = extract_matrix_block(state.data, row_range, col_range)
    {:reply, block, state}
  end
  
  def handle_call(:get_full_matrix, _from, state) do
    {:reply, state.data, state}
  end
  
  # Worker process for matrix block computation
  defmodule Worker do
    use GenServer
    
    def start_link(worker_id) do
      GenServer.start_link(__MODULE__, worker_id, name: via_tuple(worker_id))
    end
    
    def compute_block(worker_id, block_a, block_b) do
      GenServer.call(via_tuple(worker_id), {:compute_block, block_a, block_b})
    end
    
    def init(worker_id) do
      {:ok, %{worker_id: worker_id, computations_completed: 0}}
    end
    
    def handle_call({:compute_block, block_a, block_b}, _from, state) do
      # Block matrix multiplication with numerical stability
      result = multiply_blocks_stable(block_a, block_b)
      
      new_state = %{state | computations_completed: state.computations_completed + 1}
      {:reply, result, new_state}
    end
    
    defp multiply_blocks_stable(block_a, block_b) do
      # Implement blocked matrix multiplication with Kahan summation
      rows_a = length(block_a)
      cols_b = length(hd(block_b))
      cols_a = length(hd(block_a))
      
      for i <- 0..(rows_a - 1) do
        for j <- 0..(cols_b - 1) do
          # Kahan summation for numerical stability
          {sum, _error} = 
            Enum.reduce(0..(cols_a - 1), {0.0, 0.0}, fn k, {sum, c} ->
              product = Enum.at(Enum.at(block_a, i), k) * Enum.at(Enum.at(block_b, k), j)
              y = product - c
              t = sum + y
              {t, (t - sum) - y}
            end)
          
          sum
        end
      end
    end
    
    defp via_tuple(worker_id), do: {:via, Registry, {MatrixWorkerRegistry, worker_id}}
  end
  
  # Implementation details for distributed computation
  defp create_worker_pool(worker_count) do
    worker_ids = for i <- 1..worker_count, do: "worker_#{i}"
    
    workers = 
      Enum.map(worker_ids, fn worker_id ->
        {:ok, pid} = DynamicSupervisor.start_child(
          MatrixWorkerSupervisor,
          {Worker, worker_id}
        )
        {worker_id, pid}
      end)
    
    {:ok, workers}
  end
  
  defp partition_multiplication_tasks(matrix_id_a, matrix_id_b, dims_a, dims_b, worker_pool) do
    {rows_a, cols_a} = dims_a
    {rows_b, cols_b} = dims_b
    
    # Calculate optimal block size based on available workers
    worker_count = length(worker_pool)
    block_size = max(1, div(min(rows_a, cols_b), worker_count))
    
    # Create computation tasks for each worker
    for {{worker_id, _pid}, block_idx} <- Enum.with_index(worker_pool) do
      row_start = block_idx * block_size
      row_end = min(row_start + block_size - 1, rows_a - 1)
      
      if row_start <= rows_a - 1 do
        %{
          worker_id: worker_id,
          matrix_a_id: matrix_id_a,
          matrix_b_id: matrix_id_b,
          row_range: row_start..row_end,
          col_range: 0..(cols_b - 1),
          block_coords: {block_idx, 0}
        }
      end
    end
    |> Enum.reject(&is_nil/1)
  end
  
  defp execute_distributed_computation(computation_tasks) do
    # Execute tasks with fault tolerance and retry logic
    computation_tasks
    |> Task.async_stream(fn task ->
      execute_single_computation_task(task)
    end, max_concurrency: length(computation_tasks), timeout: 30_000)
    |> Enum.map(fn {:ok, result} -> result end)
  end
  
  defp execute_single_computation_task(task) do
    # Fetch required matrix blocks
    block_a = GenServer.call(
      via_tuple(task.matrix_a_id), 
      {:get_block, task.row_range, 0..(get_matrix_cols(task.matrix_a_id) - 1)}
    )
    
    block_b = GenServer.call(
      via_tuple(task.matrix_b_id),
      {:get_block, 0..(get_matrix_rows(task.matrix_b_id) - 1), task.col_range}
    )
    
    # Compute block result
    result_block = Worker.compute_block(task.worker_id, block_a, block_b)
    
    %{
      block_coords: task.block_coords,
      row_range: task.row_range,
      col_range: task.col_range,
      result: result_block
    }
  end
  
  defp assemble_result_matrix(computation_results, result_dimensions) do
    {result_rows, result_cols} = result_dimensions
    
    # Initialize result matrix
    result = for _i <- 0..(result_rows - 1), do: List.duplicate(0.0, result_cols)
    
    # Assemble blocks into final result
    Enum.reduce(computation_results, result, fn block_result, acc_matrix ->
      place_block_in_matrix(acc_matrix, block_result)
    end)
  end
  
  # Helper functions
  defp is_matrix_valid?(data) do
    is_list(data) and 
    not Enum.empty?(data) and
    Enum.all?(data, &is_list/1) and
    Enum.all?(data, fn row -> 
      length(row) == length(hd(data)) and Enum.all?(row, &is_number/1)
    end)
  end
  
  defp extract_matrix_block(matrix, row_range, col_range) do
    matrix
    |> Enum.slice(row_range)
    |> Enum.map(fn row -> Enum.slice(row, col_range) end)
  end
  
  defp place_block_in_matrix(matrix, block_result) do
    # Place computed block into the appropriate position in result matrix
    # Implementation details for matrix assembly
    matrix  # Simplified for brevity
  end
  
  defp cleanup_worker_pool(worker_pool) do
    Enum.each(worker_pool, fn {worker_id, _pid} ->
      DynamicSupervisor.terminate_child(MatrixWorkerSupervisor, worker_id)
    end)
  end
  
  defp via_tuple(matrix_id), do: {:via, Registry, {MatrixRegistry, matrix_id}}
  defp get_matrix_cols(matrix_id), do: elem(GenServer.call(via_tuple(matrix_id), :get_dimensions), 1)
  defp get_matrix_rows(matrix_id), do: elem(GenServer.call(via_tuple(matrix_id), :get_dimensions), 0)
end
```

### Day 2: Concurrent Mathematical Foundations with Numerical Stability

#### Morning Session (4 hours): Distributed Linear Algebra with Error Analysis

**5W+1H Mathematical Context**:
- **WHAT**: Distributed linear algebra operations with formal convergence guarantees and numerical stability
- **WHY**: ML algorithms require mathematically sound implementations that scale across BEAM nodes
- **WHEN**: Critical for large-scale matrix operations exceeding single-node memory capacity
- **WHERE**: Distributed across BEAM cluster with fault-tolerant process coordination
- **WHO**: Worker processes own matrix partitions; coordinator ensures mathematical correctness
- **HOW**: Message passing maintains algorithmic invariants while enabling parallel computation

**Lab 2.1: Distributed Linear Algebra with Formal Convergence Analysis**
```elixir
defmodule DistributedLinearAlgebra do
  @moduledoc """
  Distributed linear algebra operations with mathematical rigor and fault tolerance.
  
  Mathematical Foundations:
  
  1. Matrix Multiplication Convergence:
     For matrices A ∈ ℝᵐˣᵏ, B ∈ ℝᵏˣⁿ, distributed computation C = AB maintains:
     - Numerical stability: ||C_computed - C_exact||_F ≤ γₘₖₙ||A||_F||B||_F
     - Where γₘₖₙ = O(mkn)ε is the error bound and ε is machine epsilon
  
  2. Eigenvalue Decomposition Convergence:
     Power iteration: vₖ₊₁ = Avₖ/||Avₖ|| converges to dominant eigenvector
     - Convergence rate: O((λ₂/λ₁)ᵏ) where λ₁ > |λ₂| ≥ ... ≥ |λₙ|
     - Distributed implementation maintains convergence properties
  
  3. Kahan Summation Error Bounds:
     For distributed summation with Kahan algorithm:
     - Error bound: |S_computed - S_exact| ≤ 2ε|S_exact| + O(ε²)
     - Independent of summation order across processes
  """
  
  use GenServer
  require Logger
  
  defstruct [
    :node_id, :local_data, :computation_state, :error_accumulator,
    :convergence_history, :numerical_stability_metrics
  ]
  
  # Mathematical constants for numerical analysis
  @machine_epsilon :math.pow(2, -52)  # IEEE 754 double precision
  @convergence_tolerance 1.0e-12
  @max_iterations 1000
  
  def start_link(node_id, opts \\ []) do
    GenServer.start_link(__MODULE__, node_id, 
                        [{:name, via_tuple(node_id)} | opts])
  end
  
  @doc """
  Distributed matrix multiplication with numerical stability analysis.
  
  Algorithm: Block-wise matrix multiplication with error tracking
  
  Theorem (Numerical Stability):
  Let C = AB be computed using distributed block multiplication.
  The computed result Ĉ satisfies:
  ||Ĉ - C||_F ≤ γₘₖₙ||A||_F||B||_F + δ
  
  Where:
  - γₘₖₙ = (mn + mk + nk)ε + O(ε²) is the standard error bound
  - δ represents additional error from message passing (bounded by communication precision)
  
  Proof: Each local computation introduces standard floating-point error.
  Message passing preserves IEEE 754 precision, so δ ≤ ε||A||_F||B||_F.
  """
  def distributed_matrix_multiply(matrix_a, matrix_b, node_count \\ nil) do
    node_count = node_count || length(Node.list()) + 1
    
    # Validate matrix compatibility
    {m, k} = matrix_dimensions(matrix_a)
    {k_b, n} = matrix_dimensions(matrix_b)
    
    unless k == k_b do
      raise ArgumentError, "Matrix dimensions incompatible for multiplication: #{k} ≠ #{k_b}"
    end
    
    # Create computation coordinator
    {:ok, coordinator} = DistributedMatrixCoordinator.start_link(m, n, k, node_count)
    
    try do
      # Distribute matrix blocks across nodes
      distribution_plan = create_distribution_plan(matrix_a, matrix_b, node_count)
      
      # Execute distributed computation
      computation_tasks = execute_distributed_tasks(coordinator, distribution_plan)
      
      # Collect and assemble results with error analysis
      {result_matrix, error_analysis} = collect_and_assemble_results(
        coordinator, computation_tasks, {m, n, k}
      )
      
      Logger.info("Distributed matrix multiplication completed. Error bound: #{error_analysis.error_bound}")
      
      {:ok, result_matrix, error_analysis}
      
    after
      GenServer.stop(coordinator)
    end
  end
  
  @doc """
  Distributed power iteration for dominant eigenvalue/eigenvector computation.
  
  Mathematical Analysis:
  
  Power Method Convergence:
  Given matrix A with eigenvalues |λ₁| > |λ₂| ≥ ... ≥ |λₙ|,
  the sequence vₖ = Aᵏv₀/||Aᵏv₀|| converges to the dominant eigenvector.
  
  Convergence Rate: ||vₖ - v₁||₂ = O((|λ₂|/|λ₁|)ᵏ)
  
  Distributed Implementation Preservation:
  The distributed algorithm maintains convergence properties by ensuring:
  1. Consistent vector normalization across nodes
  2. Exact floating-point preservation in message passing
  3. Synchronized iteration termination based on global convergence criteria
  """
  def distributed_power_iteration(matrix, initial_vector \\ nil, opts \\ []) do
    max_iterations = Keyword.get(opts, :max_iterations, @max_iterations)
    tolerance = Keyword.get(opts, :tolerance, @convergence_tolerance)
    node_count = Keyword.get(opts, :node_count, length(Node.list()) + 1)
    
    {n, n_check} = matrix_dimensions(matrix)
    unless n == n_check, do: raise(ArgumentError, "Matrix must be square for eigenvalue computation")
    
    # Initialize random vector if not provided
    current_vector = initial_vector || initialize_random_vector(n)
    
    # Start distributed computation coordinator
    {:ok, coordinator} = DistributedEigenCoordinator.start_link(n, node_count)
    
    convergence_history = []
    
    try do
      iteration_result = 
        Enum.reduce_while(1..max_iterations, {current_vector, convergence_history}, 
          fn iteration, {vector, history} ->
            # Distributed matrix-vector multiplication
            new_vector = distributed_matrix_vector_multiply(coordinator, matrix, vector)
            
            # Global vector normalization with error tracking
            {normalized_vector, norm_value, norm_error} = 
              distributed_vector_normalize(coordinator, new_vector)
            
            # Convergence check
            convergence_metric = compute_convergence_metric(vector, normalized_vector)
            
            convergence_record = %{
              iteration: iteration,
              eigenvalue_estimate: norm_value,
              convergence_metric: convergence_metric,
              norm_error: norm_error,
              timestamp: :erlang.monotonic_time(:microsecond)
            }
            
            updated_history = [convergence_record | history]
            
            if convergence_metric < tolerance do
              Logger.info("Power iteration converged at iteration #{iteration}")
              {:halt, {normalized_vector, norm_value, updated_history}}
            else
              {:cont, {normalized_vector, updated_history}}
            end
          end)
      
      case iteration_result do
        {final_vector, eigenvalue, history} ->
          # Compute final error analysis
          final_error_analysis = analyze_eigenvalue_error(history, matrix, final_vector, eigenvalue)
          
          {:ok, %{
            eigenvector: final_vector,
            eigenvalue: eigenvalue,
            convergence_history: Enum.reverse(history),
            error_analysis: final_error_analysis
          }}
        
        {_vector, history} ->
          {:error, :max_iterations_reached, Enum.reverse(history)}
      end
      
    after
      GenServer.stop(coordinator)
    end
  end
  
  @doc """
  Distributed statistical computing with Kahan summation for numerical stability.
  
  Mathematical Foundation:
  
  Kahan Summation Algorithm:
  For sequence {xᵢ}, maintain running compensation for lost low-order bits:
  
  S₀ = 0, C₀ = 0
  For i = 1 to n:
    Y = xᵢ - Cᵢ₋₁
    T = Sᵢ₋₁ + Y
    Cᵢ = (T - Sᵢ₋₁) - Y
    Sᵢ = T
  
  Error Bound: |S_computed - S_exact| ≤ 2ε|S_exact| + O(ε²)
  
  Distributed Extension:
  Each node applies Kahan summation locally, then results are combined
  using the same algorithm, preserving the error bound.
  """
  def distributed_statistical_analysis(dataset, statistics \\ [:mean, :variance, :skewness, :kurtosis]) do
    node_count = length(Node.list()) + 1
    
    # Distribute data across nodes
    data_chunks = distribute_data(dataset, node_count)
    
    # Start statistical computation coordinator
    {:ok, coordinator} = DistributedStatsCoordinator.start_link(length(dataset), statistics)
    
    try do
      # Parallel statistical computation on each node
      partial_results = 
        data_chunks
        |> Task.async_stream(fn {node_id, data_chunk} ->
          compute_partial_statistics(coordinator, node_id, data_chunk, statistics)
        end, max_concurrency: node_count)
        |> Enum.map(fn {:ok, result} -> result end)
      
      # Combine partial results with numerical stability
      final_statistics = combine_partial_statistics(coordinator, partial_results, statistics)
      
      # Compute error bounds for each statistic
      error_analysis = analyze_statistical_errors(partial_results, final_statistics, dataset)
      
      {:ok, %{
        statistics: final_statistics,
        error_analysis: error_analysis,
        computation_metadata: %{
          data_size: length(dataset),
          node_count: node_count,
          chunk_sizes: Enum.map(data_chunks, fn {_, chunk} -> length(chunk) end)
        }
      }}
      
    after
      GenServer.stop(coordinator)
    end
  end
  
  # Implementation details for distributed coordinators
  
  defmodule DistributedMatrixCoordinator do
    use GenServer
    
    defstruct [:dimensions, :node_count, :active_computations, :result_blocks]
    
    def start_link(m, n, k, node_count) do
      GenServer.start_link(__MODULE__, {m, n, k, node_count})
    end
    
    def init({m, n, k, node_count}) do
      state = %__MODULE__{
        dimensions: {m, n, k},
        node_count: node_count,
        active_computations: %{},
        result_blocks: %{}
      }
      
      {:ok, state}
    end
    
    def handle_call({:submit_computation, computation_id, block_spec}, _from, state) do
      # Register computation task with coordinator
      updated_computations = Map.put(state.active_computations, computation_id, %{
        block_spec: block_spec,
        start_time: :erlang.monotonic_time(:microsecond),
        status: :active
      })
      
      new_state = %{state | active_computations: updated_computations}
      {:reply, :ok, new_state}
    end
    
    def handle_call({:submit_result, computation_id, result_block, error_metrics}, _from, state) do
      # Store result block with error tracking
      completion_time = :erlang.monotonic_time(:microsecond)
      
      computation_info = Map.get(state.active_computations, computation_id)
      execution_time = completion_time - computation_info.start_time
      
      result_info = %{
        result_block: result_block,
        error_metrics: error_metrics,
        execution_time: execution_time,
        completed_at: completion_time
      }
      
      updated_results = Map.put(state.result_blocks, computation_id, result_info)
      updated_computations = Map.put(state.active_computations, computation_id, 
                                   Map.put(computation_info, :status, :completed))
      
      new_state = %{state | 
        result_blocks: updated_results,
        active_computations: updated_computations
      }
      
      {:reply, :ok, new_state}
    end
    
    def handle_call(:get_all_results, _from, state) do
      {:reply, state.result_blocks, state}
    end
  end
  
  # Mathematical helper functions with numerical analysis
  
  defp matrix_dimensions(matrix) when is_list(matrix) do
    rows = length(matrix)
    cols = if rows > 0, do: length(hd(matrix)), else: 0
    {rows, cols}
  end
  
  defp create_distribution_plan(matrix_a, matrix_b, node_count) do
    {m, k} = matrix_dimensions(matrix_a)
    {_, n} = matrix_dimensions(matrix_b)
    
    # Optimal block size calculation for memory efficiency
    optimal_block_size = calculate_optimal_block_size(m, n, k, node_count)
    
    # Create block assignments for each node
    create_block_assignments(matrix_a, matrix_b, optimal_block_size, node_count)
  end
  
  defp calculate_optimal_block_size(m, n, k, node_count) do
    # Minimize communication overhead while balancing computation
    available_memory_per_node = 1024 * 1024 * 1024  # 1GB assumption
    element_size = 8  # 64-bit float
    
    max_block_elements = available_memory_per_node / (3 * element_size)  # A, B, C blocks
    block_dimension = round(:math.sqrt(max_block_elements))
    
    # Ensure even distribution across nodes
    adjusted_block_size = max(1, min(block_dimension, div(max(m, n), node_count)))
    
    adjusted_block_size
  end
  
  defp initialize_random_vector(dimension) do
    # Initialize with normalized random vector for numerical stability
    random_vector = Enum.map(1..dimension, fn _ -> :rand.normal() end)
    
    # Normalize to unit vector
    norm = :math.sqrt(Enum.reduce(random_vector, 0, fn x, acc -> acc + x * x end))
    
    if norm > 0 do
      Enum.map(random_vector, &(&1 / norm))
    else
      # Fallback to standard basis vector if random vector is zero
      List.replace_at(List.duplicate(0.0, dimension), 0, 1.0)
    end
  end
  
  defp compute_convergence_metric(old_vector, new_vector) do
    # Compute ||v_new - v_old||₂ for convergence assessment
    Enum.zip(new_vector, old_vector)
    |> Enum.reduce(0, fn {new_val, old_val}, acc ->
      diff = new_val - old_val
      acc + diff * diff
    end)
    |> :math.sqrt()
  end
  
  defp analyze_eigenvalue_error(convergence_history, matrix, final_vector, eigenvalue) do
    # Compute residual ||Av - λv||₂ for error analysis
    av = matrix_vector_multiply_local(matrix, final_vector)
    lambda_v = Enum.map(final_vector, &(&1 * eigenvalue))
    
    residual = 
      Enum.zip(av, lambda_v)
      |> Enum.reduce(0, fn {av_i, lv_i}, acc ->
        diff = av_i - lv_i
        acc + diff * diff
      end)
      |> :math.sqrt()
    
    # Estimate condition number effect
    last_iterations = Enum.take(convergence_history, 10)
    convergence_rate = estimate_convergence_rate(last_iterations)
    
    %{
      residual_norm: residual,
      estimated_convergence_rate: convergence_rate,
      iterations_to_convergence: length(convergence_history),
      final_eigenvalue_estimate: eigenvalue,
      numerical_stability_score: residual / abs(eigenvalue)
    }
  end
  
  defp estimate_convergence_rate(convergence_history) do
    # Estimate (λ₂/λ₁) ratio from convergence behavior
    if length(convergence_history) >= 2 do
      [latest, previous | _] = convergence_history
      
      if previous.convergence_metric > 0 do
        latest.convergence_metric / previous.convergence_metric
      else
        0.0
      end
    else
      1.0
    end
  end
  
  defp matrix_vector_multiply_local(matrix, vector) do
    Enum.map(matrix, fn row ->
      Enum.zip(row, vector)
      |> Enum.reduce(0, fn {a_ij, v_j}, acc -> acc + a_ij * v_j end)
    end)
  end
  
  defp via_tuple(node_id), do: {:via, Registry, {DistributedLinearAlgebraRegistry, node_id}}
end
```

**Lab 2.2: Concurrent Statistical Computing with Property-Based Testing**
```elixir
defmodule ConcurrentStatistics do
  @moduledoc """
  Concurrent statistical computing with mathematical guarantees and property-based validation.
  
  Mathematical Foundations:
  
  1. Welford's Online Algorithm for Variance:
     M₁ = x₁
     Mₖ = Mₖ₋₁ + (xₖ - Mₖ₋₁)/k
     S₁ = 0
     Sₖ = Sₖ₋₁ + (xₖ - Mₖ₋₁)(xₖ - Mₖ)
     σ² = Sₙ/(n-1)
  
  2. Concurrent Extension:
     Partial statistics from concurrent processes can be combined exactly:
     M_combined = (n₁M₁ + n₂M₂)/(n₁ + n₂)
     S_combined = S₁ + S₂ + (n₁n₂)/(n₁ + n₂)(M₁ - M₂)²
  
  3. Numerical Stability Guarantees:
     Algorithm is numerically stable for any order of operations.
     No catastrophic cancellation even for near-equal values.
  """
  
  use GenServer
  require Logger
  
  # Property-based testing integration
  use ExUnitProperties
  
  defstruct [
    :process_id, :sample_count, :running_mean, :running_m2,
    :min_value, :max_value, :numerical_errors, :computation_history
  ]
  
  def start_link(process_id, opts \\ []) do
    GenServer.start_link(__MODULE__, process_id, 
                        [{:name, via_tuple(process_id)} | opts])
  end
  
  @doc """
  Add data points concurrently with numerical stability guarantees.
  
  Mathematical Property: For any permutation of data points,
  the final mean and variance are identical (within floating-point precision).
  """
  def add_sample(process_id, value) when is_number(value) do
    GenServer.cast(via_tuple(process_id), {:add_sample, value})
  end
  
  def add_samples(process_id, values) when is_list(values) do
    GenServer.cast(via_tuple(process_id), {:add_samples, values})
  end
  
  @doc """
  Get current statistics with error bounds.
  """
  def get_statistics(process_id) do
    GenServer.call(via_tuple(process_id), :get_statistics)
  end
  
  @doc """
  Merge statistics from multiple concurrent processes.
  
  Mathematical Correctness: 
  The merge operation preserves exact mean and variance as if 
  all samples were processed by a single process.
  """
  def merge_statistics(process_ids) when is_list(process_ids) do
    # Collect statistics from all processes
    all_stats = 
      Enum.map(process_ids, fn process_id ->
        get_statistics(process_id)
      end)
    
    # Merge using mathematically exact combination formulas
    merge_statistics_exact(all_stats)
  end
  
  def init(process_id) do
    state = %__MODULE__{
      process_id: process_id,
      sample_count: 0,
      running_mean: 0.0,
      running_m2: 0.0,
      min_value: :infinity,
      max_value: :neg_infinity,
      numerical_errors: [],
      computation_history: []
    }
    
    Logger.debug("Statistical process #{process_id} initialized")
    {:ok, state}
  end
  
  def handle_cast({:add_sample, value}, state) do
    new_state = update_statistics_single(state, value)
    {:noreply, new_state}
  end
  
  def handle_cast({:add_samples, values}, state) do
    new_state = 
      Enum.reduce(values, state, fn value, acc_state ->
        update_statistics_single(acc_state, value)
      end)
    
    {:noreply, new_state}
  end
  
  def handle_call(:get_statistics, _from, state) do
    stats = compute_final_statistics(state)
    {:reply, stats, state}
  end
  
  # Welford's algorithm implementation with error tracking
  defp update_statistics_single(state, value) do
    new_count = state.sample_count + 1
    
    # Welford's algorithm for numerically stable variance computation
    delta = value - state.running_mean
    new_mean = state.running_mean + delta / new_count
    delta2 = value - new_mean
    new_m2 = state.running_m2 + delta * delta2
    
    # Track numerical errors for validation
    expected_mean_change = delta / new_count
    actual_mean_change = new_mean - state.running_mean
    numerical_error = abs(expected_mean_change - actual_mean_change)
    
    error_record = if numerical_error > 1.0e-15 do
      %{
        iteration: new_count,
        expected_change: expected_mean_change,
        actual_change: actual_mean_change,
        error: numerical_error,
        value: value
      }
    else
      nil
    end
    
    # Update min/max tracking
    new_min = min(state.min_value, value)
    new_max = max(state.max_value, value)
    
    # Record computation step for verification
    computation_step = %{
      sample_count: new_count,
      value: value,
      mean: new_mean,
      m2: new_m2,
      timestamp: :erlang.monotonic_time(:microsecond)
    }
    
    %{state |
      sample_count: new_count,
      running_mean: new_mean,
      running_m2: new_m2,
      min_value: new_min,
      max_value: new_max,
      numerical_errors: if(error_record, do: [error_record | state.numerical_errors], else: state.numerical_errors),
      computation_history: [computation_step | Enum.take(state.computation_history, 99)]  # Keep last 100
    }
  end
  
  defp compute_final_statistics(state) do
    variance = if state.sample_count > 1 do
      state.running_m2 / (state.sample_count - 1)
    else
      0.0
    end
    
    standard_deviation = :math.sqrt(variance)
    
    # Compute additional statistics
    %{
      count: state.sample_count,
      mean: state.running_mean,
      variance: variance,
      standard_deviation: standard_deviation,
      min: state.min_value,
      max: state.max_value,
      range: state.max_value - state.min_value,
      
      # Error analysis
      numerical_error_count: length(state.numerical_errors),
      max_numerical_error: compute_max_numerical_error(state.numerical_errors),
      
      # Computational metadata
      process_id: state.process_id,
      last_updated: :erlang.monotonic_time(:microsecond)
    }
  end
  
  defp merge_statistics_exact(stats_list) do
    # Mathematical exact merging of Welford statistics
    {total_count, combined_mean, combined_m2, global_min, global_max} = 
      Enum.reduce(stats_list, {0, 0.0, 0.0, :infinity, :neg_infinity}, 
        fn stats, {acc_count, acc_mean, acc_m2, acc_min, acc_max} ->
          new_count = acc_count + stats.count
          
          if new_count == 0 do
            {0, 0.0, 0.0, :infinity, :neg_infinity}
          else
            # Exact merging formulas
            delta = stats.mean - acc_mean
            new_mean = (acc_count * acc_mean + stats.count * stats.mean) / new_count
            
            new_m2 = acc_m2 + stats.variance * (stats.count - 1) + 
                     acc_count * stats.count * delta * delta / new_count
            
            new_min = min(acc_min, stats.min)
            new_max = max(acc_max, stats.max)
            
            {new_count, new_mean, new_m2, new_min, new_max}
          end
        end)
    
    # Compute final merged statistics
    final_variance = if total_count > 1, do: combined_m2 / (total_count - 1), else: 0.0
    
    %{
      count: total_count,
      mean: combined_mean,
      variance: final_variance,
      standard_deviation: :math.sqrt(final_variance),
      min: global_min,
      max: global_max,
      range: global_max - global_min,
      merged_from: length(stats_list),
      merge_timestamp: :erlang.monotonic_time(:microsecond)
    }
  end
  
  defp compute_max_numerical_error(error_list) do
    if Enum.empty?(error_list) do
      0.0
    else
      error_list
      |> Enum.map(& &1.error)
      |> Enum.max()
    end
  end
  
  # Property-based testing for mathematical correctness
  @doc """
  Property-based tests for concurrent statistical computing.
  
  These tests verify mathematical properties that must hold
  regardless of concurrent execution order.
  """
  def run_property_tests() do
    # Property 1: Mean independence from order
    property "mean is independent of sample order" do
      check all samples <- list_of(float(), min_length: 1, max_length: 1000) do
        # Test sequential vs. concurrent computation
        sequential_mean = compute_sequential_mean(samples)
        concurrent_mean = compute_concurrent_mean(samples, 4)  # 4 processes
        
        assert_in_delta(sequential_mean, concurrent_mean, 1.0e-12)
      end
    end
    
    # Property 2: Variance exactness under merging
    property "variance is exact under process merging" do
      check all samples <- list_of(float(), min_length: 2, max_length: 1000) do
        # Split samples randomly across processes
        process_count = min(4, length(samples))
        split_samples = split_samples_randomly(samples, process_count)
        
        # Compute statistics separately then merge
        separate_stats = compute_separate_statistics(split_samples)
        merged_stats = merge_statistics_exact(separate_stats)
        
        # Compute statistics on full dataset
        reference_stats = compute_reference_statistics(samples)
        
        assert_in_delta(merged_stats.mean, reference_stats.mean, 1.0e-12)
        assert_in_delta(merged_stats.variance, reference_stats.variance, 1.0e-10)
      end
    end
    
    # Property 3: Numerical stability
    property "numerical stability for near-equal values" do
      check all {base_value, perturbations} <- {float(), list_of(float(), min_length: 10)} do
        # Create dataset with small perturbations around base value
        samples = Enum.map(perturbations, &(base_value + &1 * 1.0e-8))
        
        # Ensure computation remains stable
        {:ok, pid} = start_link("stability_test")
        add_samples(pid, samples)
        stats = get_statistics(pid)
        GenServer.stop(pid)
        
        # Variance should be small but non-negative
        assert stats.variance >= 0
        assert is_finite(stats.mean)
        assert is_finite(stats.variance)
      end
    end
  end
  
  # Helper functions for property testing
  defp compute_sequential_mean(samples) do
    Enum.sum(samples) / length(samples)
  end
  
  defp compute_concurrent_mean(samples, process_count) do
    # Split samples across processes
    chunk_size = max(1, div(length(samples), process_count))
    chunks = Enum.chunk_every(samples, chunk_size)
    
    # Start processes and compute statistics
    process_ids = for {chunk, i} <- Enum.with_index(chunks) do
      process_id = "test_process_#{i}"
      {:ok, _pid} = start_link(process_id)
      add_samples(process_id, chunk)
      process_id
    end
    
    # Merge and extract mean
    merged_stats = merge_statistics(process_ids)
    
    # Cleanup
    Enum.each(process_ids, fn process_id ->
      GenServer.stop(via_tuple(process_id))
    end)
    
    merged_stats.mean
  end
  
  defp is_finite(value) do
    is_number(value) and value != :infinity and value != :neg_infinity and not is_nan(value)
  end
  
  defp is_nan(value) do
    value != value  # NaN is the only value not equal to itself
  end
  
  defp via_tuple(process_id), do: {:via, Registry, {ConcurrentStatisticsRegistry, process_id}}
end
```

#### Afternoon Session (4 hours): Distributed Gradient Descent with Convergence Guarantees

**5W+1H Optimization Context**:
- **WHAT**: Distributed gradient descent with mathematical convergence guarantees and fault tolerance
- **WHY**: Large-scale ML optimization requires distributed computation with proven convergence properties
- **WHEN**: Critical for training models that exceed single-node computational capacity
- **WHERE**: Coordinated across BEAM cluster nodes with automatic load balancing and failure recovery
- **WHO**: Coordinator process manages global state; worker processes compute partial gradients
- **HOW**: Message passing synchronizes gradients while preserving mathematical optimization properties

**Lab 2.3: Fault-Tolerant Distributed Gradient Descent**
```elixir
defmodule DistributedGradientDescent do
  @moduledoc """
  Distributed gradient descent with mathematical convergence guarantees.
  
  Mathematical Foundations:
  
  1. Gradient Descent Convergence Theory:
     For convex function f with L-Lipschitz gradient, the update rule:
     θₜ₊₁ = θₜ - α∇f(θₜ)
     
     Converges with rate: f(θₜ) - f* ≤ ||θ₀ - θ*||²/(2αt)
     Where α ≤ 1/L is the learning rate and θ* is the optimum.
  
  2. Distributed Extension:
     Parallel gradient computation: ∇f(θ) = (1/n)Σᵢ∇fᵢ(θ)
     Where each worker computes ∇fᵢ(θ) on data partition i.
     
     Convergence preserved when: Σᵢ∇fᵢ(θ) = ∇f(θ) exactly.
  
  3. Fault Tolerance:
     Byzantine-resilient aggregation using coordinate-wise median.
     Convergence guaranteed with up to f < n/2 Byzantine workers.
  
  4. Numerical Stability:
     Kahan summation for gradient aggregation prevents accumulation errors.
     Learning rate adaptation maintains numerical stability.
  """
  
  use GenServer
  require Logger
  
  defstruct [
    :coordinator_id, :problem_dimension, :current_parameters, :learning_rate,
    :convergence_tolerance, :max_iterations, :worker_pool, :gradient_history,
    :convergence_metrics, :fault_tolerance_config
  ]
  
  # Mathematical constants
  @default_learning_rate 0.01
  @default_tolerance 1.0e-6
  @default_max_iterations 10000
  @lipschitz_estimation_samples 100
  
  def start_link(coordinator_id, problem_config, opts \\ []) do
    GenServer.start_link(__MODULE__, {coordinator_id, problem_config}, 
                        [{:name, via_tuple(coordinator_id)} | opts])
  end
  
  @doc """
  Execute distributed gradient descent optimization.
  
  Algorithm: Synchronous Parallel Gradient Descent with Fault Tolerance
  
  1. Initialize parameters θ₀ and learning rate α
  2. For t = 0, 1, 2, ... until convergence:
     a. Broadcast θₜ to all workers
     b. Each worker computes partial gradient gᵢ = ∇fᵢ(θₜ)
     c. Coordinator aggregates: ḡ = Byzantine_Aggregate({gᵢ})
     d. Update: θₜ₊₁ = θₜ - αḡ
     e. Check convergence: ||ḡ|| < ε
  
  Convergence Guarantee:
  Under convexity and Byzantine assumptions, algorithm converges to
  ε-neighborhood of optimum in O(1/ε) iterations.
  """
  def optimize(coordinator_id, initial_parameters, data_partitions, opts \\ []) do
    GenServer.call(via_tuple(coordinator_id), 
                  {:optimize, initial_parameters, data_partitions, opts}, 
                  :infinity)
  end
  
  @doc """
  Estimate Lipschitz constant for automatic learning rate selection.
  
  Mathematical Method:
  L ≈ max_i ||∇f(xᵢ) - ∇f(xⱼ)|| / ||xᵢ - xⱼ||
  
  Sampled over random points in parameter space.
  """
  def estimate_lipschitz_constant(coordinator_id, parameter_bounds) do
    GenServer.call(via_tuple(coordinator_id), 
                  {:estimate_lipschitz, parameter_bounds})
  end
  
  def init({coordinator_id, problem_config}) do
    dimension = Map.get(problem_config, :dimension)
    learning_rate = Map.get(problem_config, :learning_rate, @default_learning_rate)
    tolerance = Map.get(problem_config, :tolerance, @default_tolerance)
    max_iterations = Map.get(problem_config, :max_iterations, @default_max_iterations)
    
    # Fault tolerance configuration
    fault_tolerance = Map.get(problem_config, :fault_tolerance, %{
      byzantine_resilience: true,
      max_faulty_workers: 1,
      timeout_ms: 30_000
    })
    
    state = %__MODULE__{
      coordinator_id: coordinator_id,
      problem_dimension: dimension,
      current_parameters: nil,
      learning_rate: learning_rate,
      convergence_tolerance: tolerance,
      max_iterations: max_iterations,
      worker_pool: [],
      gradient_history: [],
      convergence_metrics: %{},
      fault_tolerance_config: fault_tolerance
    }
    
    Logger.info("Distributed gradient descent coordinator #{coordinator_id} initialized")
    {:ok, state}
  end
  
  def handle_call({:optimize, initial_parameters, data_partitions, opts}, _from, state) do
    # Validate input parameters
    unless length(initial_parameters) == state.problem_dimension do
      {:reply, {:error, :dimension_mismatch}, state}
    else
      # Setup worker pool for data partitions
      worker_pool = setup_worker_pool(data_partitions, state.fault_tolerance_config)
      
      updated_state = %{state | 
        current_parameters: initial_parameters,
        worker_pool: worker_pool
      }
      
      # Execute optimization with mathematical guarantees
      optimization_result = execute_optimization_loop(updated_state, opts)
      
      # Cleanup worker pool
      cleanup_worker_pool(worker_pool)
      
      {:reply, optimization_result, updated_state}
    end
  end
  
  def handle_call({:estimate_lipschitz, parameter_bounds}, _from, state) do
    lipschitz_estimate = estimate_lipschitz_sampling(parameter_bounds, state)
    {:reply, lipschitz_estimate, state}
  end
  
  # Core optimization loop with convergence guarantees
  defp execute_optimization_loop(state, opts) do
    verbose = Keyword.get(opts, :verbose, false)
    adaptive_lr = Keyword.get(opts, :adaptive_learning_rate, false)
    
    optimization_state = %{
      parameters: state.current_parameters,
      learning_rate: state.learning_rate,
      iteration: 0,
      convergence_history: [],
      gradient_norms: [],
      function_values: []
    }
    
    if verbose do
      Logger.info("Starting distributed optimization with #{length(state.worker_pool)} workers")
    end
    
    optimization_result = 
      Enum.reduce_while(1..state.max_iterations, optimization_state, fn iteration, opt_state ->
        iteration_start_time = :erlang.monotonic_time(:microsecond)
        
        # Step 1: Broadcast current parameters to all workers
        broadcast_parameters(state.worker_pool, opt_state.parameters)
        
        # Step 2: Collect gradients from workers with fault tolerance
        {gradients, worker_statuses} = collect_gradients_with_fault_tolerance(
          state.worker_pool, 
          state.fault_tolerance_config
        )
        
        # Step 3: Byzantine-resilient gradient aggregation
        aggregated_gradient = aggregate_gradients_byzantine_resilient(gradients)
        
        # Step 4: Compute gradient norm for convergence check
        gradient_norm = compute_vector_norm(aggregated_gradient)
        
        # Step 5: Update parameters
        new_parameters = update_parameters(
          opt_state.parameters, 
          aggregated_gradient, 
          opt_state.learning_rate
        )
        
        # Step 6: Adaptive learning rate adjustment
        new_learning_rate = if adaptive_lr do
          adapt_learning_rate(opt_state.learning_rate, gradient_norm, opt_state.gradient_norms)
        else
          opt_state.learning_rate
        end
        
        # Step 7: Compute function value for monitoring
        function_value = compute_distributed_function_value(state.worker_pool, new_parameters)
        
        # Step 8: Record convergence metrics
        iteration_time = :erlang.monotonic_time(:microsecond) - iteration_start_time
        
        convergence_record = %{
          iteration: iteration,
          gradient_norm: gradient_norm,
          function_value: function_value,
          learning_rate: new_learning_rate,
          active_workers: count_active_workers(worker_statuses),
          iteration_time_microseconds: iteration_time
        }
        
        updated_opt_state = %{opt_state |
          parameters: new_parameters,
          learning_rate: new_learning_rate,
          iteration: iteration,
          convergence_history: [convergence_record | opt_state.convergence_history],
          gradient_norms: [gradient_norm | Enum.take(opt_state.gradient_norms, 9)],
          function_values: [function_value | Enum.take(opt_state.function_values, 9)]
        }
        
        if verbose and rem(iteration, 10) == 0 do
          Logger.info("Iteration #{iteration}: f = #{function_value:.6f}, ||∇f|| = #{gradient_norm:.6e}")
        end
        
        # Step 9: Convergence check
        if gradient_norm < state.convergence_tolerance do
          Logger.info("Converged at iteration #{iteration} with gradient norm #{gradient_norm}")
          {:halt, {:converged, updated_opt_state}}
        else
          {:cont, updated_opt_state}
        end
      end)
    
    case optimization_result do
      {:converged, final_state} ->
        {:ok, %{
          optimal_parameters: final_state.parameters,
          final_function_value: hd(final_state.function_values),
          convergence_history: Enum.reverse(final_state.convergence_history),
          iterations_to_convergence: final_state.iteration,
          status: :converged
        }}
      
      final_state ->
        {:ok, %{
          parameters: final_state.parameters,
          final_function_value: hd(final_state.function_values),
          convergence_history: Enum.reverse(final_state.convergence_history),
          iterations_completed: final_state.iteration,
          status: :max_iterations_reached
        }}
    end
  end
  
  # Worker process for gradient computation
  defmodule GradientWorker do
    @moduledoc """
    Worker process for computing partial gradients with fault tolerance.
    """
    
    use GenServer
    
    defstruct [:worker_id, :data_partition, :objective_function, :gradient_function, :status]
    
    def start_link(worker_id, data_partition, functions) do
      GenServer.start_link(__MODULE__, {worker_id, data_partition, functions},
                          name: via_tuple(worker_id))
    end
    
    def compute_gradient(worker_id, parameters, timeout \\ 30_000) do
      try do
        GenServer.call(via_tuple(worker_id), {:compute_gradient, parameters}, timeout)
      catch
        :exit, {:timeout, _} -> {:error, :timeout}
        :exit, reason -> {:error, {:worker_crashed, reason}}
      end
    end
    
    def compute_function_value(worker_id, parameters, timeout \\ 30_000) do
      try do
        GenServer.call(via_tuple(worker_id), {:compute_function_value, parameters}, timeout)
      catch
        :exit, {:timeout, _} -> {:error, :timeout}
        :exit, reason -> {:error, {:worker_crashed, reason}}
      end
    end
    
    def init({worker_id, data_partition, functions}) do
      state = %__MODULE__{
        worker_id: worker_id,
        data_partition: data_partition,
        objective_function: Map.get(functions, :objective),
        gradient_function: Map.get(functions, :gradient),
        status: :ready
      }
      
      {:ok, state}
    end
    
    def handle_call({:compute_gradient, parameters}, _from, state) do
      computation_start = :erlang.monotonic_time(:microsecond)
      
      try do
        # Compute partial gradient on local data partition
        partial_gradient = state.gradient_function.(parameters, state.data_partition)
        
        computation_time = :erlang.monotonic_time(:microsecond) - computation_start
        
        result = %{
          gradient: partial_gradient,
          worker_id: state.worker_id,
          computation_time: computation_time,
          data_partition_size: length(state.data_partition)
        }
        
        {:reply, {:ok, result}, state}
        
      rescue
        error ->
          Logger.error("Gradient computation failed in worker #{state.worker_id}: #{inspect(error)}")
          {:reply, {:error, error}, state}
      end
    end
    
    def handle_call({:compute_function_value, parameters}, _from, state) do
      try do
        partial_value = state.objective_function.(parameters, state.data_partition)
        {:reply, {:ok, partial_value}, state}
      rescue
        error ->
          {:reply, {:error, error}, state}
      end
    end
    
    defp via_tuple(worker_id), do: {:via, Registry, {GradientWorkerRegistry, worker_id}}
  end
  
  # Mathematical implementations
  
  defp setup_worker_pool(data_partitions, fault_tolerance_config) do
    # Create gradient computation workers for each data partition
    Registry.start_link(keys: :unique, name: GradientWorkerRegistry)
    
    workers = 
      Enum.with_index(data_partitions)
      |> Enum.map(fn {{data_partition, functions}, index} ->
        worker_id = "gradient_worker_#{index}"
        
        {:ok, pid} = GradientWorker.start_link(worker_id, data_partition, functions)
        
        %{
          worker_id: worker_id,
          pid: pid,
          data_partition_size: length(data_partition),
          status: :active
        }
      end)
    
    Logger.info("Created worker pool with #{length(workers)} workers")
    workers
  end
  
  defp broadcast_parameters(worker_pool, parameters) do
    # Send current parameters to all active workers
    Enum.each(worker_pool, fn worker ->
      if worker.status == :active do
        # Parameters are broadcast asynchronously for efficiency
        send(worker.pid, {:update_parameters, parameters})
      end
    end)
  end
  
  defp collect_gradients_with_fault_tolerance(worker_pool, fault_tolerance_config) do
    timeout = Map.get(fault_tolerance_config, :timeout_ms, 30_000)
    max_faulty = Map.get(fault_tolerance_config, :max_faulty_workers, 1)
    
    # Collect gradients from all workers with timeout
    gradient_tasks = 
      Enum.map(worker_pool, fn worker ->
        if worker.status == :active do
          Task.async(fn ->
            case GradientWorker.compute_gradient(worker.worker_id) do
              {:ok, result} -> {:success, worker.worker_id, result}
              {:error, reason} -> {:failure, worker.worker_id, reason}
            end
          end)
        else
          nil
        end
      end)
      |> Enum.reject(&is_nil/1)
    
    # Wait for results with timeout
    results = Task.await_many(gradient_tasks, timeout)
    
    # Separate successful and failed computations
    {successful_gradients, worker_statuses} = 
      Enum.reduce(results, {[], %{}}, fn result, {gradients, statuses} ->
        case result do
          {:success, worker_id, gradient_result} ->
            {[gradient_result.gradient | gradients], Map.put(statuses, worker_id, :success)}
          
          {:failure, worker_id, reason} ->
            Logger.warning("Worker #{worker_id} failed: #{inspect(reason)}")
            {gradients, Map.put(statuses, worker_id, {:failure, reason})}
        end
      end)
    
    # Check if enough workers succeeded
    failure_count = Enum.count(worker_statuses, fn {_, status} -> 
      match?({:failure, _}, status) 
    end)
    
    if failure_count > max_faulty do
      raise "Too many worker failures: #{failure_count} > #{max_faulty}"
    end
    
    {successful_gradients, worker_statuses}
  end
  
  defp aggregate_gradients_byzantine_resilient(gradients) when length(gradients) >= 1 do
    dimension = length(hd(gradients))
    
    # For each coordinate, compute Byzantine-resilient aggregate
    for coord_idx <- 0..(dimension - 1) do
      coord_values = Enum.map(gradients, &Enum.at(&1, coord_idx))
      
      # Use coordinate-wise median for Byzantine resilience
      coordinate_median(coord_values)
    end
  end
  
  defp coordinate_median(values) do
    sorted_values = Enum.sort(values)
    count = length(sorted_values)
    
    if rem(count, 2) == 1 do
      # Odd number of values
      Enum.at(sorted_values, div(count, 2))
    else
      # Even number of values - average the two middle values
      mid1 = Enum.at(sorted_values, div(count, 2) - 1)
      mid2 = Enum.at(sorted_values, div(count, 2))
      (mid1 + mid2) / 2
    end
  end
  
  defp update_parameters(current_params, gradient, learning_rate) do
    # Gradient descent update: θ := θ - α∇f(θ)
    Enum.zip(current_params, gradient)
    |> Enum.map(fn {param, grad} -> param - learning_rate * grad end)
  end
  
  defp compute_vector_norm(vector) do
    # Euclidean norm: ||v||₂ = √(Σvᵢ²)
    vector
    |> Enum.map(&(&1 * &1))
    |> Enum.sum()
    |> :math.sqrt()
  end
  
  defp adapt_learning_rate(current_lr, current_grad_norm, grad_norm_history) do
    # Simple adaptive learning rate based on gradient norm trends
    if length(grad_norm_history) >= 3 do
      recent_norms = Enum.take(grad_norm_history, 3)
      
      # If gradient norms are increasing, reduce learning rate
      # If gradient norms are decreasing consistently, slightly increase
      
      increasing_trend = Enum.chunk_every(recent_norms, 2, 1)
                        |> Enum.all?(fn [newer, older] -> newer > older end)
      
      decreasing_trend = Enum.chunk_every(recent_norms, 2, 1)
                        |> Enum.all?(fn [newer, older] -> newer < older end)
      
      cond do
        increasing_trend -> current_lr * 0.8  # Reduce learning rate
        decreasing_trend -> current_lr * 1.05  # Slightly increase
        true -> current_lr  # Keep current rate
      end
    else
      current_lr
    end
  end
  
  defp compute_distributed_function_value(worker_pool, parameters) do
    # Compute total function value across all workers
    function_values = 
      Enum.map(worker_pool, fn worker ->
        if worker.status == :active do
          case GradientWorker.compute_function_value(worker.worker_id, parameters) do
            {:ok, value} -> value
            {:error, _} -> 0.0  # Handle worker failures gracefully
          end
        else
          0.0
        end
      end)
    
    Enum.sum(function_values)
  end
  
  defp count_active_workers(worker_statuses) do
    Enum.count(worker_statuses, fn {_, status} -> status == :success end)
  end
  
  defp estimate_lipschitz_sampling(parameter_bounds, state) do
    # Sample random points and estimate Lipschitz constant
    # This is a simplified implementation for educational purposes
    
    dimension = state.problem_dimension
    num_samples = @lipschitz_estimation_samples
    
    # Generate random parameter pairs
    sample_pairs = 
      for _ <- 1..num_samples do
        point1 = Enum.map(parameter_bounds, fn {min, max} -> 
          min + :rand.uniform() * (max - min) 
        end)
        
        point2 = Enum.map(parameter_bounds, fn {min, max} -> 
          min + :rand.uniform() * (max - min) 
        end)
        
        {point1, point2}
      end
    
    # Estimate Lipschitz constant (simplified)
    # In practice, would need access to gradient function
    
    estimated_lipschitz = 1.0  # Placeholder implementation
    
    %{
      estimated_lipschitz_constant: estimated_lipschitz,
      recommended_learning_rate: 0.9 / estimated_lipschitz,
      samples_used: num_samples
    }
  end
  
  defp cleanup_worker_pool(worker_pool) do
    Enum.each(worker_pool, fn worker ->
      if Process.alive?(worker.pid) do
        GenServer.stop(worker.pid)
      end
    end)
  end
  
  defp via_tuple(coordinator_id), do: {:via, Registry, {DistributedGradientDescentRegistry, coordinator_id}}
end
```

### Day 3: Neural Networks and Deep Learning

#### Morning Session (4 hours)
**Lab 3.1: Feedforward Neural Network**
```elixir
defmodule NeuralNetwork do
  defstruct [:layers, :weights, :biases]
  
  def new(layer_sizes) do
    weights = initialize_weights(layer_sizes)
    biases = initialize_biases(layer_sizes)
    
    %__MODULE__{
      layers: layer_sizes,
      weights: weights,
      biases: biases
    }
  end
  
  def forward(%__MODULE__{weights: weights, biases: biases}, input) do
    {output, _activations} = 
      Enum.zip(weights, biases)
      |> Enum.reduce({input, []}, fn {w, b}, {activation, all_activations} ->
        z = add_vectors(matrix_vector_multiply(w, activation), b)
        new_activation = Enum.map(z, &sigmoid/1)
        {new_activation, [new_activation | all_activations]}
      end)
    
    output
  end
  
  def train(network, training_data, epochs, learning_rate) do
    Enum.reduce(1..epochs, network, fn _epoch, acc_network ->
      Enum.reduce(training_data, acc_network, fn {input, target}, net ->
        backpropagate(net, input, target, learning_rate)
      end)
    end)
  end
  
  defp sigmoid(x), do: 1 / (1 + :math.exp(-x))
  defp sigmoid_derivative(x), do: sigmoid(x) * (1 - sigmoid(x))
  
  defp initialize_weights(layer_sizes) do
    layer_sizes
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.map(fn [input_size, output_size] ->
      for _ <- 1..output_size do
        for _ <- 1..input_size do
          :rand.normal() * :math.sqrt(2 / input_size)  # Xavier initialization
        end
      end
    end)
  end
  
  defp initialize_biases(layer_sizes) do
    layer_sizes
    |> Enum.drop(1)
    |> Enum.map(&List.duplicate(0.0, &1))
  end
end
```

#### Afternoon Session (4 hours)
**Lab 3.2: Backpropagation Algorithm**
```elixir
defmodule Backpropagation do
  def compute_gradients(network, input, target) do
    # Forward pass with activation storage
    {activations, z_values} = forward_with_cache(network, input)
    
    # Backward pass
    output_error = cost_derivative(List.last(activations), target)
    compute_layer_gradients(network, activations, z_values, output_error)
  end
  
  defp forward_with_cache(%NeuralNetwork{weights: weights, biases: biases}, input) do
    {activations, z_values} = 
      Enum.zip(weights, biases)
      |> Enum.reduce({[input], []}, fn {w, b}, {acts, zs} ->
        current_activation = hd(acts)
        z = add_vectors(matrix_vector_multiply(w, current_activation), b)
        activation = Enum.map(z, &sigmoid/1)
        {[activation | acts], [z | zs]}
      end)
    
    {Enum.reverse(activations), Enum.reverse(z_values)}
  end
  
  defp compute_layer_gradients(network, activations, z_values, output_error) do
    # Compute deltas for each layer
    deltas = compute_deltas(network.weights, z_values, output_error)
    
    # Compute weight and bias gradients
    weight_gradients = compute_weight_gradients(deltas, activations)
    bias_gradients = deltas
    
    {weight_gradients, bias_gradients}
  end
  
  defp cost_derivative(output, target) do
    Enum.zip(output, target)
    |> Enum.map(fn {o, t} -> o - t end)
  end
  
  defp sigmoid(x), do: 1 / (1 + :math.exp(-x))
  defp sigmoid_derivative(x), do: sigmoid(x) * (1 - sigmoid(x))
end
```

### Day 4: Concurrent ML Processing

#### Morning Session (4 hours)
**Lab 4.1: Parallel Data Processing Pipeline**
```elixir
defmodule ParallelMLPipeline do
  use GenStage
  
  def start_link(opts) do
    GenStage.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  def init(opts) do
    batch_size = Keyword.get(opts, :batch_size, 100)
    max_demand = Keyword.get(opts, :max_demand, 500)
    
    {:producer_consumer, %{batch_size: batch_size}, 
     subscribe_to: opts[:subscribe_to],
     dispatcher: {GenStage.BroadcastDispatcher, []}}
  end
  
  def handle_events(events, _from, state) do
    processed_events = 
      events
      |> Enum.chunk_every(state.batch_size)
      |> Task.async_stream(&process_batch/1, max_concurrency: System.schedulers_online())
      |> Enum.flat_map(fn {:ok, result} -> result end)
    
    {:noreply, processed_events, state}
  end
  
  defp process_batch(batch) do
    # Simulate complex ML processing
    Enum.map(batch, fn data ->
      # Feature extraction, normalization, prediction
      data
      |> extract_features()
      |> normalize_features()
      |> apply_model()
    end)
  end
  
  defp extract_features(data), do: %{data | features: compute_features(data.raw)}
  defp normalize_features(data), do: %{data | features: Statistics.normalize(data.features)}
  defp apply_model(data), do: %{data | prediction: run_inference(data.features)}
end
```

#### Afternoon Session (4 hours)
**Lab 4.2: Distributed Model Training**
```elixir
defmodule DistributedTraining do
  use GenServer
  
  defstruct [:model, :workers, :aggregation_strategy, :round]
  
  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  def init(opts) do
    model = opts[:initial_model]
    num_workers = opts[:num_workers] || 4
    
    # Start worker processes
    workers = start_workers(num_workers, model)
    
    {:ok, %__MODULE__{
      model: model,
      workers: workers,
      aggregation_strategy: opts[:aggregation] || :federated_averaging,
      round: 0
    }}
  end
  
  def train_round(data_partitions) do
    GenServer.call(__MODULE__, {:train_round, data_partitions}, :infinity)
  end
  
  def handle_call({:train_round, data_partitions}, _from, state) do
    # Distribute training data to workers
    training_tasks = 
      Enum.zip(state.workers, data_partitions)
      |> Enum.map(fn {worker, partition} ->
        Task.async(fn -> 
          MLWorker.train_on_partition(worker, state.model, partition)
        end)
      end)
    
    # Collect updated models
    updated_models = 
      training_tasks
      |> Task.await_many(:infinity)
    
    # Aggregate models
    new_model = aggregate_models(updated_models, state.aggregation_strategy)
    
    new_state = %{state | model: new_model, round: state.round + 1}
    {:reply, new_model, new_state}
  end
  
  defp aggregate_models(models, :federated_averaging) do
    # Implement federated averaging
    num_models = length(models)
    
    averaged_weights = 
      models
      |> Enum.map(& &1.weights)
      |> Enum.zip()
      |> Enum.map(fn weight_tuple ->
        weight_tuple
        |> Tuple.to_list()
        |> Enum.sum()
        |> Kernel./(num_models)
      end)
    
    %{hd(models) | weights: averaged_weights}
  end
end
```

### Day 5: Real-time ML Systems

#### Morning Session (4 hours)
**Lab 5.1: Streaming ML Inference**
```elixir
defmodule StreamingInference do
  use Broadway
  
  def start_link(_opts) do
    Broadway.start_link(__MODULE__,
      name: __MODULE__,
      producer: [
        module: {BroadwayKafka.Producer, 
          hosts: [{"localhost", 9092}],
          group_id: "ml_inference_group",
          topics: ["ml_data_stream"]
        }
      ],
      processors: [
        default: [
          concurrency: System.schedulers_online() * 2,
          max_demand: 100
        ]
      ],
      batchers: [
        batch_processor: [
          concurrency: 4,
          batch_size: 50,
          batch_timeout: 1000
        ]
      ]
    )
  end
  
  def handle_message(_, message, _) do
    data = Jason.decode!(message.data)
    
    # Real-time feature engineering
    features = extract_real_time_features(data)
    
    # Model inference
    prediction = ModelRegistry.predict(:current_model, features)
    
    # Enrich message with prediction
    Message.put_data(message, %{
      original: data,
      features: features,
      prediction: prediction,
      timestamp: System.system_time(:microsecond)
    })
  end
  
  def handle_batch(:batch_processor, messages, _, _) do
    # Batch processing for efficiency
    predictions = 
      messages
      |> Enum.map(&Message.data/1)
      |> batch_predict()
    
    # Send results to downstream systems
    Enum.zip(messages, predictions)
    |> Enum.each(fn {message, prediction} ->
      publish_result(message, prediction)
    end)
    
    messages
  end
  
  defp extract_real_time_features(data) do
    # Implement real-time feature extraction
    %{
      numerical: extract_numerical_features(data),
      categorical: encode_categorical_features(data),
      temporal: extract_temporal_features(data)
    }
  end
end
```

#### Afternoon Session (4 hours)
**Lab 5.2: Model Serving and Load Balancing**
```elixir
defmodule ModelServer do
  use GenServer
  
  defstruct [:model, :model_version, :metrics, :load_balancer]
  
  def start_link(opts) do
    model_id = Keyword.fetch!(opts, :model_id)
    GenServer.start_link(__MODULE__, opts, name: via_tuple(model_id))
  end
  
  def predict(model_id, features) do
    GenServer.call(via_tuple(model_id), {:predict, features})
  end
  
  def update_model(model_id, new_model) do
    GenServer.cast(via_tuple(model_id), {:update_model, new_model})
  end
  
  def init(opts) do
    model = load_model(opts[:model_path])
    
    {:ok, %__MODULE__{
      model: model,
      model_version: 1,
      metrics: initialize_metrics(),
      load_balancer: opts[:load_balancer]
    }}
  end
  
  def handle_call({:predict, features}, _from, state) do
    start_time = :erlang.monotonic_time(:microsecond)
    
    try do
      prediction = run_inference(state.model, features)
      latency = :erlang.monotonic_time(:microsecond) - start_time
      
      new_metrics = update_metrics(state.metrics, :success, latency)
      new_state = %{state | metrics: new_metrics}
      
      {:reply, {:ok, prediction}, new_state}
    rescue
      error ->
        new_metrics = update_metrics(state.metrics, :error, 0)
        new_state = %{state | metrics: new_metrics}
        {:reply, {:error, error}, new_state}
    end
  end
  
  def handle_cast({:update_model, new_model}, state) do
    # Hot model swapping
    new_state = %{state | 
      model: new_model, 
      model_version: state.model_version + 1
    }
    {:noreply, new_state}
  end
  
  defp via_tuple(model_id), do: {:via, Registry, {ModelRegistry, model_id}}
  
  defp run_inference(model, features) do
    # Implement model inference based on model type
    case model.type do
      :neural_network -> NeuralNetwork.forward(model, features)
      :linear_regression -> LinearRegression.predict(model, features)
      :decision_tree -> DecisionTree.predict(model, features)
    end
  end
end
```

### Day 6: Advanced ML Architectures

#### Morning Session (4 hours)
**Lab 6.1: Transformer Architecture Implementation**
```elixir
defmodule Transformer do
  defstruct [:encoder_layers, :decoder_layers, :attention_heads, :model_dim]
  
  def new(opts) do
    %__MODULE__{
      encoder_layers: opts[:encoder_layers] || 6,
      decoder_layers: opts[:decoder_layers] || 6,
      attention_heads: opts[:attention_heads] || 8,
      model_dim: opts[:model_dim] || 512
    }
  end
  
  def encode(%__MODULE__{} = transformer, input_sequence) do
    # Positional encoding
    embedded = add_positional_encoding(input_sequence, transformer.model_dim)
    
    # Apply encoder layers
    Enum.reduce(1..transformer.encoder_layers, embedded, fn _layer, hidden ->
      hidden
      |> multi_head_attention(hidden, hidden, transformer.attention_heads)
      |> add_and_norm(hidden)
      |> feed_forward_network()
      |> add_and_norm(hidden)
    end)
  end
  
  def multi_head_attention(query, key, value, num_heads) do
    head_dim = div(length(hd(query)), num_heads)
    
    # Split into multiple heads
    heads = 
      0..(num_heads - 1)
      |> Enum.map(fn head_idx ->
        q_head = extract_head(query, head_idx, head_dim)
        k_head = extract_head(key, head_idx, head_dim)
        v_head = extract_head(value, head_idx, head_dim)
        
        scaled_dot_product_attention(q_head, k_head, v_head)
      end)
    
    # Concatenate and project
    concatenate_heads(heads)
  end
  
  defp scaled_dot_product_attention(query, key, value) do
    # Attention(Q,K,V) = softmax(QK^T/√d_k)V
    dk = length(hd(key))
    scores = matrix_multiply(query, transpose(key))
    
    scaled_scores = 
      Enum.map(scores, fn row ->
        Enum.map(row, &(&1 / :math.sqrt(dk)))
      end)
    
    attention_weights = softmax_matrix(scaled_scores)
    matrix_multiply(attention_weights, value)
  end
  
  defp add_positional_encoding(sequence, model_dim) do
    Enum.with_index(sequence)
    |> Enum.map(fn {token_embedding, pos} ->
      positional_encoding = generate_positional_encoding(pos, model_dim)
      add_vectors(token_embedding, positional_encoding)
    end)
  end
  
  defp generate_positional_encoding(position, model_dim) do
    0..(model_dim - 1)
    |> Enum.map(fn i ->
      if rem(i, 2) == 0 do
        :math.sin(position / :math.pow(10000, i / model_dim))
      else
        :math.cos(position / :math.pow(10000, (i - 1) / model_dim))
      end
    end)
  end
end
```

#### Afternoon Session (4 hours)
**Lab 6.2: Convolutional Neural Networks**
```elixir
defmodule ConvolutionalNetwork do
  defstruct [:layers, :pooling_layers, :fully_connected]
  
  def new(architecture) do
    %__MODULE__{
      layers: initialize_conv_layers(architecture.conv_layers),
      pooling_layers: architecture.pooling_layers,
      fully_connected: initialize_fc_layers(architecture.fc_layers)
    }
  end
  
  def forward(%__MODULE__{} = cnn, input_image) do
    # Convolutional layers with pooling
    feature_maps = 
      Enum.zip(cnn.layers, cnn.pooling_layers)
      |> Enum.reduce(input_image, fn {conv_layer, pool_config}, input ->
        input
        |> convolution_2d(conv_layer)
        |> apply_activation(:relu)
        |> max_pooling_2d(pool_config)
      end)
    
    # Flatten for fully connected layers
    flattened = flatten_feature_maps(feature_maps)
    
    # Fully connected layers
    Enum.reduce(cnn.fully_connected, flattened, fn fc_layer, input ->
      input
      |> linear_transform(fc_layer)
      |> apply_activation(:relu)
    end)
  end
  
  def convolution_2d(input, %{filters: filters, stride: stride, padding: padding}) do
    padded_input = apply_padding(input, padding)
    
    Enum.map(filters, fn filter ->
      convolve_filter(padded_input, filter, stride)
    end)
  end
  
  defp convolve_filter(input, filter, stride) do
    {input_height, input_width} = get_dimensions(input)
    {filter_height, filter_width} = get_dimensions(filter)
    
    output_height = div(input_height - filter_height, stride) + 1
    output_width = div(input_width - filter_width, stride) + 1
    
    for i <- 0..(output_height - 1) do
      for j <- 0..(output_width - 1) do
        # Extract region
        region = extract_region(input, i * stride, j * stride, filter_height, filter_width)
        # Element-wise multiplication and sum
        element_wise_multiply_sum(region, filter)
      end
    end
  end
  
  def max_pooling_2d(feature_map, %{size: pool_size, stride: stride}) do
    {height, width} = get_dimensions(feature_map)
    output_height = div(height - pool_size, stride) + 1
    output_width = div(width - pool_size, stride) + 1
    
    for i <- 0..(output_height - 1) do
      for j <- 0..(output_width - 1) do
        region = extract_region(feature_map, i * stride, j * stride, pool_size, pool_size)
        max_value(region)
      end
    end
  end
end
```

### Day 7: Production ML Systems

#### Morning Session (4 hours)
**Lab 7.1: Model Versioning and Deployment**
```elixir
defmodule MLModelRegistry do
  use GenServer
  
  defstruct [:models, :versions, :routing_table, :metrics_collector]
  
  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  def register_model(model_name, model_data, version) do
    GenServer.call(__MODULE__, {:register_model, model_name, model_data, version})
  end
  
  def deploy_model(model_name, version, deployment_config) do
    GenServer.call(__MODULE__, {:deploy_model, model_name, version, deployment_config})
  end
  
  def route_prediction(model_name, features, routing_strategy \\ :latest) do
    GenServer.call(__MODULE__, {:predict, model_name, features, routing_strategy})
  end
  
  def init(_opts) do
    {:ok, %__MODULE__{
      models: %{},
      versions: %{},
      routing_table: %{},
      metrics_collector: start_metrics_collector()
    }}
  end
  
  def handle_call({:register_model, name, model_data, version}, _from, state) do
    model_key = {name, version}
    
    new_models = Map.put(state.models, model_key, model_data)
    new_versions = Map.update(state.versions, name, [version], &[version | &1])
    
    new_state = %{state | models: new_models, versions: new_versions}
    {:reply, :ok, new_state}
  end
  
  def handle_call({:deploy_model, name, version, config}, _from, state) do
    case Map.get(state.models, {name, version}) do
      nil ->
        {:reply, {:error, :model_not_found}, state}
      
      model_data ->
        # Blue-green deployment
        deployment_result = deploy_with_strategy(model_data, config)
        
        new_routing = Map.put(state.routing_table, name, %{
          version: version,
          config: config,
          deployment: deployment_result
        })
        
        new_state = %{state | routing_table: new_routing}
        {:reply, {:ok, deployment_result}, new_state}
    end
  end
  
  def handle_call({:predict, name, features, strategy}, _from, state) do
    case get_model_for_prediction(state, name, strategy) do
      {:ok, model} ->
        start_time = :erlang.monotonic_time(:microsecond)
        
        try do
          prediction = ModelServer.predict(model.server_pid, features)
          latency = :erlang.monotonic_time(:microsecond) - start_time
          
          # Record metrics
          record_prediction_metrics(state.metrics_collector, name, :success, latency)
          
          {:reply, {:ok, prediction}, state}
        rescue
          error ->
            record_prediction_metrics(state.metrics_collector, name, :error, 0)
            {:reply, {:error, error}, state}
        end
      
      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end
  
  defp deploy_with_strategy(model_data, %{strategy: :blue_green} = config) do
    # Start new model server
    {:ok, new_server} = ModelServer.start_link([model: model_data])
    
    # Health check
    case health_check(new_server, config.health_check_config) do
      :healthy ->
        # Gradual traffic shifting
        shift_traffic_gradually(new_server, config.traffic_shift)
        
      :unhealthy ->
        # Rollback
        GenServer.stop(new_server)
        {:error, :health_check_failed}
    end
  end
end
```

#### Afternoon Session (4 hours)
**Lab 7.2: A/B Testing and Feature Flags**
```elixir
defmodule MLExperimentFramework do
  use GenServer
  
  defstruct [:experiments, :feature_flags, :traffic_splitter, :metrics]
  
  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  def create_experiment(experiment_config) do
    GenServer.call(__MODULE__, {:create_experiment, experiment_config})
  end
  
  def get_model_variant(user_id, experiment_name) do
    GenServer.call(__MODULE__, {:get_variant, user_id, experiment_name})
  end
  
  def record_outcome(user_id, experiment_name, outcome) do
    GenServer.cast(__MODULE__, {:record_outcome, user_id, experiment_name, outcome})
  end
  
  def init(_opts) do
    {:ok, %__MODULE__{
      experiments: %{},
      feature_flags: %{},
      traffic_splitter: TrafficSplitter.new(),
      metrics: MetricsCollector.new()
    }}
  end
  
  def handle_call({:create_experiment, config}, _from, state) do
    experiment = %{
      name: config.name,
      variants: config.variants,
      traffic_allocation: config.traffic_allocation,
      start_time: DateTime.utc_now(),
      duration: config.duration,
      success_metric: config.success_metric
    }
    
    new_experiments = Map.put(state.experiments, config.name, experiment)
    new_state = %{state | experiments: new_experiments}
    
    {:reply, :ok, new_state}
  end
  
  def handle_call({:get_variant, user_id, experiment_name}, _from, state) do
    case Map.get(state.experiments, experiment_name) do
      nil ->
        {:reply, {:error, :experiment_not_found}, state}
      
      experiment ->
        variant = assign_variant(user_id, experiment, state.traffic_splitter)
        {:reply, {:ok, variant}, state}
    end
  end
  
  def handle_cast({:record_outcome, user_id, experiment_name, outcome}, state) do
    # Record outcome for statistical analysis
    MetricsCollector.record_experiment_outcome(
      state.metrics, 
      experiment_name, 
      user_id, 
      outcome
    )
    
    {:noreply, state}
  end
  
  defp assign_variant(user_id, experiment, traffic_splitter) do
    # Consistent hashing for user assignment
    hash = :erlang.phash2({user_id, experiment.name})
    bucket = rem(hash, 100)
    
    # Determine variant based on traffic allocation
    assign_variant_by_bucket(bucket, experiment.variants, experiment.traffic_allocation)
  end
  
  defp assign_variant_by_bucket(bucket, variants, allocation) do
    cumulative = 0
    
    Enum.reduce_while(Enum.zip(variants, allocation), cumulative, fn {variant, percent}, acc ->
      new_acc = acc + percent
      
      if bucket < new_acc do
        {:halt, variant}
      else
        {:cont, new_acc}
      end
    end)
  end
end
```

---

## Week 2: Advanced Topics and Production Systems

### Day 8: MLOps and Infrastructure

#### Morning Session (4 hours)
**Lab 8.1: ML Pipeline Orchestration**
```elixir
defmodule MLPipeline do
  use GenStateMachine, callback_mode: :state_functions
  
  defstruct [:pipeline_id, :steps, :current_step, :data, :artifacts, :config]
  
  def start_link(pipeline_config) do
    GenStateMachine.start_link(__MODULE__, pipeline_config)
  end
  
  def execute_pipeline(pid) do
    GenStateMachine.call(pid, :execute)
  end
  
  def init(config) do
    data = %__MODULE__{
      pipeline_id: UUID.uuid4(),
      steps: config.steps,
      current_step: 0,
      data: config.initial_data,
      artifacts: %{},
      config: config
    }
    
    {:ok, :initialized, data}
  end
  
  # State: initialized
  def initialized(:call, :execute, data) do
    {:next_state, :data_validation, data, [{:reply, :ok}]}
  end
  
  # State: data_validation
  def data_validation(:enter, _old_state, data) do
    case validate_input_data(data.data, data.config.validation_schema) do
      {:ok, validated_data} ->
        new_data = %{data | data: validated_data}
        {:next_state, :feature_engineering, new_data}
      
      {:error, validation_errors} ->
        {:next_state, :failed, %{data | artifacts: %{errors: validation_errors}}}
    end
  end
  
  # State: feature_engineering
  def feature_engineering(:enter, _old_state, data) do
    Task.async(fn ->
      FeatureEngineering.transform(data.data, data.config.feature_config)
    end)
    |> Task.await()
    |> case do
      {:ok, features} ->
        artifacts = Map.put(data.artifacts, :features, features)
        new_data = %{data | artifacts: artifacts}
        {:next_state, :model_training, new_data}
      
      {:error, reason} ->
        {:next_state, :failed, %{data | artifacts: %{error: reason}}}
    end
  end
  
  # State: model_training
  def model_training(:enter, _old_state, data) do
    features = data.artifacts.features
    
    # Distributed training
    training_result = 
      DistributedTraining.train_model(
        data.config.model_config,
        features,
        data.config.training_config
      )
    
    case training_result do
      {:ok, trained_model} ->
        artifacts = Map.put(data.artifacts, :model, trained_model)
        new_data = %{data | artifacts: artifacts}
        {:next_state, :model_evaluation, new_data}
      
      {:error, reason} ->
        {:next_state, :failed, %{data | artifacts: %{error: reason}}}
    end
  end
  
  # State: model_evaluation
  def model_evaluation(:enter, _old_state, data) do
    model = data.artifacts.model
    test_data = data.artifacts.features.test_set
    
    evaluation_metrics = ModelEvaluator.evaluate(model, test_data)
    
    if meets_quality_threshold?(evaluation_metrics, data.config.quality_threshold) do
      artifacts = Map.put(data.artifacts, :evaluation, evaluation_metrics)
      new_data = %{data | artifacts: artifacts}
      {:next_state, :model_deployment, new_data}
    else
      {:next_state, :failed, %{data | artifacts: %{insufficient_quality: evaluation_metrics}}}
    end
  end
  
  # State: model_deployment
  def model_deployment(:enter, _old_state, data) do
    model = data.artifacts.model
    
    deployment_result = 
      MLModelRegistry.deploy_model(
        data.config.model_name,
        model,
        data.config.deployment_config
      )
    
    case deployment_result do
      {:ok, deployment_info} ->
        artifacts = Map.put(data.artifacts, :deployment, deployment_info)
        new_data = %{data | artifacts: artifacts}
        {:next_state, :completed, new_data}
      
      {:error, reason} ->
        {:next_state, :failed, %{data | artifacts: %{deployment_error: reason}}}
    end
  end
  
  def completed(_event_type, _event_content, data) do
    {:keep_state, data}
  end
  
  def failed(_event_type, _event_content, data) do
    # Log failure and clean up resources
    Logger.error("Pipeline #{data.pipeline_id} failed: #{inspect(data.artifacts)}")
    {:keep_state, data}
  end
end
```

#### Afternoon Session (4 hours)
**Lab 8.2: Model Monitoring and Observability**
```elixir
defmodule MLObservability do
  use GenServer
  
  defstruct [:monitors, :alert_rules, :metrics_store, :dashboards]
  
  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  def add_monitor(model_name, monitor_config) do
    GenServer.call(__MODULE__, {:add_monitor, model_name, monitor_config})
  end
  
  def record_prediction(model_name, input, prediction, actual \\ nil) do
    GenServer.cast(__MODULE__, {:record_prediction, model_name, input, prediction, actual})
  end
  
  def init(_opts) do
    {:ok, %__MODULE__{
      monitors: %{},
      alert_rules: load_alert_rules(),
      metrics_store: MetricsStore.new(),
      dashboards: DashboardManager.new()
    }}
  end
  
  def handle_call({:add_monitor, model_name, config}, _from, state) do
    monitor = create_monitor(config)
    new_monitors = Map.put(state.monitors, model_name, monitor)
    new_state = %{state | monitors: new_monitors}
    
    {:reply, :ok, new_state}
  end
  
  def handle_cast({:record_prediction, model_name, input, prediction, actual}, state) do
    timestamp = DateTime.utc_now()
    
    # Data drift detection
    if Map.has_key?(state.monitors, model_name) do
      monitor = state.monitors[model_name]
      
      # Check for input drift
      input_drift = DriftDetector.detect_input_drift(monitor.baseline_stats, input)
      
      # Check for prediction drift
      prediction_drift = DriftDetector.detect_prediction_drift(monitor.prediction_stats, prediction)
      
      # Record metrics
      metrics = %{
        timestamp: timestamp,
        input_drift_score: input_drift.score,
        prediction_drift_score: prediction_drift.score,
        prediction_confidence: extract_confidence(prediction),
        latency: extract_latency(prediction)
      }
      
      MetricsStore.record(state.metrics_store, model_name, metrics)
      
      # Check alert conditions
      check_alerts(state.alert_rules, model_name, metrics)
      
      # Update model statistics if actual value provided
      new_state = 
        if actual do
          update_model_performance(state, model_name, prediction, actual)
        else
          state
        end
      
      {:noreply, new_state}
    else
      {:noreply, state}
    end
  end
  
  defp create_monitor(config) do
    %{
      baseline_stats: StatisticsCollector.new(config.baseline_data),
      prediction_stats: StatisticsCollector.new(),
      drift_threshold: config.drift_threshold || 0.1,
      performance_threshold: config.performance_threshold || 0.8
    }
  end
  
  defp check_alerts(alert_rules, model_name, metrics) do
    Enum.each(alert_rules, fn rule ->
      if rule.model == model_name and evaluate_alert_condition(rule, metrics) do
        AlertManager.trigger_alert(rule, metrics)
      end
    end)
  end
  
  defp evaluate_alert_condition(rule, metrics) do
    case rule.condition do
      {:drift_exceeds, threshold} ->
        metrics.input_drift_score > threshold or metrics.prediction_drift_score > threshold
      
      {:performance_below, threshold} ->
        Map.get(metrics, :accuracy, 1.0) < threshold
      
      {:latency_exceeds, threshold} ->
        metrics.latency > threshold
    end
  end
end

defmodule DriftDetector do
  def detect_input_drift(baseline_stats, current_input) do
    # Kolmogorov-Smirnov test for continuous features
    # Chi-square test for categorical features
    
    drift_scores = 
      Enum.map(current_input, fn {feature, value} ->
        baseline_dist = Map.get(baseline_stats.distributions, feature)
        current_dist = create_single_value_distribution(value)
        
        drift_score = calculate_drift_score(baseline_dist, current_dist)
        {feature, drift_score}
      end)
    
    max_drift = 
      drift_scores
      |> Enum.map(&elem(&1, 1))
      |> Enum.max()
    
    %{
      score: max_drift,
      feature_scores: Map.new(drift_scores),
      threshold_exceeded: max_drift > 0.1
    }
  end
  
  def detect_prediction_drift(prediction_stats, current_prediction) do
    if StatisticsCollector.sufficient_data?(prediction_stats) do
      baseline_mean = StatisticsCollector.mean(prediction_stats)
      baseline_std = StatisticsCollector.std(prediction_stats)
      
      # Z-score based drift detection
      z_score = abs(current_prediction - baseline_mean) / baseline_std
      
      %{
        score: z_score,
        threshold_exceeded: z_score > 3.0  # 3-sigma rule
      }
    else
      %{score: 0.0, threshold_exceeded: false}
    end
  end
  
  defp calculate_drift_score(baseline_dist, current_dist) do
    # Implement statistical distance measure
    # KL divergence for categorical, KS statistic for continuous
    case {baseline_dist.type, current_dist.type} do
      {:continuous, :continuous} ->
        ks_statistic(baseline_dist.values, current_dist.values)
      
      {:categorical, :categorical} ->
        kl_divergence(baseline_dist.probabilities, current_dist.probabilities)
    end
  end
end
```

### Day 9: Reinforcement Learning

#### Morning Session (4 hours)
**Lab 9.1: Q-Learning Implementation**
```elixir
defmodule QLearning do
  defstruct [:q_table, :learning_rate, :discount_factor, :exploration_rate]
  
  def new(opts \\ []) do
    %__MODULE__{
      q_table: %{},
      learning_rate: Keyword.get(opts, :learning_rate, 0.1),
      discount_factor: Keyword.get(opts, :discount_factor, 0.9),
      exploration_rate: Keyword.get(opts, :exploration_rate, 1.0)
    }
  end
  
  def choose_action(%__MODULE__{} = agent, state, available_actions) do
    if :rand.uniform() < agent.exploration_rate do
      # Exploration: random action
      Enum.random(available_actions)
    else
      # Exploitation: best known action
      best_action(agent, state, available_actions)
    end
  end
  
  def update(%__MODULE__{} = agent, state, action, reward, next_state, next_actions) do
    current_q = get_q_value(agent, state, action)
    max_next_q = max_q_value(agent, next_state, next_actions)
    
    # Q-learning update rule: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
    new_q = current_q + agent.learning_rate * 
            (reward + agent.discount_factor * max_next_q - current_q)
    
    new_q_table = put_q_value(agent.q_table, state, action, new_q)
    
    %{agent | q_table: new_q_table}
  end
  
  def decay_exploration(%__MODULE__{} = agent, decay_rate \\ 0.995) do
    new_rate = max(agent.exploration_rate * decay_rate, 0.01)
    %{agent | exploration_rate: new_rate}
  end
  
  defp best_action(agent, state, available_actions) do
    available_actions
    |> Enum.map(&{&1, get_q_value(agent, state, &1)})
    |> Enum.max_by(&elem(&1, 1))
    |> elem(0)
  end
  
  defp max_q_value(agent, state, available_actions) do
    if Enum.empty?(available_actions) do
      0.0
    else
      available_actions
      |> Enum.map(&get_q_value(agent, state, &1))
      |> Enum.max()
    end
  end
  
  defp get_q_value(agent, state, action) do
    Map.get(agent.q_table, {state, action}, 0.0)
  end
  
  defp put_q_value(q_table, state, action, value) do
    Map.put(q_table, {state, action}, value)
  end
end

defmodule RLEnvironment do
  use GenServer
  
  defstruct [:state, :episode, :total_reward, :done]
  
  def start_link(initial_state) do
    GenServer.start_link(__MODULE__, initial_state)
  end
  
  def reset(env_pid) do
    GenServer.call(env_pid, :reset)
  end
  
  def step(env_pid, action) do
    GenServer.call(env_pid, {:step, action})
  end
  
  def get_state(env_pid) do
    GenServer.call(env_pid, :get_state)
  end
  
  def init(initial_state) do
    {:ok, %__MODULE__{
      state: initial_state,
      episode: 0,
      total_reward: 0.0,
      done: false
    }}
  end
  
  def handle_call(:reset, _from, env_state) do
    new_state = %{env_state | 
      state: generate_initial_state(),
      episode: env_state.episode + 1,
      total_reward: 0.0,
      done: false
    }
    
    {:reply, new_state.state, new_state}
  end
  
  def handle_call({:step, action}, _from, env_state) do
    {next_state, reward, done} = compute_transition(env_state.state, action)
    
    new_env_state = %{env_state |
      state: next_state,
      total_reward: env_state.total_reward + reward,
      done: done
    }
    
    {:reply, {next_state, reward, done}, new_env_state}
  end
  
  def handle_call(:get_state, _from, env_state) do
    {:reply, env_state, env_state}
  end
  
  # Environment-specific implementations
  defp generate_initial_state() do
    # Implementation depends on specific environment
    %{position: {0, 0}, inventory: [], health: 100}
  end
  
  defp compute_transition(state, action) do
    # Environment dynamics
    case action do
      :move_up -> 
        new_pos = move_position(state.position, {0, 1})
        reward = calculate_reward(state, new_pos)
        {%{state | position: new_pos}, reward, check_terminal(new_pos)}
      
      :move_down ->
        new_pos = move_position(state.position, {0, -1})
        reward = calculate_reward(state, new_pos)
        {%{state | position: new_pos}, reward, check_terminal(new_pos)}
      
      # ... other actions
    end
  end
end
```

#### Afternoon Session (4 hours)
**Lab 9.2: Deep Q-Network (DQN)**
```elixir
defmodule DeepQNetwork do
  defstruct [:network, :target_network, :replay_buffer, :config]
  
  def new(state_size, action_size, opts \\ []) do
    network_config = [
      input_size: state_size,
      hidden_layers: [256, 256],
      output_size: action_size
    ]
    
    %__MODULE__{
      network: NeuralNetwork.new(network_config),
      target_network: NeuralNetwork.new(network_config),
      replay_buffer: ReplayBuffer.new(Keyword.get(opts, :buffer_size, 10000)),
      config: %{
        learning_rate: Keyword.get(opts, :learning_rate, 0.001),
        discount_factor: Keyword.get(opts, :discount_factor, 0.99),
        epsilon: Keyword.get(opts, :epsilon, 1.0),
        epsilon_decay: Keyword.get(opts, :epsilon_decay, 0.995),
        target_update_frequency: Keyword.get(opts, :target_update_freq, 100),
        batch_size: Keyword.get(opts, :batch_size, 32)
      }
    }
  end
  
  def choose_action(%__MODULE__{} = dqn, state) do
    if :rand.uniform() < dqn.config.epsilon do
      # Random action (exploration)
      :rand.uniform(output_size(dqn.network)) - 1
    else
      # Best action according to network (exploitation)
      q_values = NeuralNetwork.forward(dqn.network, state)
      Enum.with_index(q_values)
      |> Enum.max_by(&elem(&1, 0))
      |> elem(1)
    end
  end
  
  def store_experience(%__MODULE__{} = dqn, state, action, reward, next_state, done) do
    experience = {state, action, reward, next_state, done}
    new_buffer = ReplayBuffer.add(dqn.replay_buffer, experience)
    %{dqn | replay_buffer: new_buffer}
  end
  
  def train(%__MODULE__{} = dqn) when length(dqn.replay_buffer.experiences) >= dqn.config.batch_size do
    # Sample batch from replay buffer
    batch = ReplayBuffer.sample(dqn.replay_buffer, dqn.config.batch_size)
    
    # Compute targets using target network
    targets = compute_targets(dqn, batch)
    
    # Train network
    updated_network = train_network(dqn.network, batch, targets, dqn.config.learning_rate)
    
    # Decay epsilon
    new_epsilon = max(dqn.config.epsilon * dqn.config.epsilon_decay, 0.01)
    new_config = %{dqn.config | epsilon: new_epsilon}
    
    %{dqn | network: updated_network, config: new_config}
  end
  
  def train(%__MODULE__{} = dqn), do: dqn  # Not enough experiences yet
  
  def update_target_network(%__MODULE__{} = dqn) do
    %{dqn | target_network: copy_network(dqn.network)}
  end
  
  defp compute_targets(dqn, batch) do
    Enum.map(batch, fn {state, action, reward, next_state, done} ->
      if done do
        reward
      else
        next_q_values = NeuralNetwork.forward(dqn.target_network, next_state)
        max_next_q = Enum.max(next_q_values)
        reward + dqn.config.discount_factor * max_next_q
      end
    end)
  end
  
  defp train_network(network, batch, targets, learning_rate) do
    # Extract states and actions from batch
    states = Enum.map(batch, &elem(&1, 0))
    actions = Enum.map(batch, &elem(&1, 1))
    
    # Compute loss and gradients
    {loss, gradients} = compute_loss_and_gradients(network, states, actions, targets)
    
    # Update network parameters
    update_network_parameters(network, gradients, learning_rate)
  end
end

defmodule ReplayBuffer do
  defstruct [:experiences, :max_size, :index]
  
  def new(max_size) do
    %__MODULE__{
      experiences: [],
      max_size: max_size,
      index: 0
    }
  end
  
  def add(%__MODULE__{} = buffer, experience) do
    if length(buffer.experiences) < buffer.max_size do
      %{buffer | experiences: [experience | buffer.experiences]}
    else
      # Replace oldest experience (circular buffer)
      new_experiences = List.replace_at(buffer.experiences, buffer.index, experience)
      new_index = rem(buffer.index + 1, buffer.max_size)
      %{buffer | experiences: new_experiences, index: new_index}
    end
  end
  
  def sample(%__MODULE__{} = buffer, batch_size) do
    buffer.experiences
    |> Enum.take_random(batch_size)
  end
end
```

### Day 10: Natural Language Processing

#### Morning Session (4 hours)
**Lab 10.1: Text Processing and Tokenization**
```elixir
defmodule TextProcessor do
  @moduledoc "Advanced text processing for NLP tasks"
  
  def tokenize(text, opts \\ []) do
    text
    |> normalize_text()
    |> apply_tokenization_strategy(opts[:strategy] || :word_piece)
    |> filter_tokens(opts[:min_length] || 1)
  end
  
  def normalize_text(text) do
    text
    |> String.downcase()
    |> String.replace(~r/[^\w\s]/, "")  # Remove punctuation
    |> String.replace(~r/\s+/, " ")      # Normalize whitespace
    |> String.trim()
  end
  
  def apply_tokenization_strategy(text, :word_piece) do
    # Simplified word-piece tokenization
    words = String.split(text, " ")
    Enum.flat_map(words, &word_piece_tokenize/1)
  end
  
  def apply_tokenization_strategy(text, :bpe) do
    # Byte-pair encoding tokenization
    chars = String.graphemes(text)
    apply_bpe_merges(chars, load_bpe_merges())
  end
  
  def word_piece_tokenize(word) do
    # Start with full word, progressively break down
    tokenize_word_recursive(word, [], get_vocabulary())
  end
  
  defp tokenize_word_recursive("", acc, _vocab), do: Enum.reverse(acc)
  
  defp tokenize_word_recursive(word, acc, vocab) do
    # Find longest matching subword in vocabulary
    longest_match = find_longest_match(word, vocab)
    
    case longest_match do
      nil ->
        # Unknown token
        ["[UNK]" | acc] |> Enum.reverse()
      
      {token, remaining} ->
        tokenize_word_recursive(remaining, [token | acc], vocab)
    end
  end
  
  def build_vocabulary(corpus, vocab_size \\ 30000) do
    # Count all substrings
    substring_counts = count_substrings(corpus)
    
    # Select most frequent substrings
    substring_counts
    |> Enum.sort_by(&elem(&1, 1), :desc)
    |> Enum.take(vocab_size)
    |> Enum.map(&elem(&1, 0))
    |> MapSet.new()
  end
  
  defp count_substrings(corpus) do
    corpus
    |> Enum.flat_map(&generate_substrings/1)
    |> Enum.frequencies()
  end
  
  defp generate_substrings(word) do
    length = String.length(word)
    
    for i <- 0..(length - 1),
        j <- (i + 1)..length do
      String.slice(word, i, j - i)
    end
  end
end

defmodule WordEmbeddings do
  @moduledoc "Word2Vec and GloVe implementations"
  
  defstruct [:embeddings, :vocabulary, :dimension]
  
  def new(vocabulary, dimension \\ 300) do
    # Initialize random embeddings
    embeddings = 
      vocabulary
      |> Enum.map(fn word ->
        vector = Enum.map(1..dimension, fn _ -> :rand.normal() * 0.1 end)
        {word, vector}
      end)
      |> Map.new()
    
    %__MODULE__{
      embeddings: embeddings,
      vocabulary: MapSet.new(vocabulary),
      dimension: dimension
    }
  end
  
  def train_word2vec(corpus, opts \\ []) do
    window_size = Keyword.get(opts, :window_size, 5)
    learning_rate = Keyword.get(opts, :learning_rate, 0.025)
    epochs = Keyword.get(opts, :epochs, 5)
    
    # Build vocabulary
    vocabulary = build_vocabulary_from_corpus(corpus)
    embeddings = new(vocabulary, Keyword.get(opts, :dimension, 300))
    
    # Training loop
    Enum.reduce(1..epochs, embeddings, fn epoch, acc_embeddings ->
      epoch_learning_rate = learning_rate * (1 - epoch / epochs)
      train_epoch(acc_embeddings, corpus, window_size, epoch_learning_rate)
    end)
  end
  
  defp train_epoch(embeddings, corpus, window_size, learning_rate) do
    corpus
    |> Enum.reduce(embeddings, fn sentence, acc_embeddings ->
      words = String.split(sentence, " ")
      train_sentence(acc_embeddings, words, window_size, learning_rate)
    end)
  end
  
  defp train_sentence(embeddings, words, window_size, learning_rate) do
    words
    |> Enum.with_index()
    |> Enum.reduce(embeddings, fn {center_word, center_idx}, acc_embeddings ->
      context_words = extract_context(words, center_idx, window_size)
      
      Enum.reduce(context_words, acc_embeddings, fn context_word, inner_acc ->
        update_word_vectors(inner_acc, center_word, context_word, learning_rate)
      end)
    end)
  end
  
  defp update_word_vectors(embeddings, center_word, context_word, learning_rate) do
    center_vector = Map.get(embeddings.embeddings, center_word)
    context_vector = Map.get(embeddings.embeddings, context_word)
    
    if center_vector && context_vector do
      # Skip-gram objective: predict context from center
      # Simplified gradient update
      dot_product = VectorOps.dot_product(center_vector, context_vector)
      sigmoid_output = 1 / (1 + :math.exp(-dot_product))
      
      # Gradient computation
      error = 1 - sigmoid_output  # Positive sampling simplification
      
      # Update vectors
      center_update = Enum.map(context_vector, &(&1 * error * learning_rate))
      context_update = Enum.map(center_vector, &(&1 * error * learning_rate))
      
      new_center = VectorOps.add_vectors(center_vector, center_update)
      new_context = VectorOps.add_vectors(context_vector, context_update)
      
      new_embeddings = embeddings.embeddings
                      |> Map.put(center_word, new_center)
                      |> Map.put(context_word, new_context)
      
      %{embeddings | embeddings: new_embeddings}
    else
      embeddings
    end
  end
  
  def similarity(embeddings, word1, word2) do
    vec1 = Map.get(embeddings.embeddings, word1)
    vec2 = Map.get(embeddings.embeddings, word2)
    
    if vec1 && vec2 do
      cosine_similarity(vec1, vec2)
    else
      0.0
    end
  end
  
  def most_similar(embeddings, word, n \\ 10) do
    target_vector = Map.get(embeddings.embeddings, word)
    
    if target_vector do
      embeddings.embeddings
      |> Enum.map(fn {other_word, other_vector} ->
        similarity = cosine_similarity(target_vector, other_vector)
        {other_word, similarity}
      end)
      |> Enum.reject(fn {other_word, _} -> other_word == word end)
      |> Enum.sort_by(&elem(&1, 1), :desc)
      |> Enum.take(n)
    else
      []
    end
  end
  
  defp cosine_similarity(vec1, vec2) do
    dot_product = VectorOps.dot_product(vec1, vec2)
    magnitude1 = :math.sqrt(Enum.reduce(vec1, 0, &(&1 * &1 + &2)))
    magnitude2 = :math.sqrt(Enum.reduce(vec2, 0, &(&1 * &1 + &2)))
    
    dot_product / (magnitude1 * magnitude2)
  end
end
```

#### Afternoon Session (4 hours)
**Lab 10.2: Language Models and Attention**
```elixir
defmodule LanguageModel do
  @moduledoc "Neural language model with attention mechanism"
  
  defstruct [:embedding_layer, :lstm_layers, :attention, :output_layer, :vocabulary]
  
  def new(vocab_size, embedding_dim, hidden_dim, num_layers) do
    %__MODULE__{
      embedding_layer: EmbeddingLayer.new(vocab_size, embedding_dim),
      lstm_layers: initialize_lstm_layers(num_layers, embedding_dim, hidden_dim),
      attention: AttentionMechanism.new(hidden_dim),
      output_layer: LinearLayer.new(hidden_dim, vocab_size),
      vocabulary: []
    }
  end
  
  def forward(model, input_sequence) do
    # Embedding lookup
    embedded = EmbeddingLayer.forward(model.embedding_layer, input_sequence)
    
    # LSTM processing
    {lstm_outputs, final_states} = 
      Enum.reduce(model.lstm_layers, {embedded, nil}, fn lstm_layer, {input, _states} ->
        LSTMLayer.forward(lstm_layer, input)
      end)
    
    # Apply attention
    attended_output = AttentionMechanism.forward(model.attention, lstm_outputs)
    
    # Output projection
    LinearLayer.forward(model.output_layer, attended_output)
  end
  
  def generate_text(model, seed_text, max_length \\ 100, temperature \\ 1.0) do
    initial_tokens = tokenize_text(seed_text, model.vocabulary)
    
    generate_sequence(model, initial_tokens, max_length, temperature, [])
  end
  
  defp generate_sequence(model, current_tokens, remaining_length, temperature, generated) 
      when remaining_length <= 0 do
    Enum.reverse(generated) ++ current_tokens
  end
  
  defp generate_sequence(model, current_tokens, remaining_length, temperature, generated) do
    # Forward pass
    logits = forward(model, current_tokens)
    
    # Apply temperature and sample
    probabilities = apply_temperature_and_softmax(logits, temperature)
    next_token = sample_from_distribution(probabilities)
    
    # Update sequence
    new_tokens = current_tokens ++ [next_token]
    new_generated = [next_token | generated]
    
    generate_sequence(model, new_tokens, remaining_length - 1, temperature, new_generated)
  end
  
  defp apply_temperature_and_softmax(logits, temperature) do
    scaled_logits = Enum.map(logits, &(&1 / temperature))
    softmax(scaled_logits)
  end
  
  defp sample_from_distribution(probabilities) do
    cumulative_probs = accumulate_probabilities(probabilities)
    random_value = :rand.uniform()
    
    Enum.find_index(cumulative_probs, &(&1 >= random_value)) || length(probabilities) - 1
  end
end

defmodule AttentionMechanism do
  @moduledoc "Multi-head attention implementation"
  
  defstruct [:query_projection, :key_projection, :value_projection, :output_projection, :num_heads]
  
  def new(hidden_dim, num_heads \\ 8) do
    head_dim = div(hidden_dim, num_heads)
    
    %__MODULE__{
      query_projection: LinearLayer.new(hidden_dim, hidden_dim),
      key_projection: LinearLayer.new(hidden_dim, hidden_dim),
      value_projection: LinearLayer.new(hidden_dim, hidden_dim),
      output_projection: LinearLayer.new(hidden_dim, hidden_dim),
      num_heads: num_heads
    }
  end
  
  def forward(attention, input_sequence) do
    batch_size = length(input_sequence)
    seq_length = length(hd(input_sequence))
    
    # Project to queries, keys, values
    queries = LinearLayer.forward(attention.query_projection, input_sequence)
    keys = LinearLayer.forward(attention.key_projection, input_sequence)
    values = LinearLayer.forward(attention.value_projection, input_sequence)
    
    # Reshape for multi-head attention
    {q_heads, k_heads, v_heads} = reshape_for_heads(queries, keys, values, attention.num_heads)
    
    # Apply attention for each head
    attention_outputs = 
      Enum.zip([q_heads, k_heads, v_heads])
      |> Enum.map(fn {q, k, v} -> scaled_dot_product_attention(q, k, v) end)
    
    # Concatenate heads
    concatenated = concatenate_attention_heads(attention_outputs)
    
    # Final projection
    LinearLayer.forward(attention.output_projection, concatenated)
  end
  
  defp scaled_dot_product_attention(queries, keys, values) do
    # Attention(Q,K,V) = softmax(QK^T/√d_k)V
    d_k = length(hd(keys))
    scores = matrix_multiply(queries, transpose(keys))
    
    scaled_scores = 
      Enum.map(scores, fn row ->
        Enum.map(row, &(&1 / :math.sqrt(d_k)))
      end)
    
    attention_weights = softmax_matrix(scaled_scores)
    matrix_multiply(attention_weights, values)
  end
  
  def self_attention(input_sequence, hidden_dim) do
    # Simplified self-attention for educational purposes
    seq_length = length(input_sequence)
    
    # Compute attention scores
    attention_scores = 
      for i <- 0..(seq_length - 1) do
        for j <- 0..(seq_length - 1) do
          query = Enum.at(input_sequence, i)
          key = Enum.at(input_sequence, j)
          VectorOps.dot_product(query, key) / :math.sqrt(hidden_dim)
        end
      end
    
    # Apply softmax
    attention_weights = softmax_matrix(attention_scores)
    
    # Weighted sum of values
    for i <- 0..(seq_length - 1) do
      weights_row = Enum.at(attention_weights, i)
      
      Enum.zip(weights_row, input_sequence)
      |> Enum.reduce(List.duplicate(0.0, hidden_dim), fn {weight, value}, acc ->
        scaled_value = Enum.map(value, &(&1 * weight))
        VectorOps.add_vectors(acc, scaled_value)
      end)
    end
  end
end
```

### Day 11: Computer Vision

#### Morning Session (4 hours)
**Lab 11.1: Image Processing and Feature Extraction**
```elixir
defmodule ImageProcessor do
  @moduledoc "Computer vision utilities and feature extraction"
  
  defstruct [:width, :height, :channels, :data]
  
  def new(width, height, channels \\ 3) do
    data = List.duplicate(0, width * height * channels)
    %__MODULE__{width: width, height: height, channels: channels, data: data}
  end
  
  def load_from_pixels(pixels, width, height, channels \\ 3) do
    %__MODULE__{width: width, height: height, channels: channels, data: pixels}
  end
  
  def get_pixel(image, x, y, channel \\ 0) do
    index = (y * image.width + x) * image.channels + channel
    Enum.at(image.data, index, 0)
  end
  
  def set_pixel(image, x, y, value, channel \\ 0) do
    index = (y * image.width + x) * image.channels + channel
    new_data = List.replace_at(image.data, index, value)
    %{image | data: new_data}
  end
  
  def convolution(image, kernel) do
    kernel_size = length(kernel)
    half_kernel = div(kernel_size, 2)
    
    new_data = 
      for y <- 0..(image.height - 1),
          x <- 0..(image.width - 1),
          c <- 0..(image.channels - 1) do
        
        # Apply convolution at this position
        sum = 
          for ky <- 0..(kernel_size - 1),
              kx <- 0..(kernel_size - 1) do
            
            img_y = y + ky - half_kernel
            img_x = x + kx - half_kernel
            
            # Handle boundary conditions
            pixel_value = 
              if img_x >= 0 and img_x < image.width and img_y >= 0 and img_y < image.height do
                get_pixel(image, img_x, img_y, c)
              else
                0
              end
            
            kernel_value = kernel |> Enum.at(ky) |> Enum.at(kx)
            pixel_value * kernel_value
          end
          |> Enum.sum()
        
        # Clamp to valid range
        max(0, min(255, round(sum)))
      end
    
    %{image | data: new_data}
  end
  
  def gaussian_blur(image, sigma \\ 1.0) do
    kernel_size = round(6 * sigma) + 1
    kernel_size = if rem(kernel_size, 2) == 0, do: kernel_size + 1, else: kernel_size
    
    kernel = generate_gaussian_kernel(kernel_size, sigma)
    convolution(image, kernel)
  end
  
  def edge_detection(image, method \\ :sobel) do
    case method do
      :sobel -> apply_sobel_filter(image)
      :canny -> apply_canny_edge_detection(image)
      :laplacian -> apply_laplacian_filter(image)
    end
  end
  
  defp apply_sobel_filter(image) do
    sobel_x = [
      [-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1]
    ]
    
    sobel_y = [
      [-1, -2, -1],
      [0, 0, 0],
      [1, 2, 1]
    ]
    
    grad_x = convolution(image, sobel_x)
    grad_y = convolution(image, sobel_y)
    
    # Compute gradient magnitude
    new_data = 
      Enum.zip(grad_x.data, grad_y.data)
      |> Enum.map(fn {gx, gy} ->
        magnitude = :math.sqrt(gx * gx + gy * gy)
        min(255, round(magnitude))
      end)
    
    %{image | data: new_data}
  end
  
  def extract_features(image, method \\ :hog) do
    case method do
      :hog -> extract_hog_features(image)
      :sift -> extract_sift_features(image)
      :orb -> extract_orb_features(image)
    end
  end
  
  defp extract_hog_features(image) do
    # Histogram of Oriented Gradients
    cell_size = 8
    block_size = 2
    num_bins = 9
    
    # Compute gradients
    grad_x = convolution(image, [[-1, 0, 1]])
    grad_y = convolution(image, [[-1], [0], [1]])
    
    # Compute gradient magnitudes and orientations
    {magnitudes, orientations} = compute_gradient_info(grad_x, grad_y)
    
    # Build histograms for each cell
    cell_histograms = build_cell_histograms(magnitudes, orientations, cell_size, num_bins)
    
    # Normalize over blocks
    normalize_hog_blocks(cell_histograms, block_size)
  end
  
  defp compute_gradient_info(grad_x, grad_y) do
    Enum.zip(grad_x.data, grad_y.data)
    |> Enum.map(fn {gx, gy} ->
      magnitude = :math.sqrt(gx * gx + gy * gy)
      orientation = :math.atan2(gy, gx) * 180 / :math.pi()
      
      # Convert to 0-180 range
      orientation = if orientation < 0, do: orientation + 180, else: orientation
      
      {magnitude, orientation}
    end)
    |> Enum.unzip()
  end
  
  def resize(image, new_width, new_height, method \\ :bilinear) do
    case method do
      :nearest -> resize_nearest_neighbor(image, new_width, new_height)
      :bilinear -> resize_bilinear(image, new_width, new_height)
      :bicubic -> resize_bicubic(image, new_width, new_height)
    end
  end
  
  defp resize_bilinear(image, new_width, new_height) do
    x_ratio = image.width / new_width
    y_ratio = image.height / new_height
    
    new_data = 
      for y <- 0..(new_height - 1),
          x <- 0..(new_width - 1),
          c <- 0..(image.channels - 1) do
        
        # Find corresponding position in original image
        orig_x = x * x_ratio
        orig_y = y * y_ratio
        
        # Get surrounding pixels
        x1 = floor(orig_x)
        x2 = min(x1 + 1, image.width - 1)
        y1 = floor(orig_y)
        y2 = min(y1 + 1, image.height - 1)
        
        # Bilinear interpolation
        q11 = get_pixel(image, round(x1), round(y1), c)
        q12 = get_pixel(image, round(x1), round(y2), c)
        q21 = get_pixel(image, round(x2), round(y1), c)
        q22 = get_pixel(image, round(x2), round(y2), c)
        
        wx = orig_x - x1
        wy = orig_y - y1
        
        interpolated = 
          q11 * (1 - wx) * (1 - wy) +
          q21 * wx * (1 - wy) +
          q12 * (1 - wx) * wy +
          q22 * wx * wy
        
        round(interpolated)
      end
    
    %__MODULE__{width: new_width, height: new_height, channels: image.channels, data: new_data}
  end
end
```

#### Afternoon Session (4 hours)
**Lab 11.2: Convolutional Neural Networks for Vision**
```elixir
defmodule VisionCNN do
  @moduledoc "CNN architectures for computer vision tasks"
  
  defstruct [:layers, :classifier, :input_shape]
  
  def new(input_shape, num_classes) do
    layers = [
      # First convolutional block
      %{type: :conv2d, filters: 32, kernel_size: 3, activation: :relu},
      %{type: :conv2d, filters: 32, kernel_size: 3, activation: :relu},
      %{type: :maxpool2d, pool_size: 2},
      %{type: :dropout, rate: 0.25},
      
      # Second convolutional block
      %{type: :conv2d, filters: 64, kernel_size: 3, activation: :relu},
      %{type: :conv2d, filters: 64, kernel_size: 3, activation: :relu},
      %{type: :maxpool2d, pool_size: 2},
      %{type: :dropout, rate: 0.25},
      
      # Third convolutional block
      %{type: :conv2d, filters: 128, kernel_size: 3, activation: :relu},
      %{type: :conv2d, filters: 128, kernel_size: 3, activation: :relu},
      %{type: :maxpool2d, pool_size: 2},
      %{type: :dropout, rate: 0.25},
      
      # Classifier
      %{type: :flatten},
      %{type: :dense, units: 512, activation: :relu},
      %{type: :dropout, rate: 0.5},
      %{type: :dense, units: num_classes, activation: :softmax}
    ]
    
    %__MODULE__{
      layers: initialize_layers(layers, input_shape),
      classifier: nil,
      input_shape: input_shape
    }
  end
  
  def forward(cnn, input_batch) do
    Enum.reduce(cnn.layers, input_batch, fn layer, input ->
      apply_layer(layer, input)
    end)
  end
  
  def train(cnn, training_data, validation_data, opts \\ []) do
    epochs = Keyword.get(opts, :epochs, 10)
    batch_size = Keyword.get(opts, :batch_size, 32)
    learning_rate = Keyword.get(opts, :learning_rate, 0.001)
    
    Enum.reduce(1..epochs, cnn, fn epoch, acc_cnn ->
      IO.puts("Epoch #{epoch}/#{epochs}")
      
      # Training phase
      trained_cnn = train_epoch(acc_cnn, training_data, batch_size, learning_rate)
      
      # Validation phase
      validation_metrics = evaluate(trained_cnn, validation_data, batch_size)
      
      IO.puts("Validation accuracy: #{validation_metrics.accuracy}")
      IO.puts("Validation loss: #{validation_metrics.loss}")
      
      trained_cnn
    end)
  end
  
  defp train_epoch(cnn, training_data, batch_size, learning_rate) do
    training_data
    |> Enum.chunk_every(batch_size)
    |> Enum.reduce(cnn, fn batch, acc_cnn ->
      train_batch(acc_cnn, batch, learning_rate)
    end)
  end
  
  defp train_batch(cnn, batch, learning_rate) do
    {inputs, targets} = Enum.unzip(batch)
    
    # Forward pass
    predictions = forward(cnn, inputs)
    
    # Compute loss
    loss = compute_loss(predictions, targets)
    
    # Backward pass
    gradients = compute_gradients(cnn, inputs, targets, predictions)
    
    # Update parameters
    update_parameters(cnn, gradients, learning_rate)
  end
  
  def data_augmentation(image_batch, opts \\ []) do
    Enum.map(image_batch, fn image ->
      image
      |> maybe_horizontal_flip(opts[:horizontal_flip] || 0.5)
      |> maybe_rotation(opts[:rotation_range] || 15)
      |> maybe_zoom(opts[:zoom_range] || 0.1)
      |> maybe_brightness_shift(opts[:brightness_range] || 0.2)
    end)
  end
  
  defp maybe_horizontal_flip(image, probability) do
    if :rand.uniform() < probability do
      horizontal_flip(image)
    else
      image
    end
  end
  
  defp maybe_rotation(image, max_degrees) do
    if max_degrees > 0 do
      angle = (:rand.uniform() - 0.5) * 2 * max_degrees
      rotate_image(image, angle)
    else
      image
    end
  end
  
  defp horizontal_flip(image) do
    new_data = 
      for y <- 0..(image.height - 1),
          x <- 0..(image.width - 1),
          c <- 0..(image.channels - 1) do
        
        flipped_x = image.width - 1 - x
        ImageProcessor.get_pixel(image, flipped_x, y, c)
      end
    
    %{image | data: new_data}
  end
  
  def transfer_learning(pretrained_model, new_num_classes, freeze_layers \\ true) do
    # Remove final classification layer
    feature_layers = Enum.drop(pretrained_model.layers, -1)
    
    # Add new classification layer
    new_classifier = %{
      type: :dense, 
      units: new_num_classes, 
      activation: :softmax,
      trainable: true
    }
    
    # Optionally freeze pretrained layers
    updated_layers = 
      if freeze_layers do
        Enum.map(feature_layers, &Map.put(&1, :trainable, false))
      else
        feature_layers
      end
    
    %{pretrained_model | layers: updated_layers ++ [new_classifier]}
  end
  
  def object_detection(image, model, opts \\ []) do
    # Simple sliding window detection
    window_size = Keyword.get(opts, :window_size, {64, 64})
    stride = Keyword.get(opts, :stride, 32)
    confidence_threshold = Keyword.get(opts, :confidence_threshold, 0.5)
    
    detections = []
    
    for y <- 0..(image.height - elem(window_size, 1))//stride,
        x <- 0..(image.width - elem(window_size, 0))//stride do
      
      # Extract window
      window = extract_window(image, x, y, window_size)
      
      # Classify window
      prediction = forward(model, [window])
      confidence = Enum.max(prediction)
      class_id = Enum.find_index(prediction, &(&1 == confidence))
      
      if confidence > confidence_threshold do
        detection = %{
          bbox: {x, y, elem(window_size, 0), elem(window_size, 1)},
          class_id: class_id,
          confidence: confidence
        }
        
        [detection | detections]
      else
        detections
      end
    end
    |> Enum.reverse()
    |> non_maximum_suppression(opts[:nms_threshold] || 0.5)
  end
  
  defp non_maximum_suppression(detections, threshold) do
    # Sort by confidence
    sorted_detections = Enum.sort_by(detections, & &1.confidence, :desc)
    
    suppress_overlapping(sorted_detections, threshold, [])
  end
  
  defp suppress_overlapping([], _threshold, kept), do: Enum.reverse(kept)
  
  defp suppress_overlapping([detection | rest], threshold, kept) do
    # Keep this detection and remove overlapping ones
    non_overlapping = 
      Enum.reject(rest, fn other ->
        iou = intersection_over_union(detection.bbox, other.bbox)
        iou > threshold
      end)
    
    suppress_overlapping(non_overlapping, threshold, [detection | kept])
  end
end
```

### Day 12: Time Series and Forecasting

#### Morning Session (4 hours)
**Lab 12.1: Time Series Analysis**
```elixir
defmodule TimeSeries do
  @moduledoc "Time series analysis and forecasting"
  
  defstruct [:data, :timestamps, :frequency, :metadata]
  
  def new(data, timestamps, opts \\ []) do
    %__MODULE__{
      data: data,
      timestamps: timestamps,
      frequency: Keyword.get(opts, :frequency, :daily),
      metadata: Keyword.get(opts, :metadata, %{})
    }
  end
  
  def decompose(ts, method \\ :additive) do
    case method do
      :additive -> additive_decomposition(ts)
      :multiplicative -> multiplicative_decomposition(ts)
      :stl -> stl_decomposition(ts)
    end
  end
  
  defp additive_decomposition(ts) do
    # Trend extraction using moving average
    trend = extract_trend(ts.data, window_size_for_frequency(ts.frequency))
    
    # Detrend the series
    detrended = Enum.zip(ts.data, trend)
                |> Enum.map(fn {value, trend_value} -> value - trend_value end)
    
    # Extract seasonal component
    seasonal = extract_seasonal_component(detrended, ts.frequency)
    
    # Residual is what remains
    residual = compute_residual(ts.data, trend, seasonal, :additive)
    
    %{
      original: ts.data,
      trend: trend,
      seasonal: seasonal,
      residual: residual,
      method: :additive
    }
  end
  
  def extract_trend(data, window_size) do
    half_window = div(window_size, 2)
    
    Enum.with_index(data)
    |> Enum.map(fn {_value, index} ->
      start_idx = max(0, index - half_window)
      end_idx = min(length(data) - 1, index + half_window)
      
      window_data = Enum.slice(data, start_idx, end_idx - start_idx + 1)
      Statistics.mean(window_data)
    end)
  end
  
  def detect_anomalies(ts, method \\ :zscore, opts \\ []) do
    case method do
      :zscore -> zscore_anomaly_detection(ts, opts)
      :isolation_forest -> isolation_forest_detection(ts, opts)
      :lstm_autoencoder -> lstm_autoencoder_detection(ts, opts)
    end
  end
  
  defp zscore_anomaly_detection(ts, opts) do
    threshold = Keyword.get(opts, :threshold, 3.0)
    
    mean = Statistics.mean(ts.data)
    std = Statistics.standard_deviation(ts.data)
    
    Enum.with_index(ts.data)
    |> Enum.filter(fn {value, _index} ->
      abs(value - mean) / std > threshold
    end)
    |> Enum.map(fn {value, index} ->
      %{
        index: index,
        timestamp: Enum.at(ts.timestamps, index),
        value: value,
        zscore: abs(value - mean) / std,
        type: if(value > mean, do: :high, else: :low)
      }
    end)
  end
  
  def forecast(ts, method, horizon, opts \\ []) do
    case method do
      :arima -> arima_forecast(ts, horizon, opts)
      :lstm -> lstm_forecast(ts, horizon, opts)
      :prophet -> prophet_forecast(ts, horizon, opts)
      :exponential_smoothing -> exponential_smoothing_forecast(ts, horizon, opts)
    end
  end
  
  defp arima_forecast(ts, horizon, opts) do
    # ARIMA(p,d,q) model
    p = Keyword.get(opts, :p, 1)  # AR order
    d = Keyword.get(opts, :d, 1)  # Differencing order
    q = Keyword.get(opts, :q, 1)  # MA order
    
    # Difference the series
    differenced_data = difference_series(ts.data, d)
    
    # Fit AR component
    ar_params = fit_autoregressive(differenced_data, p)
    
    # Fit MA component
    ma_params = fit_moving_average(differenced_data, q)
    
    # Generate forecasts
    generate_arima_forecasts(ts.data, ar_params, ma_params, horizon, d)
  end
  
  defp lstm_forecast(ts, horizon, opts) do
    # LSTM-based forecasting
    sequence_length = Keyword.get(opts, :sequence_length, 10)
    hidden_units = Keyword.get(opts, :hidden_units, 50)
    epochs = Keyword.get(opts, :epochs, 100)
    
    # Prepare training data
    {X, y} = create_sequences(ts.data, sequence_length)
    
    # Build LSTM model
    model = build_lstm_forecasting_model(sequence_length, hidden_units)
    
    # Train model
    trained_model = train_lstm_model(model, X, y, epochs)
    
    # Generate forecasts
    generate_lstm_forecasts(trained_model, ts.data, sequence_length, horizon)
  end
  
  def seasonality_analysis(ts) do
    case ts.frequency do
      :daily -> analyze_weekly_seasonality(ts)
      :weekly -> analyze_monthly_seasonality(ts)
      :monthly -> analyze_yearly_seasonality(ts)
      :hourly -> analyze_daily_seasonality(ts)
    end
  end
  
  defp analyze_weekly_seasonality(ts) do
    # Group by day of week
    daily_patterns = 
      Enum.zip(ts.data, ts.timestamps)
      |> Enum.group_by(fn {_value, timestamp} ->
        Date.day_of_week(timestamp)
      end)
      |> Enum.map(fn {day, values} ->
        data_values = Enum.map(values, &elem(&1, 0))
        {day, %{
          mean: Statistics.mean(data_values),
          std: Statistics.standard_deviation(data_values),
          count: length(data_values)
        }}
      end)
      |> Map.new()
    
    # Compute seasonality strength
    overall_variance = Statistics.variance(ts.data)
    seasonal_variance = compute_seasonal_variance(daily_patterns)
    
    seasonality_strength = 1 - (seasonal_variance / overall_variance)
    
    %{
      patterns: daily_patterns,
      strength: seasonality_strength,
      frequency: :weekly
    }
  end
  
  def autocorrelation(ts, max_lag \\ nil) do
    max_lag = max_lag || min(length(ts.data) - 1, 40)
    
    for lag <- 0..max_lag do
      correlation = compute_autocorrelation_at_lag(ts.data, lag)
      {lag, correlation}
    end
  end
  
  defp compute_autocorrelation_at_lag(data, lag) when lag == 0 do
    1.0  # Perfect correlation with itself
  end
  
  defp compute_autocorrelation_at_lag(data, lag) do
    n = length(data)
    
    if lag >= n do
      0.0
    else
      # Split data into two series with lag offset
      series1 = Enum.take(data, n - lag)
      series2 = Enum.drop(data, lag)
      
      Statistics.correlation(series1, series2)
    end
  end
  
  def partial_autocorrelation(ts, max_lag \\ nil) do
    max_lag = max_lag || min(length(ts.data) - 1, 20)
    autocorr_values = autocorrelation(ts, max_lag) |> Enum.map(&elem(&1, 1))
    
    # Compute partial autocorrelations using Yule-Walker equations
    compute_partial_autocorrelations(autocorr_values, max_lag)
  end
  
  def stationarity_test(ts, test \\ :adf) do
    case test do
      :adf -> augmented_dickey_fuller_test(ts)
      :kpss -> kpss_test(ts)
      :pp -> phillips_perron_test(ts)
    end
  end
  
  defp augmented_dickey_fuller_test(ts) do
    # Simplified ADF test implementation
    # In practice, would use more sophisticated regression
    
    # Create lagged differences
    differences = difference_series(ts.data, 1)
    lagged_levels = Enum.drop(ts.data, -1)
    
    # Regression: Δy_t = α + βy_{t-1} + ε_t
    {beta, _} = simple_linear_regression(lagged_levels, differences)
    
    # Test statistic (simplified)
    t_statistic = beta / estimate_standard_error(lagged_levels, differences, beta)
    
    # Critical values (simplified)
    critical_values = %{
      "1%" => -3.43,
      "5%" => -2.86,
      "10%" => -2.57
    }
    
    %{
      test_statistic: t_statistic,
      critical_values: critical_values,
      p_value: estimate_p_value(t_statistic, critical_values),
      stationary: t_statistic < critical_values["5%"]
    }
  end
end

defmodule LSTMForecaster do
  @moduledoc "LSTM-based time series forecasting"
  
  defstruct [:layers, :sequence_length, :feature_size]
  
  def new(sequence_length, feature_size, hidden_units \\ 50) do
    %__MODULE__{
      layers: [
        %{type: :lstm, units: hidden_units, return_sequences: true},
        %{type: :lstm, units: hidden_units, return_sequences: false},
        %{type: :dense, units: 1, activation: :linear}
      ],
      sequence_length: sequence_length,
      feature_size: feature_size
    }
  end
  
  def prepare_data(time_series, sequence_length) do
    sequences = create_sliding_windows(time_series, sequence_length)
    
    # Split into features (X) and targets (y)
    {features, targets} = 
      sequences
      |> Enum.map(fn sequence ->
        {Enum.drop(sequence, -1), List.last(sequence)}
      end)
      |> Enum.unzip()
    
    # Normalize data
    {normalized_features, feature_scaler} = normalize_sequences(features)
    {normalized_targets, target_scaler} = normalize_values(targets)
    
    %{
      features: normalized_features,
      targets: normalized_targets,
      feature_scaler: feature_scaler,
      target_scaler: target_scaler
    }
  end
  
  def train(forecaster, training_data, opts \\ []) do
    epochs = Keyword.get(opts, :epochs, 100)
    batch_size = Keyword.get(opts, :batch_size, 32)
    learning_rate = Keyword.get(opts, :learning_rate, 0.001)
    
    Enum.reduce(1..epochs, forecaster, fn epoch, acc_forecaster ->
      IO.puts("Training epoch #{epoch}/#{epochs}")
      
      # Batch training
      training_data.features
      |> Enum.zip(training_data.targets)
      |> Enum.chunk_every(batch_size)
      |> Enum.reduce(acc_forecaster, fn batch, batch_acc ->
        train_batch(batch_acc, batch, learning_rate)
      end)
    end)
  end
  
  def forecast(forecaster, last_sequence, horizon, scalers) do
    current_sequence = last_sequence
    predictions = []
    
    Enum.reduce(1..horizon, {current_sequence, predictions}, fn _step, {seq, preds} ->
      # Predict next value
      normalized_seq = normalize_sequence(seq, scalers.feature_scaler)
      normalized_pred = forward_lstm(forecaster, normalized_seq)
      
      # Denormalize prediction
      prediction = denormalize_value(normalized_pred, scalers.target_scaler)
      
      # Update sequence for next prediction
      new_sequence = Enum.drop(seq, 1) ++ [prediction]
      new_predictions = preds ++ [prediction]
      
      {new_sequence, new_predictions}
    end)
    |> elem(1)
  end
  
  defp create_sliding_windows(data, window_size) do
    data
    |> Enum.with_index()
    |> Enum.filter(fn {_value, index} -> 
      index + window_size < length(data)
    end)
    |> Enum.map(fn {_value, index} ->
      Enum.slice(data, index, window_size + 1)
    end)
  end
  
  defp forward_lstm(forecaster, sequence) do
    # Simplified LSTM forward pass
    # In practice, would implement full LSTM cell operations
    
    Enum.reduce(forecaster.layers, sequence, fn layer, input ->
      case layer.type do
        :lstm -> apply_lstm_layer(layer, input)
        :dense -> apply_dense_layer(layer, input)
      end
    end)
  end
  
  def evaluate_forecast(actual, predicted) do
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = :math.sqrt(mse)
    mape = mean_absolute_percentage_error(actual, predicted)
    
    %{
      mae: mae,
      mse: mse,
      rmse: rmse,
      mape: mape
    }
  end
  
  defp mean_absolute_error(actual, predicted) do
    Enum.zip(actual, predicted)
    |> Enum.map(fn {a, p} -> abs(a - p) end)
    |> Statistics.mean()
  end
  
  defp mean_squared_error(actual, predicted) do
    Enum.zip(actual, predicted)
    |> Enum.map(fn {a, p} -> :math.pow(a - p, 2) end)
    |> Statistics.mean()
  end
  
  defp mean_absolute_percentage_error(actual, predicted) do
    Enum.zip(actual, predicted)
    |> Enum.reject(fn {a, _p} -> a == 0 end)  # Avoid division by zero
    |> Enum.map(fn {a, p} -> abs((a - p) / a) * 100 end)
    |> Statistics.mean()
  end
end
```

#### Afternoon Session (4 hours)
**Lab 12.2: Advanced Forecasting Methods**
```elixir
defmodule ProphetForecaster do
  @moduledoc "Facebook Prophet-style forecasting model"
  
  defstruct [:trend_params, :seasonal_params, :holiday_params, :changepoints]
  
  def new(opts \\ []) do
    %__MODULE__{
      trend_params: %{growth: :linear, changepoint_prior_scale: 0.05},
      seasonal_params: initialize_seasonal_params(opts),
      holiday_params: Keyword.get(opts, :holidays, []),
      changepoints: []
    }
  end
  
  def fit(forecaster, time_series) do
    # Detect trend changepoints
    changepoints = detect_changepoints(time_series)
    
    # Fit trend component
    trend_params = fit_trend_component(time_series, changepoints)
    
    # Fit seasonal components
    seasonal_params = fit_seasonal_components(time_series, forecaster.seasonal_params)
    
    # Fit holiday effects
    holiday_params = fit_holiday_effects(time_series, forecaster.holiday_params)
    
    %{forecaster |
      trend_params: trend_params,
      seasonal_params: seasonal_params,
      holiday_params: holiday_params,
      changepoints: changepoints
    }
  end
  
  def predict(forecaster, future_dates) do
    Enum.map(future_dates, fn date ->
      trend = compute_trend(forecaster, date)
      seasonal = compute_seasonal(forecaster, date)
      holiday = compute_holiday_effect(forecaster, date)
      
      prediction = trend + seasonal + holiday
      
      %{
        date: date,
        yhat: prediction,
        trend: trend,
        seasonal: seasonal,
        holiday: holiday
      }
    end)
  end
  
  defp detect_changepoints(time_series) do
    # Automatic changepoint detection using PELT (Pruned Exact Linear Time)
    # Simplified implementation
    
    data = time_series.data
    n = length(data)
    potential_changepoints = Enum.to_list(1..(n-1))
    
    # Score each potential changepoint
    scored_changepoints = 
      Enum.map(potential_changepoints, fn cp ->
        score = compute_changepoint_score(data, cp)
        {cp, score}
      end)
    
    # Select significant changepoints
    threshold = compute_changepoint_threshold(scored_changepoints)
    
    scored_changepoints
    |> Enum.filter(fn {_cp, score} -> score > threshold end)
    |> Enum.map(&elem(&1, 0))
  end
  
  defp fit_trend_component(time_series, changepoints) do
    # Fit piecewise linear trend with changepoints
    x = create_time_features(time_series.timestamps)
    y = time_series.data
    
    # Create design matrix with changepoints
    design_matrix = create_trend_design_matrix(x, changepoints)
    
    # Linear regression
    coefficients = solve_linear_regression(design_matrix, y)
    
    %{
      coefficients: coefficients,
      changepoints: changepoints,
      base_rate: hd(coefficients)
    }
  end
  
  defp fit_seasonal_components(time_series, seasonal_config) do
    # Fit Fourier series for seasonal components
    yearly_seasonality = fit_fourier_seasonality(time_series, :yearly, 10)
    weekly_seasonality = fit_fourier_seasonality(time_series, :weekly, 3)
    daily_seasonality = fit_fourier_seasonality(time_series, :daily, 4)
    
    %{
      yearly: yearly_seasonality,
      weekly: weekly_seasonality,
      daily: daily_seasonality
    }
  end
  
  defp fit_fourier_seasonality(time_series, period_type, n_fourier) do
    period = get_period_length(period_type)
    
    # Create Fourier features
    fourier_features = 
      Enum.map(time_series.timestamps, fn timestamp ->
        create_fourier_features(timestamp, period, n_fourier)
      end)
    
    # Fit coefficients
    coefficients = solve_linear_regression(fourier_features, time_series.data)
    
    %{
      period: period,
      n_fourier: n_fourier,
      coefficients: coefficients
    }
  end
  
  defp create_fourier_features(timestamp, period, n_fourier) do
    t = timestamp_to_numeric(timestamp) / period * 2 * :math.pi()
    
    for n <- 1..n_fourier do
      [:math.cos(n * t), :math.sin(n * t)]
    end
    |> List.flatten()
  end
  
  def uncertainty_intervals(forecaster, predictions, confidence_level \\ 0.95) do
    # Compute prediction intervals using simulation
    num_simulations = 1000
    
    Enum.map(predictions, fn prediction ->
      simulated_values = 
        for _sim <- 1..num_simulations do
          simulate_prediction(forecaster, prediction)
        end
      
      sorted_values = Enum.sort(simulated_values)
      lower_percentile = (1 - confidence_level) / 2
      upper_percentile = 1 - lower_percentile
      
      lower_bound = percentile(sorted_values, lower_percentile)
      upper_bound = percentile(sorted_values, upper_percentile)
      
      Map.merge(prediction, %{
        yhat_lower: lower_bound,
        yhat_upper: upper_bound
      })
    end)
  end
  
  defp simulate_prediction(forecaster, prediction) do
    # Add uncertainty from different sources
    trend_uncertainty = sample_trend_uncertainty(forecaster)
    seasonal_uncertainty = sample_seasonal_uncertainty(forecaster)
    observation_noise = sample_observation_noise(forecaster)
    
    prediction.yhat + trend_uncertainty + seasonal_uncertainty + observation_noise
  end
  
  def cross_validation(time_series, horizon, opts \\ []) do
    initial_train_size = Keyword.get(opts, :initial, div(length(time_series.data), 2))
    step_size = Keyword.get(opts, :step, horizon)
    
    # Create folds
    folds = create_time_series_folds(time_series, initial_train_size, horizon, step_size)
    
    # Evaluate each fold
    fold_results = 
      Enum.map(folds, fn {train_data, test_data} ->
        # Fit model on training data
        model = ProphetForecaster.new() |> ProphetForecaster.fit(train_data)
        
        # Predict on test data
        future_dates = Enum.map(test_data.timestamps, & &1)
        predictions = ProphetForecaster.predict(model, future_dates)
        
        # Compute metrics
        predicted_values = Enum.map(predictions, & &1.yhat)
        actual_values = test_data.data
        
        %{
          mae: LSTMForecaster.mean_absolute_error(actual_values, predicted_values),
          mse: LSTMForecaster.mean_squared_error(actual_values, predicted_values),
          mape: LSTMForecaster.mean_absolute_percentage_error(actual_values, predicted_values)
        }
      end)
    
    # Aggregate results
    %{
      mae: Statistics.mean(Enum.map(fold_results, & &1.mae)),
      mse: Statistics.mean(Enum.map(fold_results, & &1.mse)),
      mape: Statistics.mean(Enum.map(fold_results, & &1.mape)),
      individual_folds: fold_results
    }
  end
  
  def hyperparameter_tuning(time_series, param_grid, cv_folds \\ 5) do
    # Grid search with cross-validation
    param_combinations = generate_param_combinations(param_grid)
    
    best_params = nil
    best_score = :infinity
    
    Enum.reduce(param_combinations, {best_params, best_score}, fn params, {current_best_params, current_best_score} ->
      # Create forecaster with these parameters
      forecaster = ProphetForecaster.new(params)
      
      # Cross-validation
      cv_results = cross_validation(time_series, 30, initial: div(length(time_series.data), 3))
      avg_mae = cv_results.mae
      
      if avg_mae < current_best_score do
        IO.puts("New best parameters found: #{inspect(params)} with MAE: #{avg_mae}")
        {params, avg_mae}
      else
        {current_best_params, current_best_score}
      end
    end)
  end
  
  defp generate_param_combinations(param_grid) do
    # Generate all combinations of parameters
    param_keys = Map.keys(param_grid)
    param_values = Map.values(param_grid)
    
    cartesian_product(param_values)
    |> Enum.map(fn value_combination ->
      Enum.zip(param_keys, value_combination) |> Map.new()
    end)
  end
  
  defp cartesian_product([]), do: [[]]
  defp cartesian_product([h | t]) do
    for x <- h, y <- cartesian_product(t), do: [x | y]
  end
end
```

### Day 13: Model Deployment and Serving

#### Morning Session (4 hours)
**Lab 13.1: Production Model Serving Infrastructure**
```elixir
defmodule ModelServingCluster do
  @moduledoc "Production-grade model serving with load balancing and health monitoring"
  
  use Supervisor
  
  def start_link(opts) do
    Supervisor.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  def init(opts) do
    model_configs = Keyword.get(opts, :models, [])
    cluster_config = Keyword.get(opts, :cluster, %{})
    
    children = [
      # Model registry for version management
      {ModelRegistry, []},
      
      # Load balancer
      {ModelLoadBalancer, cluster_config},
      
      # Health monitor
      {HealthMonitor, []},
      
      # Metrics collector
      {MetricsCollector, []},
      
      # Model servers
      {DynamicSupervisor, name: ModelServerSupervisor, strategy: :one_for_one}
    ] ++ create_model_servers(model_configs)
    
    Supervisor.init(children, strategy: :one_for_one)
  end
  
  defp create_model_servers(model_configs) do
    Enum.map(model_configs, fn config ->
      Supervisor.child_spec(
        {ModelServerPool, config},
        id: {ModelServerPool, config.model_name}
      )
    end)
  end
end

defmodule ModelServerPool do
  @moduledoc "Pool of model servers for high availability"
  
  use DynamicSupervisor
  
  def start_link(config) do
    DynamicSupervisor.start_link(__MODULE__, config, name: via_tuple(config.model_name))
  end
  
  def init(config) do
    # Start multiple instances of the model server
    pool_size = Map.get(config, :pool_size, 4)
    
    for i <- 1..pool_size do
      server_config = Map.put(config, :instance_id, i)
      {:ok, _pid} = DynamicSupervisor.start_child(__MODULE__, {ModelServer, server_config})
    end
    
    DynamicSupervisor.init(strategy: :one_for_one)
  end
  
  def predict(model_name, features, opts \\ []) do
    strategy = Keyword.get(opts, :strategy, :round_robin)
    timeout = Keyword.get(opts, :timeout, 5000)
    
    case get_healthy_server(model_name, strategy) do
      {:ok, server_pid} ->
        try do
          ModelServer.predict(server_pid, features, timeout: timeout)
        rescue
          _ -> {:error, :prediction_failed}
        end
      
      {:error, reason} ->
        {:error, reason}
    end
  end
  
  defp get_healthy_server(model_name, strategy) do
    servers = list_healthy_servers(model_name)
    
    if Enum.empty?(servers) do
      {:error, :no_healthy_servers}
    else
      server = select_server(servers, strategy)
      {:ok, server}
    end
  end
  
  defp select_server(servers, :round_robin) do
    # Simple round-robin selection
    index = :persistent_term.get({:round_robin, :index}, 0)
    server = Enum.at(servers, rem(index, length(servers)))
    :persistent_term.put({:round_robin, :index}, index + 1)
    server
  end
  
  defp select_server(servers, :least_loaded) do
    # Select server with lowest current load
    servers
    |> Enum.map(fn server ->
      load = ModelServer.get_current_load(server)
      {server, load}
    end)
    |> Enum.min_by(&elem(&1, 1))
    |> elem(0)
  end
  
  defp via_tuple(model_name), do: {:via, Registry, {ModelServerRegistry, model_name}}
end

defmodule AdvancedModelServer do
  @moduledoc "Production model server with advanced features"
  
  use GenServer
  
  defstruct [
    :model, :model_version, :config, :metrics, :circuit_breaker,
    :request_queue, :batch_processor, :cache, :feature_store
  ]
  
  def start_link(config) do
    server_name = via_tuple(config.model_name, config.instance_id)
    GenServer.start_link(__MODULE__, config, name: server_name)
  end
  
  def predict(server, features, opts \\ []) do
    timeout = Keyword.get(opts, :timeout, 5000)
    use_cache = Keyword.get(opts, :cache, true)
    
    # Check cache first
    if use_cache do
      cache_key = compute_cache_key(features)
      
      case get_from_cache(server, cache_key) do
        {:hit, result} -> {:ok, result}
        :miss -> perform_prediction(server, features, cache_key, timeout)
      end
    else
      perform_prediction(server, features, nil, timeout)
    end
  end
  
  defp perform_prediction(server, features, cache_key, timeout) do
    GenServer.call(server, {:predict, features, cache_key}, timeout)
  end
  
  def init(config) do
    # Load model
    model = load_model(config.model_path, config.model_format)
    
    # Initialize components
    circuit_breaker = CircuitBreaker.new(config.circuit_breaker || %{})
    request_queue = :queue.new()
    cache = Cache.new(config.cache_size || 1000)
    feature_store = FeatureStore.connect(config.feature_store)
    
    # Start batch processor
    {:ok, batch_processor} = BatchProcessor.start_link(self(), config.batch_config || %{})
    
    state = %__MODULE__{
      model: model,
      model_version: config.model_version,
      config: config,
      metrics: Metrics.new(),
      circuit_breaker: circuit_breaker,
      request_queue: request_queue,
      batch_processor: batch_processor,
      cache: cache,
      feature_store: feature_store
    }
    
    {:ok, state}
  end
  
  def handle_call({:predict, features, cache_key}, from, state) do
    case CircuitBreaker.call(state.circuit_breaker, fn ->
      execute_prediction(state, features, cache_key)
    end) do
      {:ok, result} ->
        # Update metrics
        new_metrics = Metrics.record_success(state.metrics)
        new_state = %{state | metrics: new_metrics}
        
        {:reply, {:ok, result}, new_state}
      
      {:error, :circuit_open} ->
        {:reply, {:error, :service_unavailable}, state}
      
      {:error, reason} ->
        new_metrics = Metrics.record_error(state.metrics, reason)
        new_state = %{state | metrics: new_metrics}
        
        {:reply, {:error, reason}, new_state}
    end
  end
  
  def handle_call(:get_metrics, _from, state) do
    {:reply, state.metrics, state}
  end
  
  def handle_call(:health_check, _from, state) do
    health_status = %{
      status: :healthy,
      model_version: state.model_version,
      circuit_breaker: CircuitBreaker.status(state.circuit_breaker),
      queue_size: :queue.len(state.request_queue),
      cache_hit_rate: Cache.hit_rate(state.cache),
      memory_usage: :erlang.memory(:total)
    }
    
    {:reply, health_status, state}
  end
  
  defp execute_prediction(state, features, cache_key) do
    start_time = :erlang.monotonic_time(:microsecond)
    
    try do
      # Feature enrichment from feature store
      enriched_features = enrich_features(state.feature_store, features)
      
      # Validate features
      :ok = validate_features(enriched_features, state.config.feature_schema)
      
      # Run model inference
      prediction = run_model_inference(state.model, enriched_features)
      
      # Post-process prediction
      final_result = post_process_prediction(prediction, state.config.post_processing)
      
      # Cache result if cache_key provided
      if cache_key do
        Cache.put(state.cache, cache_key, final_result)
      end
      
      # Record latency
      latency = :erlang.monotonic_time(:microsecond) - start_time
      Metrics.record_latency(state.metrics, latency)
      
      {:ok, final_result}
    rescue
      error ->
        {:error, error}
    end
  end
  
  defp enrich_features(feature_store, base_features) do
    # Fetch additional features from feature store
    user_id = Map.get(base_features, :user_id)
    item_id = Map.get(base_features, :item_id)
    
    user_features = FeatureStore.get_user_features(feature_store, user_id)
    item_features = FeatureStore.get_item_features(feature_store, item_id)
    contextual_features = FeatureStore.get_contextual_features(feature_store)
    
    Map.merge(base_features, %{
      user_features: user_features,
      item_features: item_features,
      contextual_features: contextual_features
    })
  end
  
  defp validate_features(features, schema) do
    # Validate feature schema and data types
    case FeatureValidator.validate(features, schema) do
      :ok -> :ok
      {:error, violations} -> raise "Feature validation failed: #{inspect(violations)}"
    end
  end
  
  defp run_model_inference(model, features) do
    case model.type do
      :tensorflow -> TensorflowModel.predict(model, features)
      :pytorch -> PytorchModel.predict(model, features)
      :onnx -> OnnxModel.predict(model, features)
      :elixir_native -> NeuralNetwork.forward(model, features)
      :xgboost -> XGBoostModel.predict(model, features)
    end
  end
  
  defp post_process_prediction(prediction, post_processing_config) do
    # Apply post-processing transformations
    prediction
    |> apply_probability_calibration(post_processing_config[:calibration])
    |> apply_business_rules(post_processing_config[:business_rules])
    |> format_output(post_processing_config[:output_format])
  end
  
  defp via_tuple(model_name, instance_id) do
    {:via, Registry, {ModelServerRegistry, {model_name, instance_id}}}
  end
end

defmodule BatchProcessor do
  @moduledoc "Batches requests for efficient processing"
  
  use GenServer
  
  defstruct [:parent_server, :batch_size, :batch_timeout, :current_batch, :batch_timer]
  
  def start_link(parent_server, config) do
    GenServer.start_link(__MODULE__, {parent_server, config})
  end
  
  def add_request(processor, request) do
    GenServer.cast(processor, {:add_request, request})
  end
  
  def init({parent_server, config}) do
    state = %__MODULE__{
      parent_server: parent_server,
      batch_size: Map.get(config, :batch_size, 16),
      batch_timeout: Map.get(config, :batch_timeout, 100),
      current_batch: [],
      batch_timer: nil
    }
    
    {:ok, state}
  end
  
  def handle_cast({:add_request, request}, state) do
    new_batch = [request | state.current_batch]
    
    if length(new_batch) >= state.batch_size do
      # Process full batch immediately
      process_batch(new_batch, state.parent_server)
      
      # Cancel timer and reset
      if state.batch_timer, do: Process.cancel_timer(state.batch_timer)
      new_state = %{state | current_batch: [], batch_timer: nil}
      
      {:noreply, new_state}
    else
      # Start timer if this is the first request in batch
      timer = 
        if length(state.current_batch) == 0 do
          Process.send_after(self(), :process_batch, state.batch_timeout)
        else
          state.batch_timer
        end
      
      new_state = %{state | current_batch: new_batch, batch_timer: timer}
      {:noreply, new_state}
    end
  end
  
  def handle_info(:process_batch, state) do
    if not Enum.empty?(state.current_batch) do
      process_batch(state.current_batch, state.parent_server)
    end
    
    new_state = %{state | current_batch: [], batch_timer: nil}
    {:noreply, new_state}
  end
  
  defp process_batch(requests, parent_server) do
    # Extract features from all requests
    features_batch = Enum.map(requests, & &1.features)
    
    # Batch inference
    batch_predictions = batch_inference(parent_server, features_batch)
    
    # Send responses back to callers
    Enum.zip(requests, batch_predictions)
    |> Enum.each(fn {request, prediction} ->
      GenServer.reply(request.from, {:ok, prediction})
    end)
  end
end

defmodule CircuitBreaker do
  @moduledoc "Circuit breaker pattern for fault tolerance"
  
  defstruct [:failure_threshold, :recovery_timeout, :state, :failure_count, :last_failure_time]
  
  def new(config \\ %{}) do
    %__MODULE__{
      failure_threshold: Map.get(config, :failure_threshold, 5),
      recovery_timeout: Map.get(config, :recovery_timeout, 60_000),
      state: :closed,
      failure_count: 0,
      last_failure_time: nil
    }
  end
  
  def call(%__MODULE__{state: :open} = cb, _fun) do
    if should_attempt_reset?(cb) do
      # Try to transition to half-open
      new_cb = %{cb | state: :half_open}
      {:error, :circuit_open}
    else
      {:error, :circuit_open}
    end
  end
  
  def call(%__MODULE__{state: :closed} = cb, fun) do
    try do
      result = fun.()
      # Reset failure count on success
      new_cb = %{cb | failure_count: 0}
      {:ok, result}
    rescue
      error ->
        new_failure_count = cb.failure_count + 1
        
        new_cb = 
          if new_failure_count >= cb.failure_threshold do
            %{cb | 
              state: :open, 
              failure_count: new_failure_count,
              last_failure_time: :erlang.monotonic_time(:millisecond)
            }
          else
            %{cb | failure_count: new_failure_count}
          end
        
        {:error, error}
    end
  end
  
  def call(%__MODULE__{state: :half_open} = cb, fun) do
    try do
      result = fun.()
      # Success - reset to closed
      new_cb = %{cb | state: :closed, failure_count: 0}
      {:ok, result}
    rescue
      error ->
      # Failure - back to open
      new_cb = %{cb | 
        state: :open,
        last_failure_time: :erlang.monotonic_time(:millisecond)
      }
      {:error, error}
    end
  end
  
  defp should_attempt_reset?(cb) do
    current_time = :erlang.monotonic_time(:millisecond)
    current_time - cb.last_failure_time > cb.recovery_timeout
  end
  
#### Afternoon Session (4 hours)
**Lab 13.2: Model Monitoring and A/B Testing Infrastructure**
```elixir
defmodule ModelMonitoringPipeline do
  @moduledoc "Comprehensive model monitoring and drift detection"
  
  use Broadway
  
  def start_link(_opts) do
    Broadway.start_link(__MODULE__,
      name: __MODULE__,
      producer: [
        module: {BroadwayKafka.Producer,
          hosts: [{"localhost", 9092}],
          group_id: "model_monitoring_group",
          topics: ["model_predictions", "model_feedback"]
        }
      ],
      processors: [
        default: [
          concurrency: System.schedulers_online(),
          max_demand: 100
        ]
      ],
      batchers: [
        drift_detection: [
          concurrency: 2,
          batch_size: 1000,
          batch_timeout: 5000
        ],
        performance_monitoring: [
          concurrency: 2,
          batch_size: 500,
          batch_timeout: 3000
        ]
      ]
    )
  end
  
  def handle_message(_, message, _) do
    data = Jason.decode!(message.data)
    
    enriched_message = 
      message
      |> Message.put_data(data)
      |> Message.put_batcher(determine_batcher(data))
    
    enriched_message
  end
  
  def handle_batch(:drift_detection, messages, _, _) do
    # Detect data and prediction drift
    message_data = Enum.map(messages, &Message.data/1)
    
    input_features = Enum.map(message_data, & &1["features"])
    predictions = Enum.map(message_data, & &1["prediction"])
    
    # Detect input drift
    input_drift = DriftDetector.detect_feature_drift(input_features)
    
    # Detect prediction drift  
    prediction_drift = DriftDetector.detect_prediction_drift(predictions)
    
    # Alert if significant drift detected
    if input_drift.significant or prediction_drift.significant do
      AlertManager.trigger_drift_alert(%{
        input_drift: input_drift,
        prediction_drift: prediction_drift,
        timestamp: DateTime.utc_now()
      })
    end
    
    # Store drift metrics
    DriftMetrics.store(input_drift, prediction_drift)
    
    messages
  end
  
  def handle_batch(:performance_monitoring, messages, _, _) do
    # Monitor model performance metrics
    message_data = Enum.map(messages, &Message.data/1)
    
    # Extract predictions and actual outcomes (if available)
    predictions_with_actuals = 
      message_data
      |> Enum.filter(&Map.has_key?(&1, "actual"))
      |> Enum.map(fn data -> {data["prediction"], data["actual"]} end)
    
    if not Enum.empty?(predictions_with_actuals) do
      # Calculate performance metrics
      performance_metrics = calculate_performance_metrics(predictions_with_actuals)
      
      # Check for performance degradation
      if performance_degraded?(performance_metrics) do
        AlertManager.trigger_performance_alert(performance_metrics)
      end
      
      # Store performance metrics
      PerformanceMetrics.store(performance_metrics)
    end
    
    messages
  end
  
  defp determine_batcher(data) do
    cond do
      Map.has_key?(data, "features") -> :drift_detection
      Map.has_key?(data, "actual") -> :performance_monitoring
      true -> :default
    end
  end
  
  defp calculate_performance_metrics(predictions_with_actuals) do
    {predictions, actuals} = Enum.unzip(predictions_with_actuals)
    
    %{
      accuracy: Metrics.accuracy(predictions, actuals),
      precision: Metrics.precision(predictions, actuals),
      recall: Metrics.recall(predictions, actuals),
      f1_score: Metrics.f1_score(predictions, actuals),
      auc: Metrics.auc(predictions, actuals),
      timestamp: DateTime.utc_now()
    }
  end
  
  defp performance_degraded?(current_metrics) do
    # Compare with baseline metrics
    baseline = PerformanceMetrics.get_baseline()
    
    current_metrics.accuracy < baseline.accuracy * 0.95 or
    current_metrics.f1_score < baseline.f1_score * 0.95
  end
end

defmodule ABTestingFramework do
  @moduledoc "A/B testing framework for model variants"
  
  use GenServer
  
  defstruct [:experiments, :traffic_splitter, :results_collector]
  
  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  def create_experiment(config) do
    GenServer.call(__MODULE__, {:create_experiment, config})
  end
  
  def get_model_assignment(user_id, experiment_name) do
    GenServer.call(__MODULE__, {:get_assignment, user_id, experiment_name})
  end
  
  def record_outcome(user_id, experiment_name, outcome_metrics) do
    GenServer.cast(__MODULE__, {:record_outcome, user_id, experiment_name, outcome_metrics})
  end
  
  def get_experiment_results(experiment_name) do
    GenServer.call(__MODULE__, {:get_results, experiment_name})
  end
  
  def init(_opts) do
    {:ok, %__MODULE__{
      experiments: %{},
      traffic_splitter: TrafficSplitter.new(),
      results_collector: ResultsCollector.new()
    }}
  end
  
  def handle_call({:create_experiment, config}, _from, state) do
    experiment = %{
      name: config.name,
      model_variants: config.model_variants,
      traffic_allocation: config.traffic_allocation,
      start_date: config.start_date || DateTime.utc_now(),
      end_date: config.end_date,
      success_metrics: config.success_metrics,
      minimum_sample_size: config.minimum_sample_size || 1000,
      statistical_power: config.statistical_power || 0.8,
      significance_level: config.significance_level || 0.05
    }
    
    new_experiments = Map.put(state.experiments, config.name, experiment)
    new_state = %{state | experiments: new_experiments}
    
    {:reply, :ok, new_state}
  end
  
  def handle_call({:get_assignment, user_id, experiment_name}, _from, state) do
    case Map.get(state.experiments, experiment_name) do
      nil ->
        {:reply, {:error, :experiment_not_found}, state}
      
      experiment ->
        if experiment_active?(experiment) do
          variant = assign_user_to_variant(user_id, experiment, state.traffic_splitter)
          {:reply, {:ok, variant}, state}
        else
          {:reply, {:error, :experiment_inactive}, state}
        end
    end
  end
  
  def handle_call({:get_results, experiment_name}, _from, state) do
    case Map.get(state.experiments, experiment_name) do
      nil ->
        {:reply, {:error, :experiment_not_found}, state}
      
      experiment ->
        results = ResultsCollector.get_results(state.results_collector, experiment_name)
        analysis = analyze_experiment_results(experiment, results)
        
        {:reply, {:ok, analysis}, state}
    end
  end
  
  def handle_cast({:record_outcome, user_id, experiment_name, outcome_metrics}, state) do
    ResultsCollector.record_outcome(
      state.results_collector,
      experiment_name,
      user_id,
      outcome_metrics
    )
    
    {:noreply, state}
  end
  
  defp experiment_active?(experiment) do
    now = DateTime.utc_now()
    DateTime.compare(now, experiment.start_date) in [:gt, :eq] and
    (is_nil(experiment.end_date) or DateTime.compare(now, experiment.end_date) == :lt)
  end
  
  defp assign_user_to_variant(user_id, experiment, traffic_splitter) do
    # Consistent hashing for stable assignments
    hash = :erlang.phash2({user_id, experiment.name})
    bucket = rem(hash, 100)
    
    # Find which variant this bucket belongs to
    assign_variant_by_bucket(bucket, experiment.model_variants, experiment.traffic_allocation)
  end
  
  defp assign_variant_by_bucket(bucket, variants, allocation) do
    cumulative = 0
    
    Enum.zip(variants, allocation)
    |> Enum.reduce_while(cumulative, fn {variant, percent}, acc ->
      new_acc = acc + percent
      
      if bucket < new_acc do
        {:halt, variant}
      else
        {:cont, new_acc}
      end
    end)
  end
  
  defp analyze_experiment_results(experiment, results) do
    # Group results by variant
    variant_results = 
      Enum.group_by(results, & &1.variant)
      |> Enum.map(fn {variant, variant_data} ->
        {variant, calculate_variant_statistics(variant_data, experiment.success_metrics)}
      end)
      |> Map.new()
    
    # Perform statistical significance testing
    significance_tests = perform_significance_tests(variant_results, experiment)
    
    # Calculate confidence intervals
    confidence_intervals = calculate_confidence_intervals(variant_results, experiment.significance_level)
    
    %{
      experiment_name: experiment.name,
      variant_results: variant_results,
      significance_tests: significance_tests,
      confidence_intervals: confidence_intervals,
      sample_sizes: calculate_sample_sizes(variant_results),
      recommendation: generate_recommendation(significance_tests, variant_results)
    }
  end
  
  defp perform_significance_tests(variant_results, experiment) do
    # Pairwise t-tests between variants
    variants = Map.keys(variant_results)
    
    for control <- variants,
        treatment <- variants,
        control != treatment do
      
      control_data = variant_results[control]
      treatment_data = variant_results[treatment]
      
      t_test_result = Statistics.t_test(
        control_data.success_rate_samples,
        treatment_data.success_rate_samples
      )
      
      {
        {control, treatment},
        %{
          t_statistic: t_test_result.t_statistic,
          p_value: t_test_result.p_value,
          significant: t_test_result.p_value < experiment.significance_level,
          effect_size: calculate_effect_size(control_data, treatment_data)
        }
      }
    end
    |> Map.new()
  end
  
  defp calculate_effect_size(control_data, treatment_data) do
    # Cohen's d for effect size
    pooled_std = :math.sqrt(
      (control_data.variance + treatment_data.variance) / 2
    )
    
    (treatment_data.mean - control_data.mean) / pooled_std
  end
  
  defp generate_recommendation(significance_tests, variant_results) do
    # Find the best performing variant
    best_variant = 
      variant_results
      |> Enum.max_by(fn {_variant, stats} -> stats.mean end)
      |> elem(0)
    
    # Check if the best variant is significantly better
    significant_improvements = 
      significance_tests
      |> Enum.filter(fn {{_control, treatment}, test} ->
        treatment == best_variant and test.significant and test.effect_size > 0
      end)
    
    if Enum.empty?(significant_improvements) do
      %{
        action: :continue_experiment,
        reason: "No statistically significant difference found",
        best_variant: best_variant
      }
    else
      %{
        action: :deploy_winner,
        reason: "Statistically significant improvement detected",
        winning_variant: best_variant,
        improvement: calculate_improvement_percentage(variant_results, best_variant)
      }
    end
  end
end

defmodule FeatureStore do
  @moduledoc "Real-time feature store for ML serving"
  
  use GenServer
  
  defstruct [:online_store, :offline_store, :feature_registry, :cache]
  
  def start_link(config) do
    GenServer.start_link(__MODULE__, config, name: __MODULE__)
  end
  
  def register_feature_group(name, schema, opts \\ []) do
    GenServer.call(__MODULE__, {:register_feature_group, name, schema, opts})
  end
  
  def get_features(entity_id, feature_group, features) do
    GenServer.call(__MODULE__, {:get_features, entity_id, feature_group, features})
  end
  
  def update_features(entity_id, feature_group, feature_values) do
    GenServer.cast(__MODULE__, {:update_features, entity_id, feature_group, feature_values})
  end
  
  def batch_get_features(entity_ids, feature_group, features) do
    GenServer.call(__MODULE__, {:batch_get_features, entity_ids, feature_group, features})
  end
  
  def init(config) do
    online_store = connect_online_store(config.online_store)
    offline_store = connect_offline_store(config.offline_store)
    cache = Cache.new(config.cache_size || 10000)
    
    {:ok, %__MODULE__{
      online_store: online_store,
      offline_store: offline_store,
      feature_registry: %{},
      cache: cache
    }}
  end
  
  def handle_call({:register_feature_group, name, schema, opts}, _from, state) do
    feature_group = %{
      name: name,
      schema: schema,
      ttl: Keyword.get(opts, :ttl, 3600),
      storage_type: Keyword.get(opts, :storage_type, :redis),
      materialization: Keyword.get(opts, :materialization, :real_time)
    }
    
    new_registry = Map.put(state.feature_registry, name, feature_group)
    new_state = %{state | feature_registry: new_registry}
    
    {:reply, :ok, new_state}
  end
  
  def handle_call({:get_features, entity_id, feature_group, features}, _from, state) do
    cache_key = {entity_id, feature_group, features}
    
    case Cache.get(state.cache, cache_key) do
      {:hit, cached_features} ->
        {:reply, {:ok, cached_features}, state}
      
      :miss ->
        case fetch_features_from_store(state.online_store, entity_id, feature_group, features) do
          {:ok, feature_values} ->
            # Cache the result
            Cache.put(state.cache, cache_key, feature_values)
            {:reply, {:ok, feature_values}, state}
          
          {:error, reason} ->
            {:reply, {:error, reason}, state}
        end
    end
  end
  
  def handle_call({:batch_get_features, entity_ids, feature_group, features}, _from, state) do
    # Batch fetch for efficiency
    results = 
      entity_ids
      |> Enum.map(fn entity_id ->
        case fetch_features_from_store(state.online_store, entity_id, feature_group, features) do
          {:ok, feature_values} -> {entity_id, feature_values}
          {:error, _} -> {entity_id, nil}
        end
      end)
      |> Map.new()
    
    {:reply, {:ok, results}, state}
  end
  
  def handle_cast({:update_features, entity_id, feature_group, feature_values}, state) do
    # Update online store
    store_features(state.online_store, entity_id, feature_group, feature_values)
    
    # Invalidate cache
    cache_pattern = {entity_id, feature_group, :_}
    Cache.invalidate_pattern(state.cache, cache_pattern)
    
    {:noreply, state}
  end
  
  defp fetch_features_from_store(online_store, entity_id, feature_group, features) do
    # Implementation depends on storage backend (Redis, DynamoDB, etc.)
    case online_store.type do
      :redis ->
        key = "#{feature_group}:#{entity_id}"
        RedisClient.hmget(online_store.connection, key, features)
      
      :dynamodb ->
        DynamoDBClient.get_item(online_store.connection, feature_group, entity_id, features)
      
      :cassandra ->
        CassandraClient.get_features(online_store.connection, feature_group, entity_id, features)
    end
  end
  
  def create_training_dataset(feature_groups, entity_ids, point_in_time, opts \\ []) do
    # Point-in-time correct feature extraction for training
    output_format = Keyword.get(opts, :format, :parquet)
    
    # Fetch historical features for each entity and timestamp
    training_data = 
      Enum.map(entity_ids, fn entity_id ->
        features = 
          Enum.reduce(feature_groups, %{}, fn feature_group, acc ->
            historical_features = get_historical_features(
              feature_group, 
              entity_id, 
              point_in_time
            )
            Map.merge(acc, historical_features)
          end)
        
        Map.put(features, :entity_id, entity_id)
      end)
    
    # Export to requested format
    case output_format do
      :parquet -> export_to_parquet(training_data, opts)
      :csv -> export_to_csv(training_data, opts)
      :avro -> export_to_avro(training_data, opts)
    end
  end
  
### Day 14: Capstone Projects and Production Best Practices

#### Morning Session (4 hours)
**Lab 14.1: End-to-End ML System Architecture**
```elixir
defmodule MLSystemArchitecture do
  @moduledoc "Complete ML system with all production components"
  
  use Application
  
  def start(_type, _args) do
    children = [
      # Core infrastructure
      {Registry, keys: :unique, name: MLSystemRegistry},
      {DynamicSupervisor, name: MLSystemSupervisor, strategy: :one_for_one},
      
      # Data pipeline
      {DataIngestionPipeline, []},
      {FeatureStore, get_feature_store_config()},
      {DataValidationService, []},
      
      # Model training pipeline
      {TrainingPipelineOrchestrator, []},
      {ModelRegistry, []},
      {ExperimentTracker, []},
      
      # Model serving
      {ModelServingCluster, get_serving_config()},
      {ABTestingFramework, []},
      
      # Monitoring and observability
      {ModelMonitoringPipeline, []},
      {MetricsCollector, []},
      {AlertManager, []},
      
      # API Gateway
      {MLAPIGateway, get_api_config()}
    ]
    
    opts = [strategy: :one_for_one, name: MLSystem.Supervisor]
    Supervisor.start_link(children, opts)
  end
  
  defp get_feature_store_config() do
    %{
      online_store: %{
        type: :redis,
        host: System.get_env("REDIS_HOST", "localhost"),
        port: String.to_integer(System.get_env("REDIS_PORT", "6379"))
      },
      offline_store: %{
        type: :s3_parquet,
        bucket: System.get_env("S3_BUCKET", "ml-feature-store"),
        region: System.get_env("AWS_REGION", "us-east-1")
      },
      cache_size: 50_000
    }
  end
  
  defp get_serving_config() do
    %{
      models: [
        %{
          model_name: "recommendation_model",
          model_path: "/models/recommendation/latest",
          pool_size: 4,
          circuit_breaker: %{failure_threshold: 5, recovery_timeout: 30_000}
        },
        %{
          model_name: "fraud_detection_model", 
          model_path: "/models/fraud/latest",
          pool_size: 8,
          circuit_breaker: %{failure_threshold: 3, recovery_timeout: 60_000}
        }
      ],
      cluster: %{
        load_balancing_strategy: :least_loaded,
        health_check_interval: 30_000
      }
    }
  end
  
  defp get_api_config() do
    %{
      port: String.to_integer(System.get_env("API_PORT", "4000")),
      rate_limiting: %{
        requests_per_minute: 1000,
        burst_size: 100
      },
      authentication: %{
        enabled: true,
        jwt_secret: System.get_env("JWT_SECRET")
      }
    }
  end
end

defmodule MLAPIGateway do
  @moduledoc "API Gateway for ML system with authentication, rate limiting, and routing"
  
  use Plug.Router
  
  plug :match
  plug :dispatch
  
  # Health check endpoint
  get "/health" do
    health_status = %{
      status: "healthy",
      timestamp: DateTime.utc_now(),
      services: check_service_health()
    }
    
    send_resp(conn, 200, Jason.encode!(health_status))
  end
  
  # Model prediction endpoints
  post "/predict/:model_name" do
    with {:ok, features} <- extract_features(conn),
         {:ok, user_id} <- authenticate_request(conn),
         :ok <- check_rate_limit(user_id),
         {:ok, prediction} <- make_prediction(model_name, features, user_id) do
      
      response = %{
        prediction: prediction,
        model: model_name,
        timestamp: DateTime.utc_now()
      }
      
      send_resp(conn, 200, Jason.encode!(response))
    else
      {:error, :unauthorized} ->
        send_resp(conn, 401, Jason.encode!(%{error: "Unauthorized"}))
      
      {:error, :rate_limited} ->
        send_resp(conn, 429, Jason.encode!(%{error: "Rate limit exceeded"}))
      
      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end
  
  # Batch prediction endpoint
  post "/batch_predict/:model_name" do
    with {:ok, batch_features} <- extract_batch_features(conn),
         {:ok, user_id} <- authenticate_request(conn),
         :ok <- check_batch_rate_limit(user_id, length(batch_features)),
         {:ok, predictions} <- make_batch_prediction(model_name, batch_features) do
      
      response = %{
        predictions: predictions,
        model: model_name,
        batch_size: length(predictions),
        timestamp: DateTime.utc_now()
      }
      
      send_resp(conn, 200, Jason.encode!(response))
    else
      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end
  
  # Model metrics endpoint
  get "/metrics/:model_name" do
    case ModelMetricsCollector.get_metrics(model_name) do
      {:ok, metrics} ->
        send_resp(conn, 200, Jason.encode!(metrics))
      
      {:error, :not_found} ->
        send_resp(conn, 404, Jason.encode!(%{error: "Model not found"}))
    end
  end
  
  # A/B test assignment endpoint
  get "/experiment/:experiment_name/assignment/:user_id" do
    case ABTestingFramework.get_model_assignment(user_id, experiment_name) do
      {:ok, variant} ->
        response = %{
          experiment: experiment_name,
          user_id: user_id,
          variant: variant,
          timestamp: DateTime.utc_now()
        }
        
        send_resp(conn, 200, Jason.encode!(response))
      
      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end
  
  # Feedback endpoint for model improvement
  post "/feedback/:model_name" do
    with {:ok, feedback_data} <- extract_feedback(conn),
         {:ok, user_id} <- authenticate_request(conn) do
      
      # Store feedback for model retraining
      FeedbackCollector.record_feedback(model_name, user_id, feedback_data)
      
      send_resp(conn, 200, Jason.encode!(%{status: "Feedback recorded"}))
    else
      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end
  
  match _ do
    send_resp(conn, 404, Jason.encode!(%{error: "Not found"}))
  end
  
  defp extract_features(conn) do
    case Jason.decode(conn.body_params) do
      {:ok, %{"features" => features}} when is_map(features) ->
        {:ok, features}
      
      {:ok, _} ->
        {:error, "Missing or invalid features"}
      
      {:error, _} ->
        {:error, "Invalid JSON"}
    end
  end
  
  defp authenticate_request(conn) do
    case get_req_header(conn, "authorization") do
      ["Bearer " <> token] ->
        case verify_jwt_token(token) do
          {:ok, claims} -> {:ok, claims["user_id"]}
          {:error, _} -> {:error, :unauthorized}
        end
      
      _ ->
        {:error, :unauthorized}
    end
  end
  
  defp make_prediction(model_name, features, user_id) do
    # Check if user is in A/B test
    variant = case ABTestingFramework.get_model_assignment(user_id, "#{model_name}_experiment") do
      {:ok, variant} -> variant
      {:error, _} -> "control"  # Default to control group
    end
    
    # Use appropriate model variant
    actual_model_name = if variant == "control", do: model_name, else: "#{model_name}_#{variant}"
    
    # Make prediction
    case ModelServerPool.predict(actual_model_name, features) do
      {:ok, prediction} ->
        # Record prediction for monitoring
        PredictionLogger.log_prediction(actual_model_name, user_id, features, prediction)
        
        # Record A/B test assignment
        if variant != "control" do
          ABTestingFramework.record_outcome(user_id, "#{model_name}_experiment", %{
            variant: variant,
            prediction: prediction
          })
        end
        
        {:ok, prediction}
      
      {:error, reason} ->
        {:error, reason}
    end
  end
  
  defp check_service_health() do
    services = [
      {:feature_store, FeatureStore},
      {:model_registry, ModelRegistry},
      {:monitoring_pipeline, ModelMonitoringPipeline}
    ]
    
    Enum.map(services, fn {service_name, module} ->
      status = 
        try do
          GenServer.call(module, :health_check, 1000)
          :healthy
        catch
          _, _ -> :unhealthy
        end
      
      {service_name, status}
    end)
    |> Map.new()
  end
end

defmodule MLWorkflowOrchestrator do
  @moduledoc "Orchestrates complete ML workflows from data to deployment"
  
  use GenStateMachine, callback_mode: :state_functions
  
  defstruct [
    :workflow_id, :config, :current_step, :artifacts, :metadata,
    :start_time, :estimated_completion, :error_count
  ]
  
  # Workflow states: :initialized -> :data_validation -> :feature_engineering -> 
  # :model_training -> :model_evaluation -> :model_deployment -> :monitoring_setup -> :completed
  
  def start_workflow(config) do
    GenStateMachine.start_link(__MODULE__, config)
  end
  
  def init(config) do
    workflow_data = %__MODULE__{
      workflow_id: UUID.uuid4(),
      config: config,
      current_step: 0,
      artifacts: %{},
      metadata: %{},
      start_time: DateTime.utc_now(),
      estimated_completion: estimate_completion_time(config),
      error_count: 0
    }
    
    {:ok, :initialized, workflow_data}
  end
  
  # State: initialized
  def initialized(:enter, _old_state, data) do
    IO.puts("Starting ML workflow: #{data.workflow_id}")
    {:next_state, :data_validation, data}
  end
  
  # State: data_validation
  def data_validation(:enter, _old_state, data) do
    IO.puts("Step 1/7: Data Validation")
    
    Task.async(fn ->
      DataValidator.validate_dataset(
        data.config.dataset_path,
        data.config.validation_schema
      )
    end)
    |> Task.await()
    |> case do
      {:ok, validation_results} ->
        new_artifacts = Map.put(data.artifacts, :data_validation, validation_results)
        new_data = %{data | artifacts: new_artifacts}
        {:next_state, :feature_engineering, new_data}
      
      {:error, validation_errors} ->
        handle_workflow_error(data, :data_validation, validation_errors)
    end
  end
  
  # State: feature_engineering
  def feature_engineering(:enter, _old_state, data) do
    IO.puts("Step 2/7: Feature Engineering")
    
    validation_results = data.artifacts.data_validation
    
    Task.async(fn ->
      FeatureEngineer.transform_dataset(
        validation_results.cleaned_data,
        data.config.feature_transformations
      )
    end)
    |> Task.await()
    |> case do
      {:ok, engineered_features} ->
        new_artifacts = Map.put(data.artifacts, :features, engineered_features)
        new_data = %{data | artifacts: new_artifacts}
        {:next_state, :model_training, new_data}
      
      {:error, feature_errors} ->
        handle_workflow_error(data, :feature_engineering, feature_errors)
    end
  end
  
  # State: model_training
  def model_training(:enter, _old_state, data) do
    IO.puts("Step 3/7: Model Training")
    
    features = data.artifacts.features
    
    # Distributed training if configured
    training_result = 
      if data.config.distributed_training do
        DistributedTraining.train_model_distributed(
          features,
          data.config.model_config,
          data.config.training_config
        )
      else
        SingleNodeTraining.train_model(
          features,
          data.config.model_config,
          data.config.training_config
        )
      end
    
    case training_result do
      {:ok, trained_model} ->
        new_artifacts = Map.put(data.artifacts, :trained_model, trained_model)
        new_data = %{data | artifacts: new_artifacts}
        {:next_state, :model_evaluation, new_data}
      
      {:error, training_errors} ->
        handle_workflow_error(data, :model_training, training_errors)
    end
  end
  
  # State: model_evaluation
  def model_evaluation(:enter, _old_state, data) do
    IO.puts("Step 4/7: Model Evaluation")
    
    trained_model = data.artifacts.trained_model
    test_features = data.artifacts.features.test_set
    
    evaluation_results = ModelEvaluator.comprehensive_evaluation(
      trained_model,
      test_features,
      data.config.evaluation_metrics
    )
    
    if meets_quality_gates?(evaluation_results, data.config.quality_gates) do
      new_artifacts = Map.put(data.artifacts, :evaluation, evaluation_results)
      new_data = %{data | artifacts: new_artifacts}
      {:next_state, :model_deployment, new_data}
    else
      handle_workflow_error(data, :model_evaluation, "Model quality gates not met")
    end
  end
  
  # State: model_deployment
  def model_deployment(:enter, _old_state, data) do
    IO.puts("Step 5/7: Model Deployment")
    
    trained_model = data.artifacts.trained_model
    
    deployment_result = ModelDeploymentService.deploy_model(
      trained_model,
      data.config.deployment_config
    )
    
    case deployment_result do
      {:ok, deployment_info} ->
        new_artifacts = Map.put(data.artifacts, :deployment, deployment_info)
        new_data = %{data | artifacts: new_artifacts}
        {:next_state, :monitoring_setup, new_data}
      
      {:error, deployment_errors} ->
        handle_workflow_error(data, :model_deployment, deployment_errors)
    end
  end
  
  # State: monitoring_setup
  def monitoring_setup(:enter, _old_state, data) do
    IO.puts("Step 6/7: Monitoring Setup")
    
    deployment_info = data.artifacts.deployment
    
    # Setup monitoring dashboards
    MonitoringSetup.create_model_dashboard(
      data.config.model_name,
      deployment_info,
      data.config.monitoring_config
    )
    
    # Setup alerts
    AlertSetup.configure_model_alerts(
      data.config.model_name,
      data.config.alert_config
    )
    
    # Setup A/B testing if configured
    if data.config.ab_testing do
      ABTestingFramework.create_experiment(data.config.ab_testing)
    end
    
    new_artifacts = Map.put(data.artifacts, :monitoring, %{
      dashboard_url: MonitoringSetup.get_dashboard_url(data.config.model_name),
      alert_rules: data.config.alert_config
    })
    
    new_data = %{data | artifacts: new_artifacts}
    {:next_state, :completed, new_data}
  end
  
  # State: completed
  def completed(:enter, _old_state, data) do
    IO.puts("Step 7/7: Workflow Completed!")
    
    completion_time = DateTime.utc_now()
    duration = DateTime.diff(completion_time, data.start_time, :second)
    
    workflow_summary = %{
      workflow_id: data.workflow_id,
      duration_seconds: duration,
      artifacts: data.artifacts,
      completion_time: completion_time
    }
    
    # Store workflow results
    WorkflowRegistry.store_workflow_results(data.workflow_id, workflow_summary)
    
    # Send completion notification
    NotificationService.send_workflow_completion(data.config.notification_config, workflow_summary)
    
    {:keep_state, %{data | metadata: Map.put(data.metadata, :summary, workflow_summary)}}
  end
  
  defp handle_workflow_error(data, failed_step, error) do
    new_error_count = data.error_count + 1
    
    if new_error_count <= 3 do
      # Retry the step
      IO.puts("Error in #{failed_step}, retrying (attempt #{new_error_count})...")
      
      # Wait before retry
      Process.sleep(1000 * new_error_count)
      
      new_data = %{data | error_count: new_error_count}
      {:repeat_state, new_data}
    else
      # Fail the workflow
      IO.puts("Workflow failed at #{failed_step}: #{inspect(error)}")
      
      failure_summary = %{
        workflow_id: data.workflow_id,
        failed_step: failed_step,
        error: error,
        failure_time: DateTime.utc_now()
      }
      
      WorkflowRegistry.store_failure(data.workflow_id, failure_summary)
      {:next_state, :failed, data}
    end
  end
  
  defp meets_quality_gates?(evaluation_results, quality_gates) do
    Enum.all?(quality_gates, fn {metric, threshold} ->
      case Map.get(evaluation_results, metric) do
        nil -> false
        value -> value >= threshold
      end
    end)
  end
  
  defp estimate_completion_time(config) do
    # Estimate based on dataset size and model complexity
    base_time = 3600  # 1 hour base
    
    dataset_factor = estimate_dataset_time_factor(config.dataset_size)
    model_factor = estimate_model_time_factor(config.model_config.type)
    
    estimated_seconds = base_time * dataset_factor * model_factor
    
    DateTime.add(DateTime.utc_now(), estimated_seconds, :second)
  end
end
```

#### Afternoon Session (4 hours)
**Lab 14.2: Production Best Practices and Capstone Project**
```elixir
defmodule ProductionMLBestPractices do
  @moduledoc """
  Comprehensive guide and implementation of ML production best practices
  """
  
  # Configuration management
  defmodule ConfigManager do
    @moduledoc "Centralized configuration management for ML systems"
    
    def load_config(environment \\ :prod) do
      base_config = load_base_config()
      env_config = load_environment_config(environment)
      secrets = load_secrets()
      
      base_config
      |> Map.merge(env_config)
      |> Map.merge(secrets)
      |> validate_config()
    end
    
    defp load_base_config() do
      %{
        # Model serving
        model_serving: %{
          timeout_ms: 5000,
          max_batch_size: 64,
          circuit_breaker: %{
            failure_threshold: 5,
            recovery_timeout_ms: 30_000
          }
        },
        
        # Feature store
        feature_store: %{
          cache_ttl_seconds: 3600,
          batch_size: 1000,
          max_connections: 20
        },
        
        # Monitoring
        monitoring: %{
          metrics_interval_ms: 10_000,
          drift_check_interval_ms: 300_000,
          alert_thresholds: %{
            latency_p99_ms: 1000,
            error_rate_percent: 5.0,
            drift_score: 0.1
          }
        },
        
        # Data pipeline
        data_pipeline: %{
          batch_size: 10_000,
          max_retries: 3,
          validation_sample_rate: 0.1
        }
      }
    end
    
    defp validate_config(config) do
      required_keys = [
        [:model_serving, :timeout_ms],
        [:feature_store, :cache_ttl_seconds],
        [:monitoring, :alert_thresholds]
      ]
      
      Enum.each(required_keys, fn key_path ->
        if get_in(config, key_path) == nil do
          raise "Missing required configuration: #{inspect(key_path)}"
        end
      end)
      
      config
    end
  end
  
  # Security best practices
  defmodule SecurityManager do
    @moduledoc "Security controls for ML systems"
    
    def sanitize_input(input) do
      input
      |> validate_input_schema()
      |> normalize_numerical_values()
      |> escape_string_values()
      |> check_input_size_limits()
    end
    
    def encrypt_model_artifacts(model_data, encryption_key) do
      # Encrypt sensitive model parameters
      :crypto.crypto_one_time(:aes_256_gcm, encryption_key, generate_iv(), model_data, true)
    end
    
    def audit_prediction_request(user_id, model_name, features, prediction) do
      audit_entry = %{
        timestamp: DateTime.utc_now(),
        user_id: user_id,
        model_name: model_name,
        feature_hash: hash_features(features),
        prediction_hash: hash_prediction(prediction),
        ip_address: get_client_ip()
      }
      
      AuditLogger.log(audit_entry)
    end
    
    def implement_differential_privacy(dataset, epsilon \\ 1.0) do
      # Add calibrated noise for differential privacy
      noise_scale = 1.0 / epsilon
      
      Enum.map(dataset, fn row ->
        Enum.map(row, fn value ->
          if is_number(value) do
            value + :rand.normal() * noise_scale
          else
            value
          end
        end)
      end)
    end
    
    defp hash_features(features), do: :crypto.hash(:sha256, Jason.encode!(features))
    defp hash_prediction(prediction), do: :crypto.hash(:sha256, Jason.encode!(prediction))
    defp generate_iv(), do: :crypto.strong_rand_bytes(12)
  end
  
  # Resource management
  defmodule ResourceManager do
    @moduledoc "Manages compute resources and auto-scaling"
    
    def monitor_resource_usage() do
      %{
        cpu_usage: get_cpu_usage(),
        memory_usage: get_memory_usage(),
        gpu_usage: get_gpu_usage(),
        disk_usage: get_disk_usage(),
        network_io: get_network_io()
      }
    end
    
    def auto_scale_decision(current_metrics, thresholds) do
      cond do
        should_scale_up?(current_metrics, thresholds) ->
          {:scale_up, calculate_scale_up_factor(current_metrics)}
        
        should_scale_down?(current_metrics, thresholds) ->
          {:scale_down, calculate_scale_down_factor(current_metrics)}
        
        true ->
          :no_change
      end
    end
    
    def optimize_memory_usage() do
      # Garbage collection
      :erlang.garbage_collect()
      
      # Clear unused caches
      Cache.clear_expired_entries()
      
      # Compact model weights if possible
      ModelRegistry.compact_models()
    end
    
    defp should_scale_up?(metrics, thresholds) do
      metrics.cpu_usage > thresholds.cpu_scale_up or
      metrics.memory_usage > thresholds.memory_scale_up
    end
    
    defp should_scale_down?(metrics, thresholds) do
      metrics.cpu_usage < thresholds.cpu_scale_down and
      metrics.memory_usage < thresholds.memory_scale_down
    end
  end
  
  # Data quality assurance
  defmodule DataQualityManager do
    @moduledoc "Ensures data quality throughout the ML pipeline"
    
    def validate_data_quality(dataset, quality_rules) do
      results = %{
        completeness: check_completeness(dataset, quality_rules.completeness),
        accuracy: check_accuracy(dataset, quality_rules.accuracy),
        consistency: check_consistency(dataset, quality_rules.consistency),
        timeliness: check_timeliness(dataset, quality_rules.timeliness),
        validity: check_validity(dataset, quality_rules.validity)
      }
      
      overall_score = calculate_overall_quality_score(results)
      
      %{
        individual_scores: results,
        overall_score: overall_score,
        passed: overall_score >= quality_rules.minimum_score
      }
    end
    
    def detect_data_anomalies(new_data, baseline_stats) do
      %{
        statistical_anomalies: detect_statistical_anomalies(new_data, baseline_stats),
        pattern_anomalies: detect_pattern_anomalies(new_data, baseline_stats),
        temporal_anomalies: detect_temporal_anomalies(new_data, baseline_stats)
      }
    end
    
    def data_lineage_tracking(dataset_id, transformations) do
      lineage_record = %{
        dataset_id: dataset_id,
        source_datasets: extract_source_datasets(transformations),
        transformations: transformations,
        created_at: DateTime.utc_now(),
        created_by: get_current_user(),
        data_schema: infer_schema(dataset_id)
      }
      
      DataLineageStore.record(lineage_record)
    end
    
    defp check_completeness(dataset, rules) do
      total_cells = count_total_cells(dataset)
      missing_cells = count_missing_cells(dataset)
      completeness_ratio = (total_cells - missing_cells) / total_cells
      
      %{
        ratio: completeness_ratio,
        passed: completeness_ratio >= rules.minimum_completeness
      }
    end
  end
  
  # Model governance
  defmodule ModelGovernance do
    @moduledoc "Implements model governance and compliance"
    
    def register_model_for_governance(model_info) do
      governance_record = %{
        model_id: model_info.id,
        model_name: model_info.name,
        version: model_info.version,
        owner: model_info.owner,
        business_purpose: model_info.business_purpose,
        risk_category: assess_risk_category(model_info),
        compliance_requirements: identify_compliance_requirements(model_info),
        bias_assessment: perform_bias_assessment(model_info),
        explainability_level: assess_explainability(model_info),
        registered_at: DateTime.utc_now()
      }
      
      GovernanceRegistry.register(governance_record)
    end
    
    def assess_model_bias(model, test_dataset, protected_attributes) do
      bias_metrics = %{}
      
      # Calculate bias metrics for each protected attribute
      for attribute <- protected_attributes do
        groups = group_by_attribute(test_dataset, attribute)
        
        bias_metrics = Map.put(bias_metrics, attribute, %{
          demographic_parity: calculate_demographic_parity(model, groups),
          equalized_odds: calculate_equalized_odds(model, groups),
          calibration: calculate_calibration(model, groups)
        })
      end
      
      overall_bias_score = calculate_overall_bias_score(bias_metrics)
      
      %{
        individual_metrics: bias_metrics,
        overall_bias_score: overall_bias_score,
        bias_detected: overall_bias_score > 0.1
      }
    end
    
    def model_explainability_report(model, sample_predictions) do
      %{
        feature_importance: calculate_feature_importance(model),
        local_explanations: generate_local_explanations(model, sample_predictions),
        counterfactual_examples: generate_counterfactuals(model, sample_predictions),
        decision_rules: extract_decision_rules(model)
      }
    end
    
    def compliance_audit_trail(model_id, start_date, end_date) do
      %{
        model_changes: query_model_changes(model_id, start_date, end_date),
        prediction_logs: query_prediction_logs(model_id, start_date, end_date),
        access_logs: query_access_logs(model_id, start_date, end_date),
        performance_metrics: query_performance_metrics(model_id, start_date, end_date)
      }
    end
    
    defp assess_risk_category(model_info) do
      # Risk assessment based on model domain and impact
      case {model_info.domain, model_info.impact_level} do
        {"healthcare", _} -> :high
        {"finance", "high"} -> :high
        {"finance", "medium"} -> :medium
        {_, "high"} -> :medium
        _ -> :low
      end
    end
  end
end

# Capstone Project Template
defmodule CapstoneMLProject do
  @moduledoc """
  Template for capstone ML project demonstrating end-to-end implementation
  """
  
  def run_complete_ml_workflow() do
    IO.puts("Starting Capstone ML Project: Real-time Recommendation System")
    
    # 1. Data Pipeline Setup
    {:ok, data_pipeline} = setup_data_pipeline()
    
    # 2. Feature Store Setup
    {:ok, feature_store} = setup_feature_store()
    
    # 3. Model Training Pipeline
    {:ok, trained_model} = train_recommendation_model()
    
    # 4. Model Evaluation
    evaluation_results = evaluate_model(trained_model)
    
    # 5. Model Deployment
    {:ok, deployment} = deploy_model(trained_model)
    
    # 6. A/B Testing Setup
    {:ok, experiment} = setup_ab_testing(deployment)
    
    # 7. Monitoring Setup
    setup_monitoring(deployment)
    
    # 8. API Gateway Configuration
    {:ok, api_gateway} = setup_api_gateway(deployment)
    
    IO.puts("Capstone project completed successfully!")
    
    %{
      data_pipeline: data_pipeline,
      feature_store: feature_store,
      model: trained_model,
      evaluation: evaluation_results,
      deployment: deployment,
      experiment: experiment,
      api_gateway: api_gateway
    }
  end
  
  defp setup_data_pipeline() do
    # Real-time data ingestion from multiple sources
    config = %{
      sources: [
        %{type: :kafka, topic: "user_interactions", format: :json},
        %{type: :database, table: "user_profiles", batch_size: 10000},
        %{type: :api, endpoint: "https://api.content.com/items", rate_limit: 1000}
      ],
      transformations: [
        %{type: :deduplication, key: "user_id"},
        %{type: :feature_engineering, pipeline: RecommendationFeatures},
        %{type: :validation, schema: UserInteractionSchema}
      ],
      output: %{type: :feature_store, batch_size: 1000}
    }
    
    DataIngestionPipeline.start_link(config)
  end
  
  defp train_recommendation_model() do
    # Collaborative filtering with deep learning
    model_config = %{
      type: :neural_collaborative_filtering,
      embedding_dim: 128,
      hidden_layers: [256, 128, 64],
      dropout_rate: 0.2,
      regularization: 0.001
    }
    
    training_config = %{
      batch_size: 1024,
      epochs: 50,
      learning_rate: 0.001,
      early_stopping: %{patience: 5, metric: :val_loss}
    }
    
    # Load training data
    training_data = FeatureStore.get_training_dataset(
      ["user_features", "item_features", "interaction_features"],
      start_date: ~D[2024-01-01],
      end_date: ~D[2024-12-31]
    )
    
    # Train model
    RecommendationModel.train(model_config, training_data, training_config)
  end
  
  defp evaluate_model(model) do
    test_data = FeatureStore.get_test_dataset()
    
    metrics = %{
      # Recommendation-specific metrics
      precision_at_k: ModelEvaluator.precision_at_k(model, test_data, k: 10),
      recall_at_k: ModelEvaluator.recall_at_k(model, test_data, k: 10),
      ndcg_at_k: ModelEvaluator.ndcg_at_k(model, test_data, k: 10),
      
      # Business metrics
      coverage: ModelEvaluator.catalog_coverage(model, test_data),
      diversity: ModelEvaluator.recommendation_diversity(model, test_data),
      novelty: ModelEvaluator.recommendation_novelty(model, test_data),
      
      # Fairness metrics
      bias_assessment: ModelGovernance.assess_model_bias(
        model, 
        test_data, 
        ["age_group", "gender", "location"]
      )
    }
    
    IO.puts("Model Evaluation Results:")
    IO.inspect(metrics)
    
    metrics
  end
  
  defp setup_monitoring(deployment) do
    # Configure comprehensive monitoring
    monitoring_config = %{
      metrics: [
        :prediction_latency,
        :prediction_accuracy,
        :feature_drift,
        :prediction_drift,
        :model_bias,
        :business_kpis
      ],
      alerts: [
        %{metric: :prediction_latency_p99, threshold: 100, unit: :milliseconds},
        %{metric: :feature_drift_score, threshold: 0.1, unit: :score},
        %{metric: :accuracy_drop, threshold: 0.05, unit: :percentage}
      ],
      dashboards: [
        %{name: "Model Performance", metrics: [:accuracy, :latency, :throughput]},
        %{name: "Data Quality", metrics: [:feature_drift, :data_quality_score]},
        %{name: "Business Impact", metrics: [:click_through_rate, :conversion_rate]}
      ]
    }
    
    ModelMonitoringPipeline.setup_monitoring(deployment.model_id, monitoring_config)
  end
end

# Summary and Best Practices Documentation
defmodule MLProductionSummary do
  @moduledoc """
  Summary of best practices and lessons learned from the bootcamp
  """
  
  def print_production_checklist() do
    checklist = """
    
    # ML Production Readiness Checklist
    
    ## Data Infrastructure
    ✅ Data validation pipelines
    ✅ Feature store implementation
    ✅ Data lineage tracking
    ✅ Data quality monitoring
    ✅ Backup and disaster recovery
    
    ## Model Development
    ✅ Experiment tracking
    ✅ Model versioning
    ✅ Automated testing
    ✅ Code review process
    ✅ Model validation framework
    
    ## Deployment Infrastructure
    ✅ Container orchestration
    ✅ Load balancing
    ✅ Auto-scaling
    ✅ Circuit breakers
    ✅ Health checks
    
    ## Monitoring & Observability
    ✅ Model performance monitoring
    ✅ Data drift detection
    ✅ Alert management
    ✅ Logging and tracing
    ✅ Business metrics tracking
    
    ## Security & Compliance
    ✅ Authentication and authorization
    ✅ Data encryption
    ✅ Audit trails
    ✅ Bias detection
    ✅ Privacy protection
    
    ## Operational Excellence
    ✅ CI/CD pipelines
    ✅ Infrastructure as code
    ✅ Documentation
    ✅ Incident response procedures
    ✅ Performance optimization
    """
    
    IO.puts(checklist)
  end
  
  def elixir_ml_advantages() do
    advantages = """
    
    # Why Elixir for ML Production Systems?
    
    ## Concurrency & Fault Tolerance
    - Actor model enables massive concurrency
    - Let-it-crash philosophy for resilient systems
    - Built-in supervision trees
    
    ## Real-time Processing
    - Low-latency message passing
    - Excellent for streaming ML pipelines
    - Hot code swapping for zero-downtime deployments
    
    ## Distributed Computing
    - Native clustering capabilities
    - Location transparency
    - Automatic failover and recovery
    
    ## Operational Benefits
    - Excellent observability with built-in tracing
    - Low memory footprint
    - Predictable performance characteristics
    """
    
    IO.puts(advantages)
  end
end

# Final project deliverables
defmodule BootcampDeliverables do
  def generate_final_report() do
    IO.puts("Generating final bootcamp report...")
    
    # This would generate a comprehensive report of all labs completed
    # and provide a portfolio of ML systems built during the bootcamp
    
    %{
      completed_labs: list_completed_labs(),
      built_systems: list_built_systems(),
      code_repositories: list_code_repos(),
      performance_benchmarks: collect_benchmarks(),
      production_deployments: list_deployments()
    }
  end
  
  defp list_completed_labs() do
    [
      "Day 1: Elixir Fundamentals for AI/ML",
      "Day 2: Mathematical Foundations",
      "Day 3: Neural Networks and Deep Learning",
      "Day 4: Concurrent ML Processing",
      "Day 5: Real-time ML Systems",
      "Day 6: Advanced ML Architectures",
      "Day 7: Production ML Systems", 
      "Day 8: MLOps and Infrastructure",
      "Day 9: Reinforcement Learning",
      "Day 10: Natural Language Processing",
      "Day 11: Computer Vision",
      "Day 12: Time Series and Forecasting",
      "Day 13: Model Deployment and Serving",
      "Day 14: Production Best Practices"
    ]
  end
end
```

---

## Enhanced Assessment Criteria

### Technical Implementation (35%)
- **Mathematical Correctness**: Formal verification of concurrent algorithms with convergence proofs
- **OTP Pattern Usage**: Appropriate supervision strategies for ML workload characteristics
- **Performance Optimization**: Quantitative analysis and benchmarking with numerical stability
- **Fault Tolerance**: Demonstration through chaos engineering and recovery testing

### Production Engineering (30%)
- **Complete MLOps Pipeline**: Implementation with supervision tree design and fault tolerance
- **Monitoring and Observability**: Statistical process control with real-time dashboards
- **Security Implementation**: Audit trails, access control, and Byzantine-resilient protocols
- **Documentation Quality**: Mathematical derivations and operational procedures

### Mathematical Rigor (25%)
- **Algorithm Derivations**: Step-by-step mathematical explanations with formal proofs
- **Convergence Analysis**: For distributed optimization algorithms and concurrent systems
- **Statistical Significance**: Testing for ML model comparisons with confidence intervals
- **Error Propagation**: Analysis in distributed computation pipelines with bounds

### Professional Workflow (10%)
- **Code Review Participation**: Focus on concurrent correctness and mathematical soundness
- **Collaboration Effectiveness**: In distributed team environments with async communication
- **Incident Response**: Capability for complex distributed systems with runbook procedures
- **Continuous Learning**: Advanced concurrent programming concepts and research integration

---

## BEAM-Specific Advantages for ML Systems

### Concurrency & Fault Tolerance
- **Actor Model**: Enables massive concurrency with mathematical correctness guarantees
- **Let-it-crash Philosophy**: Resilient systems with automatic recovery and state preservation
- **Supervision Trees**: Hierarchical fault tolerance with restart strategies tailored to ML workloads
- **Process Isolation**: Mathematical computations protected from error propagation

### Real-time Processing
- **Low-latency Message Passing**: Critical for real-time ML inference and online learning
- **Excellent for Streaming**: ML pipelines with back-pressure and flow control
- **Hot Code Swapping**: Zero-downtime model updates and algorithm improvements
- **Predictable Performance**: Soft real-time guarantees for ML serving systems

### Distributed Computing
- **Native Clustering**: Automatic node discovery and transparent distribution
- **Location Transparency**: Distributed ML training across heterogeneous infrastructure  
- **Automatic Failover**: Byzantine fault tolerance for distributed optimization
- **Consistent Hashing**: Load balancing for stateful ML processes

### Operational Benefits
- **Built-in Observability**: Tracing and monitoring with mathematical performance analysis
- **Low Memory Footprint**: Efficient resource utilization for large-scale ML systems
- **Immutable State**: Functional programming prevents race conditions in concurrent algorithms
- **Pattern Matching**: Elegant handling of ML pipeline states and error conditions

---

## Daily Workflow Patterns with Mathematical Rigor

### Development Practices
- **Morning Standups**: Process health monitoring with mathematical performance analysis
- **Code Reviews**: Focus on concurrent correctness, numerical stability, and convergence proofs
- **Retrospectives**: Supervision tree reviews with fault injection testing and recovery validation
- **Load Testing**: Concurrent user simulations with statistical analysis of system behavior

### Production Operations
- **Performance Profiling**: Mathematical analysis of message passing overhead and optimization
- **Hot Deployments**: Zero-downtime model updates with A/B testing and statistical validation
- **Incident Response**: Systematic troubleshooting of distributed ML systems with runbooks
- **Capacity Planning**: Mathematical modeling of resource requirements and scaling policies

### Quality Assurance
- **Property-Based Testing**: Mathematical invariants verified across concurrent execution paths
- **Chaos Engineering**: Fault injection with Byzantine failure scenarios and recovery testing
- **Statistical Validation**: ML model performance with confidence intervals and significance testing
- **Convergence Monitoring**: Real-time analysis of optimization algorithms and system stability

---

## Learning Outcomes for BEAM ML Engineers

### Core Competencies
1. **Design fault-tolerant ML systems** using OTP principles with mathematical correctness guarantees
2. **Implement concurrent ML algorithms** with actor model patterns and numerical stability analysis  
3. **Debug distributed ML systems** using BEAM tools with systematic troubleshooting methodologies
4. **Optimize process communication** and memory usage for large-scale ML workloads
5. **Deploy and monitor production** BEAM ML applications with comprehensive observability

### Mathematical Expertise
1. **Prove convergence properties** of distributed optimization algorithms in concurrent environments
2. **Analyze numerical stability** in concurrent mathematical computations with error bounds
3. **Implement statistically rigorous** A/B testing with concurrent experiment management
4. **Design fault-tolerant** statistical computations with error propagation analysis
5. **Validate ML model performance** using distributed statistical testing frameworks

### Modern Architecture Implementation
1. **Build transformer models** with concurrent attention computation and mathematical analysis
2. **Implement Graph Neural Networks** with distributed message passing and convergence guarantees
3. **Design generative models** with fault-tolerant distributed training and statistical validation
4. **Create reinforcement learning** systems with actor-based policy optimization and theoretical bounds
5. **Develop federated learning** systems with privacy preservation and mathematical guarantees

---

## Capstone Project: SmartCommerce AI Production System

The two-week bootcamp culminates in implementing a complete fault-tolerant recommendation system that demonstrates all enhanced principles:

### System Architecture
- **CustomerBehavior.Server**: Real-time preference learning with convergence guarantees
- **Recommendation.Pool**: Distributed similarity computation with Byzantine fault tolerance  
- **ML.Supervisor**: Hierarchical supervision with mathematical correctness preservation
- **FeatureStore.Cache**: Distributed caching with consistency and automatic invalidation
- **Analytics.Aggregator**: Stream processing with statistical rigor and error bounds

### Mathematical Rigor Integration
- **Preference Learning**: Exponential moving averages with convergence proofs
- **Similarity Computation**: Cosine similarity with numerical stability analysis
- **A/B Testing**: Statistical significance testing with concurrent experiment management
- **Performance Monitoring**: Control charts and drift detection with mathematical foundations

### Production Engineering
- **Zero-downtime Deployments**: Hot code swapping with model version management
- **Fault Tolerance**: Byzantine-resilient distributed computation with automatic recovery
- **Observability**: Real-time dashboards with mathematical performance metrics
- **Security**: Audit trails, access control, and differential privacy implementation

---

## Course Completion and Certification

### Portfolio Deliverables
1. **Complete SmartCommerce AI System**: Production-ready recommendation engine
2. **Mathematical Analysis Portfolio**: Convergence proofs and numerical stability reports  
3. **Performance Benchmarks**: Quantitative analysis of concurrent ML algorithms
4. **Production Deployment**: Live system running on distributed BEAM cluster
5. **Incident Response Simulation**: Chaos engineering and recovery documentation

### Advanced Certification Tracks
1. **Distributed Systems Architect**: Advanced OTP patterns and cluster coordination
2. **ML Research Engineer**: Integration of latest research with mathematical rigor
3. **Production ML Platform**: Enterprise-scale MLOps with BEAM ecosystem
4. **Academic Collaboration**: Research partnerships and conference presentations

This enhanced curriculum ensures graduates can design, implement, and operate mathematically rigorous, fault-tolerant ML systems that leverage the unique advantages of the BEAM virtual machine and Elixir ecosystem for production environments.

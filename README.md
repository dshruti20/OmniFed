# FLORA: Federated Learning and Analytics Platform

A framework for simulated and real-world deployments of federated learning applications at scale.

# Hybrid Federated Learning (MPI + gRPC)

A 2-facility hybrid Federated Learning system combining MPI-based intra-facility
aggregation with gRPC-based inter-facility communication.

| Component | Details |
|---|---|
| Intra-facility | MPI (`torch.distributed`) → mean aggregation |
| Inter-facility | gRPC Parameter Server → weighted FedAvg |
| Model | ResNet18 |
| Dataset | CIFAR-10 |

---

## 1. Environment
```bash
conda activate omnifed_flora
pip install torch torchvision grpcio omegaconf
```

---

## 2. Configuration (YAML)

All topology is defined in `config/*.yaml`.

Example (asymmetric 2 + 8 workers):
```yaml
topology:
  world_size: 11   # 1 server + 2 + 8

  rpc:
    server_rank: 0
    addr: "127.0.0.1"
    port: 50051
    client_ranks: [1, 3]

  facilities:
    - name: "fac1"
      mpi:
        addr: "127.0.0.1"
        port: 28250
        world_size: 2
        members: [1, 2]
        leader_rank: 1

    - name: "fac2"
      mpi:
        addr: "127.0.0.1"
        port: 28290
        world_size: 8
        members: [3, 4, 5, 6, 7, 8, 9, 10]
        leader_rank: 3

training:
  dataset_total_clients: 10
```

> You can change worker counts freely by editing `facilities` and `world_size`.

---

## 3. Run
```bash
bash test_scripts/generic_hybrid_comm.sh --config asym_hybrid_topo.yaml
```

The launch script does the following:

- Reads `world_size` from the YAML config
- Launches ranks `0` through `world_size - 1`
- **Rank 0** → gRPC parameter server
- **Facility leaders** → MPI + gRPC client
- **Workers** → MPI only

---

## 4. What Happens Each Round

Every `comm_freq` steps, the following sequence runs:

1. **MPI all-reduce** (mean) inside each facility
2. Facility leader sends weighted update to **gRPC server**
3. gRPC server performs **weighted FedAvg**
4. Averaged model is returned to leaders
5. **MPI broadcast** distributes the model inside each facility

---

## 5. Logs

Each rank writes output to:
```
omnifed_data/flora_test/g{rank}/stdout.log
```

---

## 6. Supported Topologies

| Type | Example |
|---|---|
| Symmetric | 3 + 3 workers |
| Asymmetric | 2 + 8 workers |

Topology is fully YAML-driven — no code changes needed.

---

## 7. Common Fixes

**Kill stuck processes:**
```bash
pkill -f omega_launch_hybridcomm
```

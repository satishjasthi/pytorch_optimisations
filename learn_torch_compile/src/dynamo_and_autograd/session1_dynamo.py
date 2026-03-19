# session1_dynamo.py
import torch
import torch._dynamo
from src.model import TinyTransformer


torch.manual_seed(42)

if torch.cuda.is_available():
    torch.set_default_device("cuda")
    model = TinyTransformer().cuda()
    x = torch.randint(0, 1000, (2, 128)).cuda()   # batch=2, seq_len=128
else: 
    model = TinyTransformer()
    x = torch.randint(0, 1000, (2, 128))   # batch=2, seq_len=128

# ── Step 1: explain() ────────────────────────────────────────────────────────
# This is your microscope. It tells you exactly what Dynamo captured,
# what it skipped, and why — without actually compiling anything.

explanation = torch._dynamo.explain(model)(x)

print("=== GRAPHS CAPTURED ===")
print(f"Number of graphs:      {len(explanation.graphs)}")
print(f"Number of graph breaks:{len(explanation.break_reasons)}")
print()

for i, graph in enumerate(explanation.graphs):
    print(f"--- Graph {i} ---")
    graph.print_readable()   # human-readable FX graph
    print()

for i, reason in enumerate(explanation.break_reasons):
    print(f"--- Break {i}: {reason.reason} ---")
    print(f"    at: {reason.user_stack}")

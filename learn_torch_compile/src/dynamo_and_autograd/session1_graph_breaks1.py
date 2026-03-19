# session1_graph_breaks.py
import torch
import torch._dynamo
torch._dynamo.reset()
from src.model import TinyTransformer

# ── Experiment A: data-dependent control flow ─────────────────────────────
# Dynamo cannot trace through Python if-statements that depend on tensor values.
# It has to break the graph at that point.

class ModelWithBreak(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = TinyTransformer()

    def forward(self, x):
        out = self.base(x)
        # THIS causes a graph break — the condition depends on a tensor value
        if out.sum() > 0:          # Dynamo can't know this at trace time
            return out * 2
        return out

if torch.cuda.is_available():
    model_break = ModelWithBreak().cuda()
    x = torch.randint(0, 1000, (2, 128)).cuda()
else:
    model_break = ModelWithBreak()
    x = torch.randint(0, 1000, (2, 128))

explanation = torch._dynamo.explain(model_break)(x)
print(f"Graphs with data-dependent break: {len(explanation.graphs)}")
for r in explanation.break_reasons:
    print(f"  Break reason: {r.reason}")


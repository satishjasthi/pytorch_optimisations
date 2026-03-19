# session1_graph_breaks.py
import torch
import torch._dynamo
from src.model import TinyTransformer

# ── Experiment C: print() inside forward ─────────────────────────────────
class ModelWithPrint(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = TinyTransformer()

    def forward(self, x):
        out = self.base(x)
        print(f"shape: {out.shape}")  # side effect = graph break
        return out

if torch.cuda.is_available():
    model_print = ModelWithPrint().cuda()
    x = torch.randint(0, 1000, (2, 128)).cuda()
else:
    model_print = ModelWithPrint()
    x = torch.randint(0, 1000, (2, 128))

expl3 = torch._dynamo.explain(model_print)(x)
print(f"Graphs with print():              {len(expl3.graphs)}")
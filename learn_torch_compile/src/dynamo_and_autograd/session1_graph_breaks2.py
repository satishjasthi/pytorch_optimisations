# session1_graph_breaks.py
import torch
import torch._dynamo
from src.model import TinyTransformer

# ── Experiment B: unsupported Python (try/except) ─────────────────────────
class ModelWithTryCatch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = TinyTransformer()

    def forward(self, x):
        try:
            return self.base(x)    # try/except = graph break, always
        except Exception:
            return x.float()

if torch.cuda.is_available():
    model_try = ModelWithTryCatch().cuda()
    x = torch.randint(0, 1000, (2, 128)).cuda()
else:
    model_try = ModelWithTryCatch()
    x = torch.randint(0, 1000, (2, 128))

expl2 = torch._dynamo.explain(model_try)(x)
print(f"Graphs with try/except:           {len(expl2.graphs)}")


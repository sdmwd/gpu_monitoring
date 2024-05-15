import torch
import torch.nn as nn

class example_model(nn.Module):
    def __init__(self):
        super(example_model, self).__init__()
        self.fc = nn.Linear(10, 10)
        self.return_parts = True  # Attribut initialisé à True

    def forward(self, x):
        return self.fc(x)

    def predict(self, x):
        output = self.forward(x)
        pred = output.argmax(dim=1)
        out = output
        others = output + 1  # Juste pour illustrer
        return pred, out, others


model = example_model()

gpu_index = 4

if torch.cuda.is_available() and gpu_index < torch.cuda.device_count():
    device = torch.device(f'cuda:{gpu_index}')
    print(f"Utilisation du GPU: cuda:{gpu_index}")
elif torch.cuda.is_available():
    device = torch.device(f'cuda:0')
    print(f"GPU {gpu_index} non disponible, utilisation du GPU 0")
else:
    device = torch.device('cpu')
    print("GPU {gpu_index} non disponible, utilisation du CPU")

model.to(device)
print(f"Modèle déplacé vers le périphérique: {device}")

X = torch.randn(64, 10).to(device)

return_parts = model.return_parts

if return_parts:
    pred, out, others = model.predict(X)
else:
    pred, out = model.predict(X)

print("Pred shape:", pred.shape)
print("Out shape:", out.shape)
if return_parts:
    print("Others shape:", others.shape)

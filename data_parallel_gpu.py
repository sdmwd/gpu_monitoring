import torch
import torch.nn as nn

class example_model(nn.Module):
    def __init__(self):
        super(example_model, self).__init__()
        self.fc = nn.Linear(10, 10)
        self.return_parts = True

    def forward(self, x):
        return self.fc(x)

    def predict(self, x):
        output = self.forward(x)
        pred = output.argmax(dim=1)
        out = output
        others = output + 1  # Juste pour illustrer
        return pred, out, others


model = example_model()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

if torch.cuda.device_count() > 1:
    print(f"Utilisation de {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
    print("Modèle enveloppé dans DataParallel.")
else:
    print("Un seul GPU disponible ou utilisation du CPU. Pas de parallélisation.")

X = torch.randn(64, 10).to(device)

if isinstance(model, nn.DataParallel):
    return_parts = model.module.return_parts
else:
    return_parts = model.return_parts

if return_parts:
    if isinstance(model, nn.DataParallel):
        pred, out, others = model.module.predict(X)
    else:
        pred, out, others = model.predict(X)
else:
    if isinstance(model, nn.DataParallel):
        pred, out = model.module.predict(X)
    else:
        pred, out = model.predict(X)

print("Pred shape:", pred.shape)
print("Out shape:", out.shape)
if return_parts:
    print("Others shape:", others.shape)

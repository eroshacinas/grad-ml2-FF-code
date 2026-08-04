import torch
import torch.nn.functional as F

# train func
def ff_epoch(model, loader, layer_optimizers, device, train=True):
    model.train() if train else model.eval()

    n_layers   = len(model.hidden_layers)
    total_loss = torch.zeros(n_layers)

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x_pos, x_neg, _ in loader:          # label no longer needed here
            x_pos, x_neg = x_pos.to(device), x_neg.to(device)

            x = torch.cat([x_pos, x_neg], dim=0)
            y = torch.cat([
                torch.ones(x_pos.size(0),  device=device),
                torch.zeros(x_neg.size(0), device=device),
            ], dim=0)

            losses = model.ff_forward(x, y)

            if train:
                for opt, loss in zip(layer_optimizers, losses):
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            total_loss += torch.stack([l.detach().cpu() for l in losses])

    return (total_loss / len(loader)).tolist()


# inference
@torch.no_grad()
def ff_accuracy(model, infer_loader, device, skip_first=False):
    model.eval()
    correct = total = 0
    for candidates, labels in infer_loader:
        candidates, labels = candidates.to(device), labels.to(device)
        B, C, D = candidates.shape
        goodnesses = model(candidates.view(B * C, D))
        g = goodnesses[1:] if (skip_first and len(goodnesses) > 1) else goodnesses
        scores = torch.stack(g, dim=1).sum(dim=1).view(B, C)
        correct += (scores.argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return correct / total
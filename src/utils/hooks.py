"""Feature map and gradient capture hooks for DL course instrumentation."""

import torch
import torch.nn as nn


class FeatureGradientCapture:
    """Captures forward activations and backward gradients for specified layers.
    
    Used for Slide 6 of the DL course presentation:
    Shows layer output values/feature maps and backpropagated gradients.
    """

    def __init__(self):
        self.activations = {}
        self.gradients = {}
        self._hooks = []

    def register(self, model, layer_names):
        """Register hooks on named modules."""
        for name, module in model.named_modules():
            if name in layer_names:
                h1 = module.register_forward_hook(
                    lambda m, inp, out, n=name: self._save_activation(n, out)
                )
                h2 = module.register_full_backward_hook(
                    lambda m, grad_in, grad_out, n=name: self._save_gradient(n, grad_out)
                )
                self._hooks.extend([h1, h2])

    def _save_activation(self, name, output):
        if isinstance(output, torch.Tensor):
            self.activations[name] = output.detach().cpu()

    def _save_gradient(self, name, grad_output):
        if isinstance(grad_output, tuple) and len(grad_output) > 0:
            if isinstance(grad_output[0], torch.Tensor):
                self.gradients[name] = grad_output[0].detach().cpu()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def generate_layer_specs(model, input_image, input_tokens):
    """Generate layer-wise specification table for Slides 3-5.
    
    Runs a forward pass with hooks to capture each layer's specs.
    """
    specs = []
    hooks = []

    def make_hook(name):
        def hook_fn(m, inp, out):
            spec = {
                "layer_name": name,
                "type": m.__class__.__name__,
                "parameters": sum(p.numel() for p in m.parameters()),
            }
            # Input shape
            if isinstance(inp, tuple) and len(inp) > 0:
                if isinstance(inp[0], torch.Tensor):
                    spec["input_shape"] = str(list(inp[0].shape))
            # Output shape
            if isinstance(out, torch.Tensor):
                spec["output_shape"] = str(list(out.shape))
            elif isinstance(out, tuple) and len(out) > 0:
                if isinstance(out[0], torch.Tensor):
                    spec["output_shape"] = str(list(out[0].shape))

            # Layer-specific details
            if isinstance(m, nn.Conv2d):
                spec["kernel_size"] = str(m.kernel_size)
                spec["stride"] = str(m.stride)
                spec["padding"] = str(m.padding)
                spec["groups"] = m.groups
                spec["in_channels"] = m.in_channels
                spec["out_channels"] = m.out_channels
            elif isinstance(m, nn.Linear):
                spec["in_features"] = m.in_features
                spec["out_features"] = m.out_features
            elif isinstance(m, nn.LSTM):
                spec["input_size"] = m.input_size
                spec["hidden_size"] = m.hidden_size
                spec["num_layers"] = m.num_layers
                spec["bidirectional"] = m.bidirectional
            elif isinstance(m, nn.BatchNorm2d):
                spec["num_features"] = m.num_features

            specs.append(spec)
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.LSTM,
                               nn.BatchNorm2d, nn.Embedding)):
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

    # Forward pass
    with torch.no_grad():
        model(input_image, input_tokens)

    # Cleanup
    for h in hooks:
        h.remove()

    return specs

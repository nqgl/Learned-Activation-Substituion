import transformer_lens
import circuitsvis as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from jaxtyping import Float
import transformer_lens.utils as utils
# Load a model (eg GPT-2 Small)
model = transformer_lens.HookedTransformer.from_pretrained("pythia-70m")

# Run the model and get logits and activations
logits, activations = model.run_with_cache("Hello World")
print(activations["blocks.0.attn.hook_v"].shape)
class GradThruRelu(Function):
    @staticmethod
    def forward(ctx, x):
        return F.relu(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
gradthrurelu = GradThruRelu.apply
def gradthruclamp(x):
    return 1 - gradthrurelu(1 - gradthrurelu(x))


n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
#C_ablation = torch.zeros((n_layers, n_heads))
# C_ablations = {}
torch.manual_seed(1)
ablation_parameters = torch.rand(n_layers, n_heads)
ablation_parameters.requires_grad = True


def simple_value_mutator(value, layer_ablation_parameters, layer_index):
    return value * (1 - layer_ablation_parameters.view(1, 1, -1, 1))



def head_ablation_hook_tuples(ablation_parameters, value_mutator=simple_value_mutator):
    def head_ablation_hook_generator(layer_ablation_parameters: Float[torch.Tensor, "head_index"], layer_index: int):
        def head_ablation_hook_fn(
                    value: Float[torch.Tensor, "batch pos head_index d_head"],
                    hook: transformer_lens.hook_points.HookPoint
                ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
            # print(f"Shape of the value tensor: {value.shape}")
            return value_mutator(value, gradthruclamp(layer_ablation_parameters.to(value.device)), layer_index)
        return head_ablation_hook_fn
    
    hook_tuples = [(
        utils.get_act_name("v", layer_to_ablate), 
        head_ablation_hook_generator(ablation_parameters[layer_to_ablate], layer_to_ablate)
        ) for layer_to_ablate in range(n_layers)]

    return hook_tuples

hook_tuples = head_ablation_hook_tuples(ablation_parameters)

text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
tokens = model.to_tokens(text)

def loss_function1(original_logits, ablated_logits, ablation_parameters):
    ablation_parameters = gradthruclamp(ablation_parameters)
    similarity_to_original = torch.cosine_similarity(original_logits, ablated_logits, dim=-1)
    similarity_to_original = F.cross_entropy(ablated_logits, F.softmax(original_logits, dim=-1))
    return torch.mean(similarity_to_original) * 100 + torch.mean(abs(1-ablation_parameters) ** 1) * 0.04 - torch.mean(abs(1 - ablation_parameters) ** 2) * 0.01 

def parameter_loss_polynomial(ablation_parameters):
    return torch.mean(abs(ablation_parameters) ** 1) * 0.04 - torch.mean(abs(ablation_parameters) ** 2) * 0.01 



def logit_cross_entropy_loss(logits_1, logits_2):
    return F.cross_entropy(ablated_logits, F.softmax(original_logits, dim=-1))

def loss_function(original_logits, ablated_logits, ablation_parameters):
    return 100 * logit_cross_entropy_loss(original_logits, ablated_logits) + parameter_loss_polynomial(ablation_parameters) * 2 + parameter_loss_polynomial(1 - ablation_parameters)
    
    
original_loss = model(tokens, return_type="loss")
original_logits = model(tokens, return_type="logits")
ablated_loss = model.run_with_hooks(
    tokens, 
    return_type="loss",
    fwd_hooks=hook_tuples
    )

optimizer = torch.optim.SGD([ablation_parameters], lr=1, momentum=0.9)
torch.set_printoptions(sci_mode=False, linewidth=120, precision=3)
for i in range(1000):
    original_logits = model(tokens, return_type="logits")
    ablated_logits = model.run_with_hooks(
        tokens, 
        return_type="logits",
        fwd_hooks=hook_tuples
        )
    ablated_loss = model.run_with_hooks(
    tokens, 
    return_type="loss",
    fwd_hooks=hook_tuples
    )
    optimizer.zero_grad()
    loss = loss_function(original_logits, ablated_logits, ablation_parameters)
    loss.backward(retain_graph=True)
    optimizer.step()
    print(ablation_parameters)
    print(f"Ablated loss:{ablated_loss}")



original_loss = model(tokens, return_type="loss")
ablated_loss = model.run_with_hooks(
    tokens, 
    return_type="loss",
    fwd_hooks=hook_tuples
    )

print(f"Original Loss: {original_loss.item():.3f}")
print(f"Ablated Loss: {ablated_loss.item():.3f}")
print(f"ablation constant gradients: {[ablation_parameters.grad]}")

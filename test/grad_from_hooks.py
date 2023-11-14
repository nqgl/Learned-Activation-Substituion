import transformer_lens
import circuitsvis as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from jaxtyping import Float
import transformer_lens.utils as utils
import ioi_prompts  
# Load a model (eg GPT-2 Small)
model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")

#TODO clean up all of the unwrapped closures by wrapping them, and then move all of the top level state code into main()

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
prompts, corrupted_prompts, answers, answer_tokens, corrupted_tokens = ioi_prompts.prompts(model)
tokens = model.to_tokens(prompts, prepend_bos=True)
seq_len = tokens.shape[1]
torch.manual_seed(2)
ablation_parameters = torch.rand((n_layers, 1, seq_len, n_heads, 1)) / 10
ablation_parameters = torch.rand((n_layers, 1, 1, n_heads, 1)) / 10
print(ablation_parameters.shape)
ablation_parameters.requires_grad = True


def simple_value_mutator(value, layer_ablation_parameters, layer_index):
    return value * (1 - layer_ablation_parameters.view(1, 1, -1, 1))


def create_ablation_model(model, ablation_parameters, value_mutator):
    hook_tuples = head_ablation_hook_tuples(ablation_parameters=ablation_parameters, value_mutator=value_mutator)
    def _hooked_run(tokens, return_type="logits", **kwargs):
        return model.run_with_hooks(tokens, return_type=return_type, fwd_hooks=hook_tuples, **kwargs)
    return _hooked_run

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



def patching_value_mutator_generator(corrupted_activations):
    def patching_value_mutator(value, layer_ablation_parameters, layer_index): #todo, change this to activation_location
        patch_in_activations = corrupted_activations[utils.get_act_name("v", layer_index)]
        # print(patch_in_activations.shape)
        # print(value.shape)
        # layer_ablation_parameters = layer_ablation_parameters.view(1, layer_ablation_parameters.shape[1], -1, 1)
        # print(value.shape, layer_ablation_parameters.shape, patch_in_activations.shape)
        return value * (1 - layer_ablation_parameters) + layer_ablation_parameters * patch_in_activations
    return patching_value_mutator


def logit_cross_entropy_loss(logits_1, logits_2):
    return F.cross_entropy(logits_1, F.softmax(logits_2, dim=-1))

def penalty_function(ablation_parameters):
    n = torch.count_nonzero(torch.relu(ablation_parameters - 0.04))
    return torch.mean(torch.sqrt(torch.abs(ablation_parameters) + 0.0001)) * n
    return torch.mean(torch.abs(ablation_parameters)) * n

def loss_function(target_logits, ablated_logits, ablation_parameters):
    ablation_parameters = gradthruclamp(ablation_parameters)
    # print(ablation_parameters)
    ce_loss = logit_cross_entropy_loss(target_logits, ablated_logits)
    penalty = penalty_function(ablation_parameters)
    # print(f"ce_loss: {ce_loss}, penalty: {penalty}")
    return ce_loss + penalty * 0.2
    

    

torch.manual_seed(2)
# ablation_parameters = torch.rand(n_layers, n_heads) / 1
# ablation_parameters.requires_grad = True



prompts, corrupted_prompts, answers, answer_tokens, corrupted_tokens = ioi_prompts.prompts(model)

corrupted_logits, corrupted_activations = model.run_with_cache(corrupted_tokens, return_type="logits")

patching_value_mutator = patching_value_mutator_generator(corrupted_activations)
patched_model = create_ablation_model(model, ablation_parameters, patching_value_mutator)



tokens = model.to_tokens(prompts, prepend_bos=True)

ablated_loss = patched_model(tokens, return_type="loss")



optimizer = torch.optim.SGD([ablation_parameters], lr=0.04, momentum=0.9)
torch.set_printoptions(sci_mode=False, linewidth=120, precision=3)
for i in range(100):
    # original_logits = model(tokens, return_type="logits")
    corrupted_logits = model(corrupted_tokens, return_type="logits")
    ablated_logits = patched_model(tokens, return_type="logits")
    ablated_loss = patched_model(tokens, return_type="loss")
    optimizer.zero_grad()
    # print(corrupted_logits.shape)
    loss = loss_function(corrupted_logits[:, -1, answer_tokens], ablated_logits[:, -1, answer_tokens], ablation_parameters)
    loss += loss_function(corrupted_logits[:, -1, :], ablated_logits[:, -1, :], ablation_parameters)

    loss.backward()
    optimizer.step()
    print(ablation_parameters)
    print(f"Ablated loss:{ablated_loss}")


def show_prompt_differences(patched_model, model):
    prompts, corrupted_prompts, answers, answer_tokens, corrupted_tokens = ioi_prompts.prompts(model)
    corrupted_logits, corrupted_activations = model.run_with_cache(corrupted_tokens, return_type="logits")
    tokens = model.to_tokens(prompts, prepend_bos=True)
    ablated_logits = patched_model(tokens)
    for i in range(len(prompts)):
        prompt_ablated_logits = ablated_logits[i:i + 1, :, :]
        prompt = prompts[i]
        tokens = model.to_tokens(prompt, prepend_bos=True)
        original_logits = model(tokens, return_type="logits")
        print(f"\nPrompt: {prompt}")
        print(f"Answers: {answers[i]}")
        p_original = F.softmax(original_logits[:, -1, :], dim=-1)
        p_answers_original = [p_original[0, answer_tokens[i, 0]], p_original[0, answer_tokens[i, 1]]] 
        p_ablated = F.softmax(prompt_ablated_logits[:, -1, :], dim=-1)
        p_answers_ablated = [p_ablated[0, answer_tokens[i, 0]], p_ablated[0, answer_tokens[i, 1]]]
        print(f"Answer \"{answers[i][0]}\" original_prob: {p_answers_original[0].item()}, ablated_prob: {p_answers_ablated[0].item()}")
        print(f"Answer \"{answers[i][1]}\" original_prob: {p_answers_original[1].item()}, ablated_prob: {p_answers_ablated[1].item()}")
        top_token_original = torch.argmax(original_logits[:, -1, :], dim=-1)
        top_token_ablated = torch.argmax(prompt_ablated_logits[:, -1, :], dim=-1)
        print(f"Top token original: {model.tokenizer.decode(top_token_original.item())}, ablated: {model.tokenizer.decode(top_token_ablated.item())}")
        k = 5
        topk_original = torch.topk(original_logits[:, -1, :], k, dim=-1)[1][0]
        topk_ablated = torch.topk(prompt_ablated_logits[:, -1, :], k, dim=-1)[1][0]
        k_sp_original = [(model.tokenizer.decode(t.item()), p_original[0, t], p_ablated[0, t]) for t in topk_original]
        k_sp_ablated = [(model.tokenizer.decode(t.item()), p_original[0, t], p_ablated[0, t]) for t in topk_ablated]
        top_original_str = "\n\t".join([f'"{t[0]}" original_prob: {t[1].item()}, \t ablated_prob: {t[2].item()}' for i, t in enumerate(k_sp_original)])
        top_ablated_str = "\n\t".join([f'"{t[0]}" , ablated_prob: {t[2].item()} \t original_prob: {t[1].item()}' for i, t in enumerate(k_sp_ablated)])
        print(f"Top {k} tokens original: \n\t{top_original_str}")
        print(f"Top {k} tokens ablated: \n\t{top_ablated_str}")





def top_k_next_tokens(patched_model, model, prompts):
    tokens = model.to_tokens(prompts, prepend_bos=True)
    ablated_logits = patched_model(tokens)
    for i in range(len(prompts)):
        prompt_ablated_logits = ablated_logits[i:i + 1, :, :]
        prompt = prompts[i]
        tokens = model.to_tokens(prompt, prepend_bos=True)
        original_logits = model(tokens, return_type="logits")
        print(f"\nPrompt: {prompt}")
        p_original = F.softmax(original_logits[:, -1, :], dim=-1)
        p_ablated = F.softmax(prompt_ablated_logits[:, -1, :], dim=-1)
        top_token_original = torch.argmax(original_logits[:, -1, :], dim=-1)
        top_token_ablated = torch.argmax(prompt_ablated_logits[:, -1, :], dim=-1)
        print(f"Top token original: {model.tokenizer.decode(top_token_original.item())}, ablated: {model.tokenizer.decode(top_token_ablated.item())}")
        k = 5
        topk_original = torch.topk(original_logits[:, -1, :], k, dim=-1)[1][0]
        topk_ablated = torch.topk(prompt_ablated_logits[:, -1, :], k, dim=-1)[1][0]
        k_sp_original = [(model.tokenizer.decode(t.item()), p_original[0, t], p_ablated[0, t]) for t in topk_original]
        k_sp_ablated = [(model.tokenizer.decode(t.item()), p_original[0, t], p_ablated[0, t]) for t in topk_ablated]
        top_original_str = "\n\t".join([f'"{t[0]}" original_prob: {t[1].item()}, \t ablated_prob: {t[2].item()}' for i, t in enumerate(k_sp_original)])
        top_ablated_str = "\n\t".join([f'"{t[0]}" , ablated_prob: {t[2].item()} \t original_prob: {t[1].item()}' for i, t in enumerate(k_sp_ablated)])
        print(f"Top {k} tokens original: \n\t{top_original_str}")
        print(f"Top {k} tokens ablated: \n\t{top_ablated_str}")





def make_ablations_binary(ablation_parameters, threshold = 0.5):
    return torch.where(ablation_parameters > 0.5, torch.ones_like(ablation_parameters), torch.zeros_like(ablation_parameters))

show_prompt_differences(patched_model, model)
print(make_ablations_binary(ablation_parameters))
binary_patched_model = create_ablation_model(model, make_ablations_binary(ablation_parameters), patching_value_mutator)
show_prompt_differences(binary_patched_model, model)
torch.save(ablation_parameters, "ablation_parameters.pt")
print(sum(make_ablations_binary(ablation_parameters).flatten()))

# null_patched_model = create_ablation_model(model, torch.zeros_like(ablation_parameters), patching_value_mutator)
# show_prompt_differences(null_patched_model, model)


prom1tps = "After Martin and Amy went to the park,{} gave a drink to"

def print_heads_copied(ablation_parameters):
    if ablation_parameters.shape[2] == 1:    
        for layer in range(n_layers):
            for head in range(n_heads):
                if ablation_parameters[layer, 0, 0, head, 0] > 0.5:
                    print(f"Layer {layer} head {head} patched")
    else:
        for pos in range(ablation_parameters.shape[2]):
            for layer in range(n_layers):
                for head in range(n_heads):
                    if ablation_parameters[layer, 0, pos, head, 0] > 0.5:
                        print(f"Layer {layer} head {head} patched at token {pos}")

print_heads_copied(ablation_parameters)

def test_ablation_parameters(ablation_parameters, model, patching_value_mutator, custom=False):
    test_string = "After Kat and Carl went to the bar, Carl gave a ball to"
    test_string2 = "After Kat and Carl went to the bar, Kat gave a ball to"
    prompts = [test_string, test_string2] * 4
    patched_model = create_ablation_model(model, make_ablations_binary(ablation_parameters), patching_value_mutator)
    top_k_next_tokens(patched_model, model, prompts)
    top_k_next_tokens(patched_model, model, prompts[1:-1])

# test_ablation_parameters(ablation_parameters, model, patching_value_mutator)
# while True:
    # test_ablation_parameters(ablation_parameters, model, patching_value_mutator, custom=True)


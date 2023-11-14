import torch
prompt_format = [
    "When John and Mary went to the shops,{} gave the bag to",
    "When Tom and James went to the park,{} gave the ball to",
    "When Dan and Sid went to the shops,{} gave an apple to",
    "After Martin and Amy went to the park,{} gave a drink to",
]
names = [
    (" Mary", " John"),
    (" Tom", " James"),
    (" Dan", " Sid"),
    (" Martin", " Amy"),
]

def prompts(model):
    prompt_format = [
        "When John and Mary went to the shops,{} gave the bag to",
        "When Tom and James went to the park,{} gave the ball to",
        "When Dan and Sid went to the shops,{} gave an apple to",
        "After Martin and Amy went to the park,{} gave a drink to",
    ]
    names = [
        (" Mary", " John"),
        (" Tom", " James"),
        (" Dan", " Sid"),
        (" Martin", " Amy"),
    ]

    # List of prompts
    prompts = []
    # List of answers, in the format (correct, incorrect)
    answers = []
    # List of the token (ie an integer) corresponding to each answer, in the format (correct_token, incorrect_token)
    answer_tokens = []
    for i in range(len(prompt_format)):
        for j in range(2):
            answers.append((names[i][j], names[i][1 - j]))
            answer_tokens.append(
                (
                    model.to_single_token(answers[-1][0]),
                    model.to_single_token(answers[-1][1]),
                )
            )
            # Insert the *incorrect* answer to the prompt, making the correct answer the indirect object.
            prompts.append(prompt_format[i].format(answers[-1][1]))
    answer_tokens = torch.tensor(answer_tokens).cuda()

    corrupted_prompts = []
    for i in range(0, len(prompts), 2):
        corrupted_prompts.append(prompts[i + 1])
        corrupted_prompts.append(prompts[i])
    corrupted_tokens = model.to_tokens(corrupted_prompts, prepend_bos=True)
    # corrupted_logits, corrupted_cache = model.run_with_cache(
    #     corrupted_tokens, return_type="logits"
    # )
    return prompts, corrupted_prompts, answers, answer_tokens, corrupted_tokens

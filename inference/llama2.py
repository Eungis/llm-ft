import torch
import time
import os
import json
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from sentencepiece import SentencePieceProcessor
from typing import Optional, List
from models.llama2 import ModelArgs, Transformer


class Tokenizer:
    
    def __init__(self, model_path: str):
        assert os.path.isfile(model_path), f"{model_path} does not exist."
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        print(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()
    
    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t
    
    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)


class LLaMA2:
    
    def __init__(
        self,
        model: Transformer,
        tokenizer: Tokenizer,
        model_args: ModelArgs
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args
        
    @classmethod
    def build(
        cls,
        checkpoints_dir: str, 
        tokenizer_path: str,
        load_model: bool,
        max_seq_len: int,
        max_batch_size: int,
        device: str,
    ):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"
            ckpt_path = checkpoints[0] # only 1 pth file for llama-2-7b
            print(f"Loading checkpoint {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            print(f"Loaded checkpoint in {time.time() - prev_time:.2f}s")
            prev_time = time.time()
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )
        
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        
        # Sets the default torch.Tensor type to floating point tensor type t. 
        # This type will also be used as default floating point type for type inference in torch.tensor().
        # The default floating point tensor type is initially torch.FloatTensor.
        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)
        
        model = Transformer(model_args).to(device)

        if load_model:
            # The only unmatched key in the checkpoint is rope.freqs. Remove it
            del checkpoint['rope.freqs']
            # Set strict=True to avoid loading unmatched
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded state dict in {time.time() - prev_time:.2f}s")
        
        return LLaMA2(model, tokenizer, model_args)
    
    
    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float=.1,
        top_p: float=.001,
        logprobs: bool=False,
        device: str="cpu:0",
        echo: bool=False
    ):
        params, bsz = self.args, len(prompt_tokens)
        assert bsz <= params.max_batch_size, f"batch size must be less than or equal to {self.args.max_batch_size}"
        
        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)        
        
        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(prompt_tokens):
            # Populate the initial tokens with the prompt tokens
            tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)
        
        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cpu")
        input_text_mask = tokens != pad_id
        if min_prompt_len == total_len:
            logits = self.model.forward(tokens, prev_pos)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )
            
        for cur_pos in tqdm(range(min_prompt_len, total_len), desc="Generating Tokens..."):
            # |B, Seq_Len, Dim|
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                # |B, Seq_Len, Dim| -> |B, Dim|
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                # |B, 1|
                next_token = self._sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)
            
            # |B, 1| -> |1, B*1|
            next_token = next_token.reshape(-1)
            
            # Only replace token if prompt has already been generated (!=pad_id)
            # |1, B*1|
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            
            if logprobs:
                # same as below
                # -> |B*Seq_Len, 1|
                # token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                #     input=logits.contiguous().view(-1, logits.size(-1)), # |B, Dim, Seq_Len|
                #     target=tokens[:, prev_pos + 1 : cur_pos + 1].contiguous().view(-1), # |B, Seq_Len|
                #     reduction="none",
                #     ignore_index=pad_id
                # )
                # -> |B, Seq_Len|
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2), # |B, Dim, Seq_Len|
                    target=tokens[:, prev_pos + 1 : cur_pos + 1], # |B, Seq_Len|
                    reduction="none",
                    ignore_index=pad_id
                )
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break
            
        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start: len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start: len(prompt_tokens[i]) + max_gen_len]
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            out_tokens += [toks]
            out_logprobs += [probs]
        return (out_tokens, out_logprobs if logprobs else None)
    
    def text_completion(
        self,
        prompts: List[str],
        temperature: float=.1,
        top_p: float=.001,
        max_gen_len: Optional[int] = None,
        logprobs: bool=False,
        device: str="cpu:0",
        echo=False
    ):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1
        # convert each prompt into tokens
        prompt_tokens = [self.tokenizer.encode(prompt, bos=True, eos=False) for prompt in prompts]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            device=device,
            echo=echo,
        )
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]
        
    
    def _sample_top_p(self, probs, p):
        # (B, vocab_size)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        # (B, vocab_size)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        # (B, vocab_size)
        # (Substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking)
        mask = probs_sum - probs_sort > p 
        # Zero out all the probabilities of tokens that are not selected by the Top P
        probs_sort[mask] = 0.0 
        # Redistribute the probabilities so that they sum up to 1.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        # Sample a token (its index) from the top p distribution
        next_token = torch.multinomial(probs_sort, num_samples=1)
        # Get the token position in the vocabulary corresponding to the sampled index
        next_token = torch.gather(probs_idx, -1, next_token) 
        return next_token
            

if __name__ == '__main__':
    import sys
    sys.path.append("..")
    
    # set torch manual seed
    torch.manual_seed(0)

    allow_cuda = False
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'

    prompts = [
        "The capital city of Korea is",
        # Few shot prompt
        """Translate English to French:
        
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
    ]

    model = LLaMA2.build(
        checkpoints_dir='./checkpoints/Llama-2-7b/',
        tokenizer_path='./checkpoints/Llama-2-7b/tokenizer.model',
        load_model=True,
        max_seq_len=2048,
        max_batch_size=len(prompts),
        device=device
    )

    out_tokens, out_texts = (model.text_completion(prompts, max_gen_len=100))
    assert len(out_texts) == len(prompts)
    for i in range(len(out_texts)):
        print(f'{out_texts[i]}')
        print('-' * 50)
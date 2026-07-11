"""
Vision-Language Model from Scratch in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - split_image_into_patches
import torch

def split_image_into_patches(image, patch_size):
    """Split an image tensor (B, C, H, W) into a sequence of (patch_size, patch_size) patches.

    Returns a tensor of shape (B, num_patches, C, patch_size, patch_size) in row-major order.
    """
    # split the (B, C, H, W) image into (B, num_patches, C, patch_size, patch_size).
    B, C, H, W = image.shape
    P = patch_size
    N_h = H//P
    N_w = W//P
    N = N_h * N_w

    patches = image.reshape(B, C, N_h, P, N_w, P)
    patches = patches.permute(0, 2, 4, 1, 3, 5)
    patches = patches.reshape(B, N, C, P, P)
    return patches

# Step 2 - flatten_patches
def flatten_patches(patches):
    # flatten each patch's channel and spatial dims into one vector, keep (B, N) leading dims.
    B, N, C, P_h, P_w = patches.shape
    return patches.reshape(B, N, C*P_h*P_w)

# Step 3 - linear_projection
import torch

def linear_projection(x, weight, bias):
    """Apply y = x @ weight.T + bias with arbitrary leading dims on x."""
    # compute the affine map y = x @ weight.T + bias
    return x @ weight.T + bias

# Step 4 - project_patches_to_embeddings
import torch

def project_patches_to_embeddings(flat_patches, patch_proj_weight, patch_proj_bias):
    # Linearly project flattened image patches into the ViT embedding dimension.
    return linear_projection(flat_patches, patch_proj_weight, patch_proj_bias)

# Step 5 - prepend_class_token
import torch

def prepend_class_token(patch_embeddings, class_token):
    """Prepend a learnable [CLS] token to the patch embedding sequence.

    patch_embeddings: (B, num_patches, embed_dim)
    class_token:      (1, 1, embed_dim)
    returns:          (B, num_patches+1, embed_dim)
    """
    # prepend the [CLS] token to every sequence in the batch
    B = patch_embeddings.shape[0]
    return torch.cat((class_token.expand(B, -1, -1), patch_embeddings), 1)

# Step 6 - add_position_embeddings
import torch

def add_position_embeddings(tokens, position_embeddings):
    """Add learnable position embeddings to a (B, S, D) token sequence."""
    # combine tokens (B, S, D) with position_embeddings (1, S, D) via broadcasting.
    return tokens + position_embeddings

# Step 7 - compute_attention_scores
import torch

def compute_attention_scores(q, k):
    """Compute raw attention scores Q @ K^T.

    q: (..., Sq, d_head)
    k: (..., Sk, d_head)
    returns: (..., Sq, Sk)
    """
    # compute the raw attention scores as Q times K-transpose
    return q @ k.mT

# Step 8 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_head):
    """Scale raw attention scores so softmax inputs stay well-conditioned."""
    # Divide raw attention scores by a constant derived from d_head.
    return scores / math.sqrt(d_head)

# Step 9 - apply_attention_mask
def apply_attention_mask(scores, mask):
    # add an additive mask (0 = allowed, -inf = blocked) to attention scores.
    if mask is None:
        return scores
    return scores + mask

# Step 10 - attention_softmax
import torch

def attention_softmax(masked_scores):
    """Softmax over the last (key) axis of attention scores."""
    # convert masked attention scores into normalized weights over the key axis
    return torch.softmax(masked_scores, dim=-1)

# Step 11 - attention_context
import torch

def attention_context(attn_weights, v):
    """Combine attention weights with values to produce context vectors."""
    # return a tensor of shape (..., Sq, d_head) from attn_weights and v
    return attn_weights @ v

# Step 12 - scaled_dot_product_attention
import torch

def scaled_dot_product_attention(q, k, v, mask=None):
    """Compose score, scale, mask, softmax, and context into full attention."""
    # compose the five attention primitives into a single forward pass.
    scores = compute_attention_scores(q, k)
    d_head = q.size(-1)
    scaled_scores = scale_attention_scores(scores, d_head)
    masked_scores = apply_attention_mask(scaled_scores, mask)
    attn_weights = attention_softmax(masked_scores)
    return attention_context(attn_weights, v)

# Step 13 - split_into_heads
import torch

def split_into_heads(x, num_heads):
    """Reshape (B, S, d_model) into (B, num_heads, S, d_head)."""
    # split the last dim into (num_heads, d_head) and move heads next to batch
    B, S, d_model = x.shape
    return x.reshape(B, S, num_heads, -1).transpose(1, 2)

# Step 14 - merge_heads
import torch

def merge_heads(x):
    """Merge (B, num_heads, S, d_head) back to (B, S, num_heads*d_head)."""
    # merge the multi-head dimension back into the model dimension
    B, num_heads, S, d_head = x.shape    
    return x.transpose(1, 2).reshape(B, S, num_heads*d_head)

# Step 15 - project_qkv
def project_qkv(x, wq, bq, wk, bk, wv, bv):
    # project x into separate query, key, and value tensors using three linear layers.
    q = linear_projection(x, wq, bq)
    k = linear_projection(x, wk, bk)
    v = linear_projection(x, wv, bv)
    return (q, k, v)

# Step 16 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    # reshape q, k, v from (B, S, d_model) into (B, num_heads, S, d_head) each
    q_h = split_into_heads(q, num_heads)
    k_h = split_into_heads(k, num_heads)
    v_h = split_into_heads(v, num_heads)
    return (q_h, k_h, v_h)

# Step 17 - multi_head_attention_scores
import torch

def multi_head_attention_scores(q_h, k_h, v_h, mask=None):
    """Run scaled dot-product attention in parallel across all heads.

    q_h, k_h, v_h: (B, num_heads, S, d_head)
    mask: broadcastable to (B, num_heads, S, S) or None
    returns: (B, num_heads, S, d_head)
    """
    # run scaled dot-product attention across the head axis
    return scaled_dot_product_attention(q_h, k_h, v_h, mask=mask)

# Step 18 - merge_and_output_project
import torch

def merge_and_output_project(context_heads, wo, bo):
    """Merge heads back to d_model and apply the output projection."""
    # merge multi-head context to (B, S, d_model) then apply linear projection with wo, bo
    x = merge_heads(context_heads)
    return linear_projection(x, wo, bo)

# Step 19 - multi_head_self_attention
import torch

def multi_head_self_attention(x, params, num_heads, mask=None):
    """Run full multi-head self-attention: QKV proj, head split, attention, merge, output proj."""
    # compose project_qkv, split_qkv_into_heads, multi_head_attention_scores, merge_and_output_project.
    wq, bq, wk, bk, wv, bv, wo, bo = params['wq'], params['bq'], params['wk'], params['bk'], params['wv'], params['bv'], params['wo'], params['bo']
    q, k, v = project_qkv(x, wq, bq, wk, bk, wv, bv)
    q_h, k_h, v_h = split_qkv_into_heads(q, k, v, num_heads)
    context_heads = multi_head_attention_scores(q_h, k_h, v_h, mask=mask)
    output = merge_and_output_project(context_heads, wo, bo)
    return output

# Step 20 - gelu_activation
import torch
import math

def gelu_activation(x):
    """Apply the exact (erf-based) GELU activation elementwise to x."""
    # implement GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    return x * 0.5 * (1 + torch.erf(x/math.sqrt(2)))

# Step 21 - mlp_first_layer
import torch

def mlp_first_layer(x, w1, b1):
    """Apply the first linear layer of the MLP block followed by GELU."""
    # project x to the feed-forward dimension and apply GELU
    y = linear_projection(x, w1, b1)
    return gelu_activation(y)

# Step 22 - mlp_second_layer
import torch

def mlp_second_layer(h, w2, b2):
    # project the MLP hidden activations back down to d_model using w2 and b2
    return linear_projection(h, w2, b2)

# Step 23 - mlp_block
import torch

def mlp_block(x, params):
    """Two-layer position-wise MLP with GELU between the layers."""
    # Assemble the position-wise two-layer MLP block with GELU between layers.
    w1, b1, w2, b2 = params['w1'], params['b1'], params['w2'], params['b2']
    h = mlp_first_layer(x, w1, b1)
    return mlp_second_layer(h, w2, b2)

# Step 24 - compute_layernorm_stats
import torch

def compute_layernorm_stats(x, eps=1e-5):
    # return (mean, var) along the last dim, each with shape (..., 1).
    mean = torch.mean(x, -1, keepdim=True)
    var = torch.var(x, -1, keepdim=True, correction=0)
    return (mean, var)

# Step 25 - layer_norm
import torch

def layer_norm(x, gamma, beta, eps=1e-5):
    # normalize the last dim of x and apply learnable scale gamma and shift beta
    mean, var = compute_layernorm_stats(x, eps=eps)
    x_hat = (x-mean)/torch.sqrt(var + eps)
    return gamma * x_hat + beta

# Step 26 - residual_add
import torch

def residual_add(residual, sublayer_output):
    """Add residual skip connection to a sublayer's output."""
    # return the element-wise sum of residual and sublayer_output
    return residual + sublayer_output

# Step 27 - pre_norm_sublayer
import torch

def pre_norm_sublayer(x, gamma, beta, sublayer_fn):
    """Apply pre-norm: LN(x) -> sublayer -> add residual x."""
    # layer-normalize x, run sublayer_fn on it, then add the residual
    return residual_add(x, sublayer_fn(layer_norm(x, gamma, beta)))

# Step 28 - vision_encoder_block
import torch

def vision_encoder_block(x, block_params, num_heads):
    # pre-norm MHSA sublayer, then pre-norm MLP sublayer, both with residuals.
    sublayer_mha = lambda x : multi_head_self_attention(x, block_params['attn'], num_heads, mask=None)
    gamma1, beta1 = block_params['ln1_gamma'], block_params['ln1_beta']
    y = pre_norm_sublayer(x, gamma1, beta1, sublayer_mha)

    sublayer_mlp = lambda x : mlp_block(x, block_params['mlp'])
    gamma2, beta2 = block_params['ln2_gamma'], block_params['ln2_beta']
    return pre_norm_sublayer(y, gamma2, beta2, sublayer_mlp)

# Step 29 - vision_encoder
import torch

def vision_encoder(patch_sequence, encoder_params, num_heads):
    """Stack ViT encoder blocks then apply a final layer norm to the patch sequence."""
    # run patch_sequence through every block in encoder_params['blocks'], then final layer norm.
    x = patch_sequence
    for block in encoder_params['blocks']:
        x = vision_encoder_block(x, block, num_heads)
    return layer_norm(x, encoder_params['final_ln_gamma'], encoder_params['final_ln_beta'])

# Step 30 - extract_patch_features
import torch

def extract_patch_features(encoder_output):
    """Drop the [CLS] token from a ViT encoder output of shape (B, num_patches+1, d_model)."""
    # drop the class token and return only patch feature tokens
    return encoder_output[:,1:,:]

# Step 31 - projector_first_layer
import torch

def projector_first_layer(patch_features, w1, b1):
    # apply the first projector linear layer followed by GELU
    return gelu_activation(patch_features @ w1 + b1)

# Step 32 - projector_second_layer
import torch

def projector_second_layer(hidden, w2, b2):
    """Map hidden activations (N, D_hidden) into the language space (N, D_lang)."""
    # apply the second linear layer of the projector (no activation).
    return hidden @ w2 + b2

# Step 33 - vision_language_projector
import torch

def vision_language_projector(patch_features, params):
    """Map (N, D_vision) patch features to (N, D_lang) image tokens."""
    # chain the two projector layers using params 'w1','b1','w2','b2'.
    w1, b1 = params['w1'], params['b1']
    w2, b2 = params['w2'], params['b2']
    hidden = projector_first_layer(patch_features, w1, b1)
    return projector_second_layer(hidden, w2, b2)

# Step 34 - build_token_vocabulary
def build_token_vocabulary(texts, image_token='<image>', pad_token='<pad>'):
    # Build a whitespace token-to-id vocabulary with pad at 0 and image token at 1.
    vocab = {pad_token: 0, image_token: 1}
    unique_tokens = sorted({token for text in texts for token in text.split()} - vocab.keys())
    vocab.update({token: idx for idx, token in enumerate(unique_tokens, start=len(vocab))})
    return vocab

# Step 35 - encode_text_to_ids
def encode_text_to_ids(text, vocab):
    # split text on whitespace and map each token to its vocab id
    return [vocab[token] for token in text.split()]

# Step 36 - embed_token_ids
import torch

def embed_token_ids(token_ids, embedding_matrix):
    """Look up embedding vectors for each token id.

    Args:
        token_ids: Long tensor of shape (T,) with values in [0, V).
        embedding_matrix: Tensor of shape (V, D_lang).

    Returns:
        Tensor of shape (T, D_lang).
    """
    # select the row of embedding_matrix for each token id
    return embedding_matrix[token_ids]

# Step 37 - add_text_position_embeddings
import torch

def add_text_position_embeddings(text_embeddings, position_embeddings):
    """Add learnable position embeddings to text token embeddings.

    text_embeddings: (T, D_lang)
    position_embeddings: (T_max, D_lang) with T_max >= T
    returns: (T, D_lang)
    """
    # add the first T rows of position_embeddings to text_embeddings
    T = text_embeddings.size(0)
    return text_embeddings + position_embeddings[:T]

# Step 38 - find_image_placeholder_positions
import torch

def find_image_placeholder_positions(token_ids, image_token_id):
    """Return a list of indices where token_ids == image_token_id."""
    # scan token_ids and return every position whose value equals image_token_id
    return torch.nonzero(token_ids == image_token_id).flatten().tolist()

# Step 39 - insert_image_tokens
import torch

def insert_image_tokens(text_embeddings, image_tokens, placeholder_position):
    """Splice image tokens into the text embedding sequence at the placeholder slot."""
    # replace text_embeddings[placeholder_position] with the N image_tokens rows
    return torch.cat((text_embeddings[:placeholder_position], image_tokens, text_embeddings[placeholder_position+1:]))

# Step 40 - build_multimodal_embeddings
import torch

def build_multimodal_embeddings(token_ids, image_tokens, embedding_matrix, position_embeddings, image_token_id):
    # build fused multimodal embeddings by embedding text, adding positions, and splicing image tokens.
    text_embeddings = embed_token_ids(token_ids, embedding_matrix)
    text_embeddings = add_text_position_embeddings(text_embeddings, position_embeddings)
    placeholder_positions = find_image_placeholder_positions(token_ids, image_token_id)
    for placeholder_position in placeholder_positions:
        text_embeddings = insert_image_tokens(text_embeddings, image_tokens, placeholder_position)
    return text_embeddings

# Step 41 - build_label_tensor
import torch

def build_label_tensor(token_ids, image_token_id, pad_token_id, num_image_tokens, ignore_index=-100):
    """Build the label tensor aligned to the fused multimodal sequence."""
    # expand image placeholders, mask image and pad positions with ignore_index
    repeats = torch.where(token_ids == image_token_id, num_image_tokens, 1)
    label = token_ids.repeat_interleave(repeats)
    label = torch.where(
        (label == image_token_id) | (label == pad_token_id),
        ignore_index, label
    )
    return label

# Step 42 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    """Return a (seq_len, seq_len) additive causal mask: 0 on/under diag, -inf above."""
    # build a lower-triangular additive mask with 0 allowed and -inf blocked
    return torch.full((seq_len, seq_len), -torch.inf).triu(1)

# Step 43 - decoder_block
def decoder_block(x, params, causal_mask):
    # run a pre-norm masked self-attention sublayer then a pre-norm MLP sublayer over x.
    x = x.unsqueeze(0)

    sublayer_mha = lambda x : multi_head_self_attention(x, params['attn'], params['num_heads'], mask=causal_mask)
    gamma1, beta1 = params['ln1']['gamma'], params['ln1']['beta']
    y = pre_norm_sublayer(x, gamma1, beta1, sublayer_mha)

    sublayer_mlp = lambda x : mlp_block(x, params['mlp'])
    gamma2, beta2 = params['ln2']['gamma'], params['ln2']['beta']
    z = pre_norm_sublayer(y, gamma2, beta2, sublayer_mlp)

    return z.squeeze(0)

# Step 44 - language_model_decoder
import torch

def language_model_decoder(x, blocks_params, causal_mask):
    # apply every decoder block in blocks_params sequentially to x and return the result
    for block in blocks_params:
        x = decoder_block(x, block, causal_mask)
    return x

# Step 45 - final_layer_norm
import torch

def final_layer_norm(x, gamma, beta):
    # apply the existing layer_norm primitive to x using gamma and beta.
    return layer_norm(x, gamma, beta, eps=1e-5)

# Step 46 - language_model_head
def language_model_head(x, w_out, b_out):
    # project hidden states (L, D) to vocabulary logits (L, V) using w_out and b_out
    return x @ w_out + b_out

# Step 47 - encode_image_to_tokens
def encode_image_to_tokens(image, vision_params, projector_params):
    # run the vision encoder, drop the class token, and apply the projector.
    patch_sequence = image
    encoder_params = vision_params
    encoder_output = vision_encoder(patch_sequence, encoder_params, vision_params['num_heads'])
    patch_features = extract_patch_features(encoder_output)
    params = projector_params
    return vision_language_projector(patch_features, params)

# Step 48 - vision_language_forward
def vision_language_forward(image, token_ids, params):
    C = image.shape[1]
    patch_dim = params['vision']['patch_proj_weight'].shape[1]
    patch_size = int((patch_dim // C) ** 0.5)

    patches = split_image_into_patches(image, params['vision']['patch_size'])
    flat_patches = flatten_patches(patches)
    patch_embeddings = project_patches_to_embeddings(flat_patches, params['vision']['patch_proj_weight'], params['vision']['patch_proj_bias'])
    patch_embeddings = prepend_class_token(patch_embeddings, params['vision']['class_token'])
    patch_seq = add_position_embeddings(image_tokens, params['vision']['position_embeddings'])

    image_tokens = encode_image_to_tokens(patch_seq, params['vision'], params['projector'])
    fused_embeds = build_multimodal_embeddings(token_ids, image_tokens, params['embedding'], params['pos_embedding'], params['image_token_id'])
    
    seq_len = fused_embeds.shape[1]
    causal_mask = build_causal_mask(seq_len)

    hidden = language_model_decoder(fused_embeds, params['decoder_blocks'], causal_mask)

    hidden_norm = final_layer_norm(
        hidden, 
        params['final_ln']['gamma'], 
        params['final_ln']['beta']
    )

    logits = language_model_head(
        hidden_norm, 
        params['lm_head']['w_out'], 
        params['lm_head']['b_out']
    )
    return logits

# Step 49 - shift_logits_and_labels
import torch

def shift_logits_and_labels(logits, labels):
    # align each logit with the next-position label and return (shifted_logits, shifted_labels).
    return logits[:-1], labels[1:]

# Step 50 - per_position_cross_entropy
import torch

def per_position_cross_entropy(shifted_logits, shifted_labels, ignore_index=-100):
    """Per-position next-token cross-entropy with 0 at ignored positions."""
    # log-softmax over vocab, gather target log-probs, zero out ignored positions
    valid_mask = (shifted_labels != ignore_index)
    safe_labels = shifted_labels.masked_fill(~valid_mask, 0)
    log_probs = torch.log_softmax(shifted_logits, dim=-1)
    target_log_probs = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    cross_entropy = -target_log_probs
    per_position_loss = cross_entropy.masked_fill(~valid_mask, 0.0)
    return per_position_loss

# Step 51 - masked_mean_loss
import torch

def masked_mean_loss(per_position_losses, shifted_labels, ignore_index=-100):
    """Average per-position losses over positions whose label != ignore_index."""
    # average per_position_losses over positions where shifted_labels != ignore_index
    mask = shifted_labels != ignore_index
    valid_count = mask.sum()
    if valid_count == 0:
        return torch.tensor(0.0)
    masked_losses = per_position_losses * mask.float()
    total_loss = masked_losses.sum()
    return total_loss / valid_count

# Step 52 - greedy_next_token
def greedy_next_token(logits):
    # return the int token id with the highest logit at the final position
    return torch.argmax(logits[-1]).item()

# Step 53 - apply_temperature
import torch

def apply_temperature(logits, temperature):
    """Scale logits by dividing by temperature."""
    # return a tensor of logits rescaled by the temperature value
    return logits / temperature

# Step 54 - top_k_filter
import torch

def top_k_filter(logits, k):
    """Keep only the top-k logits; set all others to -inf."""
    # keep top-k logits, replace the rest with -inf
    if k == 0: return logits
    k = min(k, logits.size(-1))
    values, indices = torch.topk(logits, logits.size(-1)-k, largest=False)
    logits[indices] = -torch.inf
    return logits

# Step 55 - sample_from_logits
import torch

def sample_from_logits(logits):
    """Sample a token id from softmax(logits).

    Args:
        logits: 1D tensor of shape (V,)
    Returns:
        int token id
    """
    # turn logits into a categorical distribution and draw one token id
    probs = torch.softmax(logits, -1)
    return torch.multinomial(probs, 1).item()

# Step 56 - generate_caption (not yet solved)
# TODO: implement

# Step 57 - initialize_vlm_parameters
def initialize_vlm_parameters(config, seed=0):
    # build the full params dict with leaf tensors requiring grad for every component.
    torch.manual_seed(seed)

    def make_param(shape, init='normal'):
        param = torch.empty(shape, dtype=torch.float32)
        if init == 'normal':
            param.normal_(mean=0.0, std=0.02)
        elif init == 'ones':
            param.fill_(1.0)
        elif init == 'zeros':
            param.zero_()
        param.requires_grad = True
        return param

    image_size = config['image_size']
    patch_size = config['patch_size']
    num_patches = config['num_patches']
    in_channels = config['in_channels']
    d_vision = config['d_vision']
    d_lang = config['d_text']  # maps d_text configuration to language embedding space
    num_vision_heads = config['num_vision_heads']
    num_decoder_heads = config['num_decoder_heads']
    n_layers_vision = config['num_vision_layers']
    n_layers_decoder = config['num_decoder_layers']
    mlp_hidden_vision = config['mlp_hidden_vision']
    mlp_hidden_text = config['mlp_hidden_text']
    vocab_size = config['vocab_size']
    max_seq_len = config['max_text_len']
    num_image_tokens = config['num_image_tokens']
    
    patch_dim = in_channels * patch_size * patch_size

    # 1. Vision Parameters
    vision_blocks = []
    for _ in range(n_layers_vision):
        block = {
            'attn': {
                'wq': make_param((d_vision, d_vision)), 'bq': make_param((d_vision,), 'zeros'),
                'wk': make_param((d_vision, d_vision)), 'bk': make_param((d_vision,), 'zeros'),
                'wv': make_param((d_vision, d_vision)), 'bv': make_param((d_vision,), 'zeros'),
                'wo': make_param((d_vision, d_vision)), 'bo': make_param((d_vision,), 'zeros'),
            },
            'mlp': {
                'w1': make_param((mlp_hidden_vision, d_vision)), 'b1': make_param((mlp_hidden_vision,), 'zeros'),
                'w2': make_param((d_vision, mlp_hidden_vision)), 'b2': make_param((d_vision,), 'zeros'),
            },
            'ln1_gamma': make_param((d_vision,), 'ones'), 'ln1_beta': make_param((d_vision,), 'zeros'),
            'ln2_gamma': make_param((d_vision,), 'ones'), 'ln2_beta': make_param((d_vision,), 'zeros'),
        }
        vision_blocks.append(block)
        
    vision = {
        'patch_proj_weight': make_param((d_vision, patch_dim)),
        'patch_proj_bias': make_param((d_vision,), 'zeros'),
        'class_token': make_param((1, 1, d_vision)),
        'position_embeddings': make_param((1, num_patches + 1, d_vision)),
        'blocks': vision_blocks,
        'final_ln_gamma': make_param((d_vision,), 'ones'),
        'final_ln_beta': make_param((d_vision,), 'zeros'),
        'patch_size': patch_size,
    }

    # 2. Projector Parameters
    projector = {
        'w1': make_param((d_vision, d_lang)), 
        'b1': make_param((d_lang,), 'zeros'),
        'w2': make_param((d_lang, d_lang)),   
        'b2': make_param((d_lang,), 'zeros'),
    }

    # 3. Embedding
    embedding = make_param((vocab_size, d_lang))

    # 4. Pos_embedding
    pos_embedding = make_param((max_seq_len, d_lang))

    # 5. Decoder_blocks
    decoder_blocks = []
    for _ in range(n_layers_decoder):
        block = {
            'attn': {
                'wq': make_param((d_lang, d_lang)), 'bq': make_param((d_lang,), 'zeros'),
                'wk': make_param((d_lang, d_lang)), 'bk': make_param((d_lang,), 'zeros'),
                'wv': make_param((d_lang, d_lang)), 'bv': make_param((d_lang,), 'zeros'),
                'wo': make_param((d_lang, d_lang)), 'bo': make_param((d_lang,), 'zeros'),
            },
            'mlp': {
                'w1': make_param((mlp_hidden_text, d_lang)), 'b1': make_param((mlp_hidden_text,), 'zeros'),
                'w2': make_param((d_lang, mlp_hidden_text)), 'b2': make_param((d_lang,), 'zeros'),
            },
            'ln1': {'gamma': make_param((d_lang,), 'ones'), 'beta': make_param((d_lang,), 'zeros')},
            'ln2': {'gamma': make_param((d_lang,), 'ones'), 'beta': make_param((d_lang,), 'zeros')},
            'num_heads': num_decoder_heads,
        }
        decoder_blocks.append(block)

    # 6. Final_ln
    final_ln = {
        'gamma': make_param((d_lang,), 'ones'),
        'beta': make_param((d_lang,), 'zeros'),
    }
    
    # 7. Lm_head
    lm_head = {
        'w_out': make_param((d_lang, vocab_size)), 
        'b_out': make_param((vocab_size,), 'zeros'),
    }

    return {
        'vision': vision,
        'projector': projector,
        'embedding': embedding,
        'pos_embedding': pos_embedding,
        'decoder_blocks': decoder_blocks,
        'final_ln': final_ln,
        'lm_head': lm_head,
        'image_token_id': config.get('image_token_id', 1),
        'num_image_tokens': num_image_tokens,
    }

# Step 58 - collect_parameters (not yet solved)
# TODO: implement

# Step 59 - zero_gradients (not yet solved)
# TODO: implement

# Step 60 - training_step (not yet solved)
# TODO: implement

# Step 61 - apply_gradient_update (not yet solved)
# TODO: implement

# Step 62 - run_training_loop (not yet solved)
# TODO: implement


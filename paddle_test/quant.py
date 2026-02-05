import enum
from typing import Tuple

import torch
import paddle

class FP8KVCacheLayout(enum.Enum):
    V32_FP8Sparse = 1
    MODEL1_FP8Sparse = 2

    def get_meta(self) -> Tuple[int, int, int, int, int]:
        # Return: (d, d_nope, d_rope, tile_size, num_tiles)
        return {
            FP8KVCacheLayout.V32_FP8Sparse: (576, 512, 64, 128, 4),
            FP8KVCacheLayout.MODEL1_FP8Sparse: (512, 448, 64, 64, 7)
        }[self]

def _cast_scale_inv_to_ue8m0(scales_inv: torch.Tensor, out_dtype = torch.float32) -> torch.Tensor:
    return 2 ** torch.clip(scales_inv, 1e-4).log2().ceil().to(out_dtype)

def _float_to_e8m0_bytes(scales: torch.Tensor) -> torch.Tensor:
    """将 float32 scale factor 编码为 e8m0 字节格式（uint8）
    e8m0 格式: value = 2^(exp - 127), 其中 exp 是 0-255 的无符号整数
    注意：输入 scales 应为 2 的幂次方，使用 round() 容忍浮点误差
    """
    exp = torch.log2(scales).round().to(torch.int32) + 127
    return exp.clamp(0, 255).to(torch.uint8)

def _e8m0_bytes_to_float(bytes_tensor: torch.Tensor) -> torch.Tensor:
    """将 e8m0 字节格式（uint8）解码为 float32
    e8m0 格式: value = 2^(exp - 127)
    """
    exp = bytes_tensor.to(torch.int32) - 127
    return (2.0 ** exp.to(torch.float32))

def quantize_k_cache(
    input_k_cache: torch.Tensor,    # (num_blocks, block_size, h_k, d)
    kvcache_layout: FP8KVCacheLayout,
) -> torch.Tensor:
    """
    Quantize the k-cache
    For more detail about the layout of K/V, please refer to comments in flash_mla_interface.py
    """
    d, d_nope, d_rope, tile_size, num_tiles = kvcache_layout.get_meta()
    assert input_k_cache.shape[-1] == d
    num_blocks, block_size, h_k, _ = input_k_cache.shape
    assert h_k == 1
    input_k_cache = input_k_cache.squeeze(2)    # [num_blocks, block_size, d]
    input_elem_size = input_k_cache.element_size()

    if kvcache_layout == FP8KVCacheLayout.V32_FP8Sparse:       
        # === 新实现：分别计算三部分后直接 concat，避免 float8_e4m3fn 切片赋值 ===
        # 注意：原实现切片赋值有自动广播，新实现需显式 broadcast_to 保证形状一致
        quantized_nope_tiles = []
        scale_factor_tiles = []
        for tile_idx in range(0, num_tiles):
            cur_scale_factors_inv = torch.abs(input_k_cache[..., tile_idx*tile_size:(tile_idx+1)*tile_size]).max(axis=-1).float() / 448.0
            cur_scale_factors_inv = _cast_scale_inv_to_ue8m0(cur_scale_factors_inv)
            cur_scale_factors_inv = torch.broadcast_to(cur_scale_factors_inv, (num_blocks, block_size)).contiguous()
            scale_factor_tiles.append(cur_scale_factors_inv)  # [num_blocks, block_size]
            
            cur_scale_factors_inv_expanded = cur_scale_factors_inv.unsqueeze(-1)  # [num_blocks, block_size, 1]
            cur_quantized_nope = (input_k_cache[..., tile_idx*tile_size:(tile_idx+1)*tile_size].float() / cur_scale_factors_inv_expanded.float()).to(torch.float8_e4m3fn)
            cur_quantized_nope = torch.broadcast_to(cur_quantized_nope, (num_blocks, block_size, tile_size)).contiguous()
            quantized_nope_tiles.append(cur_quantized_nope)  # [num_blocks, block_size, tile_size]
        
        # nope_part: [num_blocks, block_size, d_nope] as float8_e4m3fn
        nope_part = torch.cat(quantized_nope_tiles, dim=-1)
        
        # scale_factor_part: [num_blocks, block_size, num_tiles] as float32 -> [num_blocks, block_size, num_tiles*4] as float8_e4m3fn
        scale_factor_part = torch.stack(scale_factor_tiles, dim=-1)
        scale_factor_as_bytes = scale_factor_part.view(torch.float8_e4m3fn)
            
        # rope_part: [num_blocks, block_size, d_rope] as bfloat16 -> [num_blocks, block_size, d_rope*2] as float8_e4m3fn
        rope_part = input_k_cache[..., d_nope:].contiguous()
        rope_as_bytes = rope_part.view(torch.float8_e4m3fn)
        
        # concat: [num_blocks, block_size, d_nope + num_tiles*4 + d_rope*input_elem_size]
        # 内存布局: [nope: d_nope bytes (fp8)] [scale: num_tiles*4 bytes (fp32)] [rope: d_rope*2 bytes (bf16)]
        actual_blocks = torch.cat([nope_part, scale_factor_as_bytes, rope_as_bytes], dim=-1)
        
        # padding: 多分配一行用于 CUDA kernel 的内存安全（防止越界读取）
        bytes_per_token = d_nope + num_tiles*4 + input_elem_size*d_rope
        padding_row = torch.empty((num_blocks, 1, bytes_per_token), dtype=torch.float8_e4m3fn, device=actual_blocks.device)
        result_full = torch.cat([actual_blocks, padding_row], dim=1)
        result = result_full[:, :block_size, :]  # 只返回 block_size 行，但底层内存多一行
        
        result = result.view(num_blocks, block_size, 1, -1)
        return result
    
    elif kvcache_layout == FP8KVCacheLayout.MODEL1_FP8Sparse:
        # === 新实现：分别计算各部分后 concat，避免 float8_e4m3fn 切片赋值 ===
        # 内存布局: [nope: d_nope bytes (fp8)] [rope: 2*d_rope bytes (bf16)] [scale: num_tiles bytes (e8m0/uint8)] [padding: 1 byte]
        
        # 1. 计算量化后的 nope 部分
        quantized_nope_tiles = []
        scale_factor_tiles = []
        for tile_idx in range(0, num_tiles):
            cur_scale_factors_inv = torch.abs(input_k_cache[..., tile_idx*tile_size:(tile_idx+1)*tile_size]).max(axis=-1).float() / 448.0
            cur_scale_factors_inv = _cast_scale_inv_to_ue8m0(cur_scale_factors_inv)
            cur_scale_factors_inv = torch.broadcast_to(cur_scale_factors_inv, (num_blocks, block_size)).contiguous()
            # e8m0 scale 编码为 uint8
            scale_factor_tiles.append(_float_to_e8m0_bytes(cur_scale_factors_inv))  # [num_blocks, block_size]
            
            cur_scale_factors_inv_expanded = cur_scale_factors_inv.unsqueeze(-1)
            cur_quantized_nope = (input_k_cache[..., tile_idx*tile_size:(tile_idx+1)*tile_size].float() / cur_scale_factors_inv_expanded.float()).to(torch.float8_e4m3fn)
            cur_quantized_nope = torch.broadcast_to(cur_quantized_nope, (num_blocks, block_size, tile_size)).contiguous()
            quantized_nope_tiles.append(cur_quantized_nope)  # [num_blocks, block_size, tile_size]
        
        # nope_part: [num_blocks, block_size, d_nope] as float8_e4m3fn
        nope_part = torch.cat(quantized_nope_tiles, dim=-1)
        
        # 2. rope 部分: [num_blocks, block_size, d_rope] as bfloat16 -> [num_blocks, block_size, 2*d_rope] as float8_e4m3fn
        rope_part = input_k_cache[..., d_nope:].contiguous()
        rope_as_bytes = rope_part.view(torch.float8_e4m3fn)
        
        # 3. scale 部分: [num_blocks, block_size, num_tiles] as uint8 -> view as float8_e4m3fn
        scale_part = torch.stack(scale_factor_tiles, dim=-1)  # [num_blocks, block_size, num_tiles] uint8
        scale_as_bytes = scale_part.view(torch.float8_e4m3fn)
        
        # 4. padding: 1 byte
        padding_part = torch.zeros((num_blocks, block_size, 1), dtype=torch.float8_e4m3fn, device=input_k_cache.device)
        
        # 5. concat: [num_blocks, block_size, d_nope + 2*d_rope + num_tiles + 1]
        result = torch.cat([nope_part, rope_as_bytes, scale_as_bytes, padding_part], dim=-1)
        result = result.view(num_blocks, block_size, 1, -1)
        return result

    else:
        raise NotImplementedError(f"Unsupported kvcache_layout: {kvcache_layout}")
    

def dequantize_k_cache(
    quant_k_cache: torch.Tensor,    # (num_blocks, block_size, 1, bytes_per_token)
    kvcache_layout: FP8KVCacheLayout,
) -> torch.Tensor:
    """
    De-quantize the k-cache
    """
    d, d_nope, d_rope, tile_size, num_tiles = kvcache_layout.get_meta()
    num_blocks, block_size, h_k, _ = quant_k_cache.shape
    assert h_k == 1
    result = torch.empty((num_blocks, block_size, d), dtype=torch.bfloat16, device=quant_k_cache.device)

    if kvcache_layout == FP8KVCacheLayout.V32_FP8Sparse:
        quant_k_cache = quant_k_cache.view(num_blocks, block_size, -1)

        input_nope = quant_k_cache[..., :d_nope]
        input_scale = quant_k_cache[..., d_nope:d_nope + num_tiles*4].contiguous().view(torch.uint8).view(torch.float32)
        input_rope = quant_k_cache[..., d_nope + num_tiles*4:].contiguous().view(torch.uint8).view(torch.bfloat16)
        result[..., d_nope:] = input_rope

        for tile_idx in range(0, num_tiles):
            cur_nope = input_nope[..., tile_idx*tile_size:(tile_idx+1)*tile_size].to(torch.float32)
            cur_scales = input_scale[..., tile_idx].unsqueeze(-1)
            result[..., tile_idx*tile_size:(tile_idx+1)*tile_size] = cur_nope * cur_scales

    elif kvcache_layout == FP8KVCacheLayout.MODEL1_FP8Sparse:
        # 新布局: [nope: d_nope bytes] [rope: 2*d_rope bytes] [scale: num_tiles bytes] [padding: 1 byte]
        quant_k_cache = quant_k_cache.view(num_blocks, block_size, -1)
        
        # 解析各部分
        input_nope = quant_k_cache[..., :d_nope]  # [num_blocks, block_size, d_nope] float8
        input_rope = quant_k_cache[..., d_nope:d_nope + 2*d_rope].contiguous().view(torch.uint8).view(torch.bfloat16)  # [num_blocks, block_size, d_rope] bf16
        input_scale_bytes = quant_k_cache[..., d_nope + 2*d_rope:d_nope + 2*d_rope + num_tiles].contiguous().view(torch.uint8)  # [num_blocks, block_size, num_tiles] uint8
        
        result[..., d_nope:] = input_rope
        for tile_idx in range(0, num_tiles):
            cur_nope = input_nope[..., tile_idx*tile_size:(tile_idx+1)*tile_size].to(torch.bfloat16)
            # 使用手动解码函数将 e8m0 字节转为 float32，再转 bfloat16
            cur_scales = _e8m0_bytes_to_float(input_scale_bytes[..., tile_idx]).to(torch.bfloat16).unsqueeze(-1)
            result[..., tile_idx*tile_size:(tile_idx+1)*tile_size] = cur_nope * cur_scales
            
    else:
        raise NotImplementedError(f"Unsupported kvcache_layout: {kvcache_layout}")
    
    result = result.view(num_blocks, block_size, 1, d)
    return result


def abs_indices2indices_in_kvcache(
    abs_indices: torch.Tensor,  # [b, s_q, topk]
    block_table: torch.Tensor,  # [b, /]
    block_size: int,
) -> torch.Tensor:
    """
    Convert abs_indices (logical index, ranging from 0 to s_k-1) to index expected by the sparse attn kernel
    Equivalent to:
    
    b, s_q, topk = abs_indices.shape
    indices_in_kvcache = torch.empty_like(abs_indices)
    for i in range(b):
        cur_abs_indices = abs_indices[i, :, :].clone()  # [s_q, topk]
        invalid_mask = cur_abs_indices == -1
        cur_abs_indices[invalid_mask] = 0
        cur_indices_in_kvcache = block_table[i].index_select(0, cur_abs_indices.flatten()//block_size).view(s_q, topk)*block_size + cur_abs_indices%block_size
        cur_indices_in_kvcache[invalid_mask] = -1
        indices_in_kvcache[i] = cur_indices_in_kvcache
    return indices_in_kvcache

    """
    b, s_q, topk = abs_indices.shape
    _, max_blocks_per_seq = block_table.shape

    abs_indices = abs_indices.clone()
    invalid_mask = abs_indices == -1
    abs_indices[invalid_mask] = 0

    real_block_idxs = block_table.view(-1).index_select(0, (abs_indices//block_size + torch.arange(0, b, dtype=abs_indices.dtype, device=abs_indices.device).view(b, 1, 1)*max_blocks_per_seq).view(-1))
    indices_in_kvcache = real_block_idxs.view(b, s_q, topk)*block_size + abs_indices%block_size
    indices_in_kvcache[invalid_mask] = -1

    return indices_in_kvcache
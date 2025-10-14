# MLM Mask 功能使用示例

## 基本用法

### 1. 简单的随机mask
```python
# 在训练步骤中使用基本的MLM mask
batch = self._random_mlm_mask(batch, mask_ratio=0.15)  # 15%的token被mask
```

### 2. 高级MLM mask策略
```python
# 使用高级MLM mask，支持不同的mask策略
batch = self._advanced_mlm_mask(
    batch, 
    mask_ratio=0.15,      # 15%的token被选中进行mask
    replace_prob=0.8,     # 80%的被选中token用[MASK]替换
    random_prob=0.1,      # 10%的被选中token用随机氨基酸替换
    keep_prob=0.1         # 10%的被选中token保持原样
)
```

## 新功能特性

### `masked_position` 标记
- **功能**: 标记哪些位置被mask了，用于精确计算MLM损失
- **类型**: `torch.Tensor` (bool类型)
- **形状**: `[batch_size, seq_length]`
- **用途**: 在loss计算时只对被mask的位置计算损失

### `use_dino` 参数
- **功能**: 指示当前训练模式，控制loss计算策略
- **值**: 
  - `0`: 正常模式，在所有有效位置计算loss
  - `1`: DINO模式
  - `2`: 对比学习模式
  - `3`: MLM模式，只在被mask的位置计算loss

## 功能特点

### `_random_mlm_mask` 函数
- **功能**: 对蛋白质序列进行简单的随机mask
- **参数**:
  - `batch`: 包含蛋白质序列数据的批次字典
  - `mask_ratio`: mask的比例，默认0.15 (15%)
- **特点**:
  - 只对有效token进行mask（排除padding）
  - 将选中的token替换为[MASK] token (ID=21)
  - 可选地将对应的结构信息置零
  - **新增**: 添加`masked_position`标记被mask的位置

### `_advanced_mlm_mask` 函数
- **功能**: 高级MLM mask，支持多种mask策略
- **参数**:
  - `batch`: 包含蛋白质序列数据的批次字典
  - `mask_ratio`: mask的比例，默认0.15 (15%)
  - `replace_prob`: 用[MASK]替换的比例，默认0.8
  - `random_prob`: 用随机氨基酸替换的比例，默认0.1
  - `keep_prob`: 保持原token的比例，默认0.1
- **特点**:
  - 支持三种mask策略：替换、随机、保持
  - 概率和必须为1.0
  - 更符合BERT-style的MLM训练
  - **新增**: 添加`masked_position`标记所有被选中的位置（无论采用哪种策略）

## Loss计算逻辑

### MLM模式 (use_dino=3)
```python
# 在foldrep_model.py中的compute_custom_loss方法
if use_dino == 3:
    # MLM模式：只在被mask的位置计算loss
    masked_position = batch.get('masked_position', None)
    if masked_position is not None:
        mask = masked_position & (chain > 0)  # 被mask的位置且有效
    else:
        # 如果没有masked_position，回退到原始逻辑
        mask = chain > 0
else:
    # 非MLM模式：在所有有效位置计算loss
    mask = chain > 0
```

### 优势
- **精确性**: 只在被mask的位置计算loss，避免信息泄露
- **效率**: 减少不必要的loss计算
- **灵活性**: 支持回退机制，确保兼容性

## 在训练中的集成

### 当前实现
在 `training_step` 方法中，当 `self.use_dino==3` 时，会使用MLM baseline：

```python
elif self.use_dino==3: # mlm baseline
    batch = self._advanced_mlm_mask(batch, mask_ratio=0.15)
    batch['use_dino'] = self.use_dino  # 传递use_dino参数
```

### 自定义使用
你可以根据需要选择使用哪种mask策略：

```python
# 使用基本mask
batch = self._random_mlm_mask(batch, mask_ratio=0.2)

# 使用高级mask
batch = self._advanced_mlm_mask(
    batch, 
    mask_ratio=0.15,
    replace_prob=0.7,  # 70%用[MASK]
    random_prob=0.2,   # 20%用随机氨基酸
    keep_prob=0.1      # 10%保持原样
)

# 设置MLM模式
batch['use_dino'] = 3
```

## 注意事项

1. **设备兼容性**: 函数会自动处理GPU/CPU设备
2. **内存效率**: 使用clone()创建副本，避免修改原始数据
3. **边界处理**: 正确处理空序列和极小序列
4. **有效性检查**: 只对有效token（data_id > 0）进行mask操作
5. **结构信息**: 可选地将对应的结构信息（blocks）置零
6. **Loss计算**: `masked_position`确保只在被mask的位置计算loss
7. **模式控制**: `use_dino`参数控制loss计算策略

## 测试验证

所有功能都经过了完整的测试，包括：
- 基本mask功能测试
- 高级mask策略测试  
- 边界情况测试（空序列、极小序列）
- 概率分布验证
- **新增**: masked_position标记测试
- **新增**: use_dino参数测试
- **新增**: loss计算逻辑测试
- **新增**: 完整集成测试

测试结果显示功能工作正常，mask比例准确，策略分布符合预期，loss计算逻辑正确。


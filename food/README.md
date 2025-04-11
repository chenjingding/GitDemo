## Vocabulary 类  

1. `mask_token` 对应的索引通过调用 `add_token` 方法赋值给 `self.mask_index` 属性。  
2. `lookup_token` 方法中，如果 `self.unk_index >= 0`，则对未登录词返回 `self.unk_index`。  
3. 调用 `add_many` 方法添加多个 token 时，实际是通过循环调用 `add_token` 方法实现的。  

## CBOWVectorizer 类  

1. `vectorize` 方法中，当 `vector_length < 0` 时，最终向量长度等于 `vocabulary` 的长度。  
2. `from_dataframe` 方法构建词表时，会遍历 DataFrame 中 `输入` 和 `输出` 两列的内容。  
3. `out_vector[len(indices):]` 的部分填充为 `self.cbow_vocab.padding_idx`。  

## CBOWDataset 类  

1. `_max_seq_length` 通过计算所有 context 列的 `长度` 的最大值得出。  
2. `set_split` 方法通过 `self._lookup_dict` 选择对应的 `训练集` 和 `验证集`。  
3. `__getitem__` 返回的字典中，`y_target` 通过查找 `目标` 列的 token 得到。  

## 模型结构  

1. `CBOWClassifier` 的 `forward` 中，`x_embedded_sum` 的计算方式是 `embedding(x_in).sum(dim=1)`。  
2. 模型输出层 `fc1` 的 `out_features` 等于 `target_size` 参数的值。  

## 训练流程  

1. `generate_batches` 函数通过 PyTorch 的 `DataLoader` 类实现批量加载。  
2. 训练时 `classifier.train()` 的作用是启用 `训练` 和 `反向传播` 模式。  
3. 反向传播前必须执行 `optimizer.zero_grad()` 清空梯度。  
4. `compute_accuracy` 中 `y_pred_indices` 通过 `torch.argmax` 方法获取预测类别。  

## 训练状态管理  

1. `make_train_state` 中 `early_stopping_best_val` 初始化为 `float('inf')`。  
2. `update_train_state` 在连续 `patience` 次验证损失未下降时会触发早停。  
3. 当验证损失下降时，`early_stopping_step` 会被重置为 `0`。  

## 设备与随机种子  

1. `set_seed_everywhere` 中与 CUDA 相关的设置是 `torch.cuda.manual_seed_all(seed)`。  
2. `args.device` 的值根据 `torch.cuda.is_available()` 确定。  

## 推理与测试  

1. `get_closest` 函数中排除计算的目标词本身是通过 `continue` 判断 `word == target_word` 实现的。  
2. 测试集评估时一定要调用 `model.eval()` 方法禁用 dropout。  

## 关键参数  

1. `CBOWClassifier` 的 `padding_idx` 参数默认值为 `0`。  
2. `args` 中控制词向量维度的参数是 `embedding_dim`。  
3. 学习率调整策略 `ReduceLROnPlateau` 的触发条件是验证损失 `增加`。  

---  

请根据具体项目和需求对上述内容进行相应的修改和补充。将这个内容保存为 `README.md` 文件后，您可以将其与代码仓库一起提交以供他人参考。如果您还有其他问题或需要进一步的帮助，请告诉我！
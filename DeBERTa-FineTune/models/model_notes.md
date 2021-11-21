# **模型加载**

- 调用 [MODEL]For[TASK].from_pretrained([PATH_OR_NAME])
- 调用父类 PretrainedModel.from_pretrained()
  - 调用 MODEL 的 config_class 的 from_pretrained() 得到 config(PretrainedConfig 实例) 和 model_kwargs,
    实质是调用 config_class 的 get_config_dict() 方法
  - 开始 load 模型权重，然后设为 eval() 模式返回

# **模型前向过程**
ie. DeBERTaForSequenceClassification
  i. 将输入序列送入到 pre-trained 的 DeBERTa backbone 中得到输出 outputs；
  ii. 将 Encoder 输出(outputs[0]) 送入到 ContextPooler 后再经过 Dropout 得到 pooler_output；
  iii. 最后将 pooler_output 送入分类头部 classifier 输出 logits；
  iv. 若 输入中含有 labels，则计算loss(否则，loss 设置为 None):
    1). 若 self.num_labels == 1, 则说明是回归任务，利用 MSE 计算 loss;
    2). 否则，若 labels 是一维的 或者 最后一维是 1，那么对 label 不小于0的样本计算 CE loss(若所有样本的 label 均小于0，则不计算 loss);
    3). 否则，用 label * -logsoftmax(logit) 计算支持多标签的分类损失；
  v. 返回输出：
    1). 若输入里指定了 return_dict= = False，则返回 (loss,logits,outputs[1:]) 元组(若 loss 为 None，则没有 loss 项)；
    2). 否则，返回 SequenceClassifierOutput 实例，包含 loss(若有的话)，logits, hidden_states, attentions。其中，后两项在调用前向过程中若指定了 output_hidden_states = True, output_attentions = True 才会有。这4项可以通过该实例的属性(如 .logits, .loss, .hidden_states, .attentions) 获取


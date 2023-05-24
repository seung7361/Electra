import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import ElectraTokenizerFast, ElectraForSequenceClassification
import time
import deepspeed

batch_size = 2048
learning_rate = 2e-5
warmup_steps = 100
weight_decay = 1e-2

tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-base-discriminator')
model = ElectraForSequenceClassification.from_pretrained('google/electra-base-discriminator')

total_params = sum(p.numel() for p in model.parameters())
print('total params: {:_}'.format(total_params))

# dataset = load_dataset('glue', 'sst2')

# train_dataset = []
# for data in tqdm(dataset['train']):
#     token = tokenizer(data['sentence'], return_tensors='pt', padding='max_length', max_length=64, truncation=True)

#     train_dataset.append({
#         'input_ids': token.input_ids.squeeze(0),
#         'attention_mask': token.attention_mask.squeeze(0),
#         'label': data['label']
#     })
train_dataset = torch.load('train_dataset.pt')

deepspeed.init_distributed()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

model_engine, optimizer, train_dataloader, _ = deepspeed.initialize(model=model,
                                                                    model_parameters=model.parameters(),
                                                                    optimizer=optimizer,
                                                                    training_data=train_dataset,
                                                                    config='ds_config.json')


loss_fn = torch.nn.CrossEntropyLoss()

step = 0

num_epochs = 5
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1} started.")

    for batch in tqdm(train_dataloader):
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        labels = batch["label"].cuda()

        # gradient

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        model_engine.backward(loss)
        model_engine.step()

    print(f"Epoch {epoch + 1} completed.")
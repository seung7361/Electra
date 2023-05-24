import numpy as np
import torch
from datasets import load_dataset, load_metric
from transformers import ElectraTokenizerFast, ElectraForSequenceClassification
from transformers import get_linear_schedule_with_warmup

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 256
learning_rate = 2e-5
warmup_steps = 100
weight_decay = 1e-2

tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-base-discriminator')
model = ElectraForSequenceClassification.from_pretrained('google/electra-base-discriminator').to(device)

total_params = sum(p.numel() for p in model.parameters())
print("{:_}".format(total_params)) # 13_549_314, 13 million,천3백만
dataset = load_dataset('glue', 'sst2')


from tqdm import tqdm

# train_dataset = []
# for data in tqdm(dataset['train']):
#     token = tokenizer(data['sentence'], return_tensors='pt', padding='max_length', max_length=64, truncation=True)

#     train_dataset.append({
#         'input_ids': token.input_ids.squeeze(0),
#         'attention_mask': token.attention_mask.squeeze(0),
#         'label': data['label']
#     })

# torch.save(train_dataset, 'train_dataset.pt')
train_dataset = torch.load('train_dataset.pt')

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print(train_dataset[0])

optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

for epoch in range(5):
    print(f"Epoch {epoch + 1} started.")

    for batch in tqdm(train_dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # gradient 
        model.zero_grad()
        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    print(f"Epoch {epoch + 1} completed.")
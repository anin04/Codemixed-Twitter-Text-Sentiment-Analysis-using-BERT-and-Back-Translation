import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import pandas as pd
import time
import numpy as np
import os

# Fungsi untuk memformat waktu
def format_time(elapsed):
    return str(time.strftime("%H:%M:%S", time.gmtime(elapsed)))

# Fungsi untuk menghitung akurasi
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Fungsi untuk menyimpan model
def save_checkpoint(filepath, model, best_valid_loss):
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_valid_loss': best_valid_loss
    }, filepath)

# Fungsi untuk menyimpan metrik
def save_metrics(filepath, metrics):
    torch.save(metrics, filepath)

# Fungsi untuk menyimpan data training ke CSV
def save_training_data_to_csv(filepath, data):
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)

# Input Hyperparameters
learning_rate = 5e-5  # Learning rate
epsilon = 1e-8       # Epsilon untuk optimizer
max_length = 128     # Max length untuk tokenisasi
epochs = 3           # Jumlah epoch
batch_size = 30      # Batch size untuk dataloader

# Load data dari CSV
train_df = pd.read_csv("Training_Data/trainingIndoG.csv")
val_df = pd.read_csv("Validation_Data/validationIndoG.csv")

# Pastikan label dalam bentuk numerik, misalnya 0, 1, 2 untuk klasifikasi sentimen
# Jika label dalam bentuk string, ubah ke numerik
label_map = {'Positif': 0, 'Netral': 1, 'Negatif': 2}
train_df['label'] = train_df['label'].map(label_map)
val_df['label'] = val_df['label'].map(label_map)

# Pastikan tidak ada nilai NaN atau tidak valid
train_df = train_df.dropna()
val_df = val_df.dropna()

# Pastikan label sudah sesuai (tidak ada nilai yang melebihi jumlah kelas)
assert train_df['label'].min() >= 0 and train_df['label'].max() < 3
assert val_df['label'].min() >= 0 and val_df['label'].max() < 3

# Load model tokenizer
pretrained_model = "indolem/indobertweet-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

# Tokenisasi data
train_input_ids = []
train_attention_masks = []
val_input_ids = []
val_attention_masks = []

# Tokenisasi untuk data training
for sent in train_df['bt']:
    encoded_dict = tokenizer.encode_plus(
        sent,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length', # Ubah dari pad_to_max_length
        truncation=True, # Tambahkan truncation
        return_attention_mask=True,
        return_tensors='pt')
    train_input_ids.append(encoded_dict['input_ids'])
    train_attention_masks.append(encoded_dict['attention_mask'])

# Tokenisasi untuk data validasi
for sent in val_df['bt']:
    val_encoded_dict = tokenizer.encode_plus(
        sent,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length', # Ubah dari pad_to_max_length
        truncation=True, # Tambahkan truncation
        return_attention_mask=True,
        return_tensors='pt')
    val_input_ids.append(val_encoded_dict['input_ids'])
    val_attention_masks.append(val_encoded_dict['attention_mask'])

# Konversi list ke tensor
train_input_ids = torch.cat(train_input_ids, dim=0)
train_attention_masks = torch.cat(train_attention_masks, dim=0)
train_labels = torch.tensor(train_df['label'].values, dtype=torch.long)  # Pastikan tipe data tensor long

val_input_ids = torch.cat(val_input_ids, dim=0)
val_attention_masks = torch.cat(val_attention_masks, dim=0)
val_labels = torch.tensor(val_df['label'].values, dtype=torch.long)  # Pastikan tipe data tensor long

# Buat DataLoader untuk train dan validation
train_data = TensorDataset(train_input_ids, train_attention_masks, train_labels)
train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)

val_data = TensorDataset(val_input_ids, val_attention_masks, val_labels)
validation_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=batch_size)

# Load model untuk klasifikasi
model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model,
    num_labels=3, # Positif, Netral, Negatif
    output_attentions=False,
    output_hidden_states=False
)

# Optimizer dan scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=epsilon)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Pindahkan model ke GPU jika tersedia
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Mulai training
training_stats = []
total_t0 = time.time()
best_valid_loss = float("Inf")

for epoch_i in range(epochs):
    print("")
    print('==================== Epoch {:} / {:} =============='.format(epoch_i + 1, epochs))
    print('Training....')
    
    t0 = time.time()
    total_train_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 20 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('Batch {:>5,} dari {:>5,}. Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        model.zero_grad()
        
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        logits = outputs.logits
        
        total_train_loss += loss.item()
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
    avg_train_loss = total_train_loss / len(train_dataloader)
    training_time = format_time(time.time() - t0)
    
    print("Rata-rata training loss: {0:.2f}".format(avg_train_loss))
    print("Waktu training epoch: {:}".format(training_time))
    
    print("")
    print("Running Validation....")
    
    t0 = time.time()
    model.eval()
    
    total_eval_accuracy = 0
    total_eval_loss = 0
    
    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            
        loss = outputs.loss
        logits = outputs.logits
        
        total_eval_loss += loss.item()
        
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("Akurasi: {0:.2f}".format(avg_val_accuracy))
    
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    validation_time = format_time(time.time() - t0)
    
    print("Validation Loss: {0:.2f}".format(avg_val_loss))
    print("Validation Took: {:}".format(validation_time))
    
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_accuracy': avg_val_accuracy,
            'train_time': training_time,
            'val_time': validation_time
        }
    )
    
    # Simpan model terbaik berdasarkan validasi loss
    if best_valid_loss > avg_val_loss:
        best_valid_loss = avg_val_loss
        save_checkpoint("Hasil/Model/model_GoogleAKU.pt", model, best_valid_loss)
    
    # Simpan data training ke CSV setiap selesai 1 epoch
    save_training_data_to_csv("Hasil/Metrics/metrics_GoogleAKU.csv", training_stats)

# Simpan metrics
#save_metrics("Hasil/metrics.pt", training_stats)
print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

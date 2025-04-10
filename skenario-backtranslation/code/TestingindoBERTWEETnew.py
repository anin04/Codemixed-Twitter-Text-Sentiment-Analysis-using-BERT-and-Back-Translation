import os
import torch
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

###########################################
# Bagian 1: Model S1_Google
###########################################

# Load model dan tokenizer
pretrained_model = "indolem/indobertweet-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=3)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

# Fungsi untuk memuat model dari checkpoint
def load_checkpoint(filepath, model):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

# Load model dari checkpoint
model = load_checkpoint('Hasil\Model\\S3_ori_tapi_di_S1.pt', model)
model.eval()

# Pindahkan model ke GPU jika tersedia
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load test data dari file CSV
test_df = pd.read_csv("S3/Testing_Data/ori.csv")  # Sesuaikan dengan file test Anda

# Peta label dari string ke integer
test_labels_map = {'Positif': 0, 'Netral': 1, 'Negatif': 2}
test_df['label'] = test_df['label'].map(test_labels_map)

# Tokenisasi data testing
test_input_ids = []
test_attention_masks = []

for sent in test_df['colloquial_3']:
    encoded_dict = tokenizer.encode_plus(
        sent,
        add_special_tokens=True,
        max_length=128,           # Sesuaikan jika perlu
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    test_input_ids.append(encoded_dict['input_ids'])
    test_attention_masks.append(encoded_dict['attention_mask'])

# Konversi list ke tensor
test_input_ids = torch.cat(test_input_ids, dim=0)
test_attention_masks = torch.cat(test_attention_masks, dim=0)
test_labels_tensor = torch.tensor(test_df['label'].values, dtype=torch.long)

# Buat DataLoader untuk test data
batch_size = 16  # Sesuaikan dengan kebutuhan
test_data = TensorDataset(test_input_ids, test_attention_masks, test_labels_tensor)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# Prediksi dan Evaluasi
print('Predicting labels for {:,} test sentences...'.format(len(test_input_ids)))

y_pred, y_true = [], []

for batch in test_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch
    
    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    
    logits = outputs.logits
    y_pred.extend(torch.argmax(logits, dim=1).tolist())
    y_true.extend(b_labels.to('cpu').numpy().tolist())

print('Testing Done')

# Mapping terbalik dari integer ke string
inverse_label_map = {0: 'Positif', 1: 'Netral', 2: 'Negatif'}

# Tambahkan kolom predicted_label ke test_df (dalam bentuk string)
test_df['predicted_label'] = [inverse_label_map[pred] for pred in y_pred]

# Ubah kolom label asli kembali ke bentuk string
test_df['label'] = test_df['label'].map(inverse_label_map)

# Simpan hasil prediksi ke CSV
os.makedirs("Hasil/Predictions", exist_ok=True)
predicted_csv_path = os.path.join("Hasil/Predictions", "predicted_S3_ori_tapi_di_S1.csv")
test_df.to_csv(predicted_csv_path, index=False)
print(f"Predicted CSV saved at: {predicted_csv_path}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
class_labels = ['Positif', 'Netral', 'Negatif']  # Pastikan urutannya sesuai

df_cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)
plt.figure(figsize=(8,6))
hmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
plt.ylabel('True Sentiment')
plt.xlabel('Predicted Sentiment')

os.makedirs("Hasil/ConfusionMatrix", exist_ok=True)
confusion_matrix_path = os.path.join("Hasil/ConfusionMatrix", "cf_matrix_S3_ori_tapi_di_S1.png")
plt.savefig(confusion_matrix_path)
plt.close()

# Classification Report
report = classification_report(y_true, y_pred, target_names=class_labels, digits=4, output_dict=True)
df_report = pd.DataFrame(report).transpose()

os.makedirs("Hasil/ClassificationReport", exist_ok=True)
classification_report_path = os.path.join("Hasil/ClassificationReport", "clsf_report_S3_ori_tapi_di_S1.csv")
df_report.to_csv(classification_report_path, index=True)

print(f"Confusion Matrix saved at: {confusion_matrix_path}")
print(f"Classification Report saved at: {classification_report_path}")
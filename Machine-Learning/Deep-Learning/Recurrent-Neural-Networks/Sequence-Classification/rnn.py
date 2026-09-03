from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
FIGURES_DIR = BASE_DIR / 'figures'
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(BASE_DIR)

def save_figure(filename):
    plt.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close('all')


# Bölüm 2 — Kurulum & Kütüphaneler

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
np.random.seed(42)
torch.manual_seed(42)
print(' NumPy  :', np.__version__)
print(' PyTorch:', torch.__version__)


# Bölüm 3 — Sıfırdan RNN (NumPy)

class VanillaRNNCell:
    """
    Tek bir RNN hücresi — sıfırdan, sadece NumPy ile.

    Formül:
        h_t = tanh( W_hh @ h_{t-1}  +  W_xh @ x_t  +  b_h )
        y_t = W_hy @ h_t  +  b_y
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        rng = np.random.default_rng(42)
        scale = 0.01
        self.W_hh = rng.standard_normal((hidden_size, hidden_size)) * scale
        self.W_xh = rng.standard_normal((hidden_size, input_size)) * scale
        self.W_hy = rng.standard_normal((output_size, hidden_size)) * scale
        self.b_h = np.zeros(hidden_size)
        self.b_y = np.zeros(output_size)
        self.hidden_size = hidden_size

    def forward_step(self, x_t: np.ndarray, h_prev: np.ndarray):
        """
        Tek bir zaman adımı.

        Parametreler:
            x_t    : (input_size,)   ← mevcut adımın girdisi
            h_prev : (hidden_size,)  ← bir önceki adımın gizli durumu

        Döndürür:
            h_t : (hidden_size,)   ← YENİ gizli durum (sonraki adıma geçecek)
            y_t : (output_size,)   ← bu adımın çıktısı
        """
        h_t = np.tanh(self.W_hh @ h_prev + self.W_xh @ x_t + self.b_h)
        y_t = self.W_hy @ h_t + self.b_y
        return (h_t, y_t)

    def forward_sequence(self, X: np.ndarray):
        """
        Tam diziyi işle.

        Parametreler:
            X : (seq_len, input_size)

        Döndürür:
            H : (seq_len, hidden_size)  ← tüm adımların gizli durumları
            Y : (seq_len, output_size)  ← tüm adımların çıktıları
        """
        seq_len, _ = X.shape
        h = np.zeros(self.hidden_size)
        H = np.zeros((seq_len, self.hidden_size))
        Y = np.zeros((seq_len, len(self.b_y)))
        for t, x_t in enumerate(X):
            h, y = self.forward_step(x_t, h)
            H[t] = h
            Y[t] = y
        return (H, Y)
print('VanillaRNNCell sınıfı tanımlandı ')


# Sıfırdan RNN — Test

INPUT_SIZE = 4
HIDDEN_SIZE = 8
OUTPUT_SIZE = 2
SEQ_LEN = 5
X_dummy = np.random.randn(SEQ_LEN, INPUT_SIZE)
cell = VanillaRNNCell(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
H, Y = cell.forward_sequence(X_dummy)
print(f'Girdi  X : {X_dummy.shape}  (seq_len × input_size)')
print(f'Gizli  H : {H.shape}       (seq_len × hidden_size)')
print(f'Çıktı  Y : {Y.shape}        (seq_len × output_size)')
print()
print(f"{'Adım':>5}  {'‖h_t‖ (L2)':>12}")
print('─' * 22)
for t, h_t in enumerate(H):
    print(f'{t:>5}  {np.linalg.norm(h_t):>12.5f}')
print(f'\nSon çıktı y_T : {Y[-1].round(5)}')


# Gizli Durum Isı Haritası

fig, ax = plt.subplots(figsize=(10, 3))
im = ax.imshow(H, aspect='auto', cmap='viridis', vmin=-1, vmax=1)
ax.set_xlabel('Gizli boyut (hidden dimension)', fontsize=11)
ax.set_ylabel('Zaman adımı (t)', fontsize=11)
ax.set_title('Gizli Durum Isı Haritası — h_t adım adım değişir', fontsize=12)
ax.set_yticks(range(SEQ_LEN))
ax.set_yticklabels([f't={i}' for i in range(SEQ_LEN)])
plt.colorbar(im, ax=ax, label='Aktivasyon değeri')
plt.tight_layout()
save_figure('hidden_state_heatmap.png')
print('Isı haritası kaydedildi ')


# Bölüm 4 — PyTorch `nn.RNN` ile Sınıflandırıcı

class RNNClassifier(nn.Module):
    """
    Dizi → tek etiket sınıflandırması.

    Parametreler:
        input_size  : her adımın özellik sayısı
        hidden_size : RNN gizli durum boyutu
        num_classes : kaç sınıf var
        num_layers  : yığılmış RNN katman sayısı (varsayılan 1)
    """

    def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        x       : (batch, seq_len, input_size)

        Döndürür:
            logits : (batch, num_classes)
        """
        output, h_n = self.rnn(x)
        last_hidden = h_n[-1]
        logits = self.fc(last_hidden)
        return logits
print('RNNClassifier tanımlandı ')
print()
model = RNNClassifier(input_size=10, hidden_size=32, num_classes=3)
print(model)
total_params = sum((p.numel() for p in model.parameters()))
print(f'\nToplam parametre sayısı: {total_params:,}')


# Veri Hazırlama & Eğitim Döngüsü

BATCH = 64
SEQ_LEN = 20
IN_SIZE = 10
HID_SIZE = 32
N_CLASSES = 3
EPOCHS = 50
LR = 0.001
X_train = torch.randn(BATCH, SEQ_LEN, IN_SIZE)
y_train = torch.randint(0, N_CLASSES, (BATCH,))
model = RNNClassifier(IN_SIZE, HID_SIZE, N_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
history = {'loss': [], 'acc': []}
print(f"{'Epoch':>6}  {'Kayıp':>8}  {'Doğruluk':>10}")
print('─' * 32)
for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train)
    loss = criterion(logits, y_train)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        acc = (preds == y_train).float().mean().item() * 100
    history['loss'].append(loss.item())
    history['acc'].append(acc)
    if epoch % 10 == 0:
        print(f'{epoch:>6}  {loss.item():>8.4f}  {acc:>9.1f}%')
print('\nEğitim tamamlandı ')


# Eğitim Eğrisi

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history['loss'], color='#534AB7', linewidth=2)
ax1.set_title('Eğitim Kaybı (CrossEntropy)', fontsize=12)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Kayıp')
ax1.grid(alpha=0.3)
ax2.plot(history['acc'], color='#1D9E75', linewidth=2)
ax2.set_title('Eğitim Doğruluğu (%)', fontsize=12)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Doğruluk (%)')
ax2.set_ylim(0, 105)
ax2.axhline(33.3, color='gray', linestyle='--', alpha=0.5, label='Rastgele tahmin (33%)')
ax2.legend()
ax2.grid(alpha=0.3)
plt.tight_layout()
save_figure('training_curves.png')
print('Eğitim eğrisi kaydedildi ')


# Bölüm 5 — Adım Adım Gizli Durum İzleme

model.eval()
test_seq = torch.randn(1, SEQ_LEN, IN_SIZE)
hidden_states = []
h = None
with torch.no_grad():
    for t in range(SEQ_LEN):
        x_t = test_seq[:, t:t + 1, :]
        out, h = model.rnn(x_t, h)
        hidden_states.append(h[-1].squeeze().numpy())
norms = [np.linalg.norm(ht) for ht in hidden_states]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
ax1.plot(range(SEQ_LEN), norms, 'o-', color='#E24B4A', linewidth=2, markersize=6)
ax1.set_title('‖h_t‖ — Gizli Durum Normu Zaman İçinde', fontsize=12)
ax1.set_xlabel('Zaman adımı (t)')
ax1.set_ylabel('L2 normu')
ax1.grid(alpha=0.3)
H_matrix = np.stack(hidden_states)[:, :16]
im = ax2.imshow(H_matrix.T, aspect='auto', cmap='RdYlBu_r', vmin=-1, vmax=1)
ax2.set_title('h_t Isı Haritası (ilk 16 boyut)', fontsize=12)
ax2.set_xlabel('Zaman adımı (t)')
ax2.set_ylabel('Gizli boyut')
plt.colorbar(im, ax=ax2)
plt.tight_layout()
save_figure('hidden_state_tracking.png')
print('Gizli durum izleme grafiği kaydedildi ')


# Bölüm 6 — Tahmin & Yorumlama

model.eval()
with torch.no_grad():
    yeni_dizi = torch.randn(1, SEQ_LEN, IN_SIZE)
    logit = model(yeni_dizi)
    olasilik = torch.softmax(logit, dim=1).squeeze()
    tahmin = olasilik.argmax().item()
siniflar = [f'Sınıf {i}' for i in range(N_CLASSES)]
renkler = ['#534AB7', '#1D9E75', '#E24B4A']
fig, ax = plt.subplots(figsize=(6, 3))
bars = ax.bar(siniflar, olasilik.numpy(), color=renkler, width=0.5)
ax.bar_label(bars, fmt='%.3f', padding=4, fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_title(f'Sınıf Olasılıkları  →  Tahmin: Sınıf {tahmin}', fontsize=12)
ax.set_ylabel('Olasılık')
ax.axhline(1 / N_CLASSES, color='gray', linestyle='--', alpha=0.5, label='Eşit dağılım')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
save_figure('class_probabilities.png')
print(f'Tahmin edilen sınıf : {tahmin}')
print(f'Olasılıklar         : {olasilik.numpy().round(4)}')


# Bölüm 7 — RNN'nin Temel Zayıflığı: Kaybolan Gradient

T = 50
grad_vanish = [0.9 ** t for t in range(T)]
grad_explode = [min(1.1 ** t, 1000000.0) for t in range(T)]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
ax1.plot(grad_vanish, color='#3B8BD4', linewidth=2)
ax1.set_title('Kaybolan Gradient  (factor=0.9)', fontsize=12)
ax1.set_xlabel('Zaman adımı (geriye doğru)')
ax1.set_ylabel('Gradient büyüklüğü')
ax1.fill_between(range(T), grad_vanish, alpha=0.2, color='#3B8BD4')
ax1.grid(alpha=0.3)
ax1.annotate('Uzak adımlar\nneredeyse sıfır!', xy=(45, grad_vanish[45]), xytext=(30, 0.3), arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, color='red')
ax2.plot(grad_explode, color='#E24B4A', linewidth=2)
ax2.set_title('Patlayan Gradient  (factor=1.1)', fontsize=12)
ax2.set_xlabel('Zaman adımı (geriye doğru)')
ax2.set_ylabel('Gradient büyüklüğü (log)')
ax2.set_yscale('log')
ax2.fill_between(range(T), grad_explode, alpha=0.2, color='#E24B4A')
ax2.grid(alpha=0.3)
plt.tight_layout()
save_figure('gradient_behavior.png')
print('Kaybolan/patlayan gradient görselleştirildi ')
print()
print("Çözüm: LSTM ve GRU, bu problemi 'kapılar' (gates) ve")
print("'hücre durumu' (cell state) ile çözer.")

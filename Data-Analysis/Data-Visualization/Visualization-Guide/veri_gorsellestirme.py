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


# 1. Kurulum ve Kütüphaneler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (8, 5)
print('Kütüphaneler başarıyla yüklendi.')


# 2. Örnek Veri Setinin Hazırlanması

np.random.seed(42)
aylar = np.arange(1, 13)
satislar = np.random.randint(100, 500, size=12)
giderler = np.random.randint(50, 300, size=12)
df_satis = pd.DataFrame({'Ay': aylar, 'Satis': satislar, 'Gider': giderler})
df_satis['Kar'] = df_satis['Satis'] - df_satis['Gider']
df_satis.head()


# 3. Matplotlib Temelleri: Çizgi Grafiği (Line Plot)

plt.figure(figsize=(8, 5))
plt.plot(df_satis['Ay'], df_satis['Satis'], marker='o', linewidth=2, color='#2E86AB', label='Satış')
plt.plot(df_satis['Ay'], df_satis['Gider'], marker='s', linewidth=2, color='#E76F51', label='Gider')
plt.title('Aylık Satış ve Gider Trendi', fontsize=14, fontweight='bold')
plt.xlabel('Ay')
plt.ylabel('Tutar (TL)')
plt.xticks(df_satis['Ay'])
plt.legend()
plt.grid(alpha=0.3)
save_figure('monthly_sales_expenses.png')


# 4. Bar (Çubuk) Grafikler

plt.figure(figsize=(8, 5))
renkler = ['#2A9D8F' if k >= 0 else '#E63946' for k in df_satis['Kar']]
plt.bar(df_satis['Ay'], df_satis['Kar'], color=renkler)
plt.axhline(0, color='black', linewidth=0.8)
plt.title('Aylık Kâr / Zarar Durumu', fontsize=14, fontweight='bold')
plt.xlabel('Ay')
plt.ylabel('Kâr (TL)')
plt.xticks(df_satis['Ay'])
save_figure('monthly_profit_loss.png')


# 5. Histogram: Dağılım Analizi

veri_dagilim = np.random.normal(loc=170, scale=10, size=1000)
plt.figure(figsize=(8, 5))
plt.hist(veri_dagilim, bins=30, color='#457B9D', edgecolor='white', alpha=0.85)
plt.axvline(veri_dagilim.mean(), color='#E63946', linestyle='--', linewidth=2, label=f'Ortalama = {veri_dagilim.mean():.1f}')
plt.title('Örnek Dağılım Histogramı', fontsize=14, fontweight='bold')
plt.xlabel('Değer')
plt.ylabel('Frekans (Sıklık)')
plt.legend()
save_figure('distribution_histogram.png')


# 6. Scatter (Serpme) Grafiği

plt.figure(figsize=(8, 5))
sc = plt.scatter(df_satis['Satis'], df_satis['Gider'], s=100, c=df_satis['Kar'], cmap='RdYlGn', edgecolor='black')
cbar = plt.colorbar(sc)
cbar.set_label('Kâr (TL)')
plt.title('Satış vs Gider İlişkisi (Renk = Kâr)', fontsize=14, fontweight='bold')
plt.xlabel('Satış (TL)')
plt.ylabel('Gider (TL)')
save_figure('sales_expense_scatter.png')


# 7. Kutu Grafiği (Boxplot): Aykırı Değer ve Yayılım Analizi

grup_a = np.random.normal(50, 5, 200)
grup_b = np.random.normal(55, 15, 200)
grup_c = np.random.normal(45, 8, 200)
plt.figure(figsize=(8, 5))
kutu = plt.boxplot([grup_a, grup_b, grup_c], tick_labels=['Grup A', 'Grup B', 'Grup C'], patch_artist=True)
renkler = ['#F4A261', '#2A9D8F', '#264653']
for kutu_govdesi, renk in zip(kutu['boxes'], renkler):
    kutu_govdesi.set_facecolor(renk)
plt.title('Gruplar Arası Dağılım Karşılaştırması', fontsize=14, fontweight='bold')
plt.ylabel('Değer')
save_figure('group_boxplots.png')


# 8. Pasta (Pie) Grafiği

kategoriler = ['Elektronik', 'Giyim', 'Ev & Yaşam', 'Kozmetik', 'Diğer']
paylar = [35, 25, 20, 12, 8]
plt.figure(figsize=(6, 6))
explode = [0.08 if p == max(paylar) else 0 for p in paylar]
plt.pie(paylar, labels=kategoriler, autopct='%1.1f%%', startangle=90, explode=explode, colors=sns.color_palette('Set2'))
plt.title('Kategori Bazında Satış Payı', fontsize=14, fontweight='bold')
plt.axis('equal')
save_figure('category_sales_share.png')


# 9. Alt Grafikler (Subplots): Çoklu Görselleştirme

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes[0, 0].plot(df_satis['Ay'], df_satis['Satis'], marker='o', color='#2E86AB')
axes[0, 0].set_title('Aylık Satış')
axes[0, 1].bar(df_satis['Ay'], df_satis['Gider'], color='#E76F51')
axes[0, 1].set_title('Aylık Gider')
axes[1, 0].hist(veri_dagilim, bins=20, color='#457B9D')
axes[1, 0].set_title('Dağılım Histogramı')
axes[1, 1].scatter(df_satis['Satis'], df_satis['Gider'], color='#2A9D8F')
axes[1, 1].set_title('Satış vs Gider')
fig.suptitle('Çoklu Grafik Paneli (Dashboard) Örneği', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
save_figure('multi_plot_dashboard.png')


# 10. Seaborn ile İleri Düzey Görselleştirme

tips = pd.read_csv(DATA_DIR / 'tips.csv')
tips.head()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='time', style='smoker', s=80)
plt.title('Hesap Tutarı vs Bahşiş (Öğün Zamanına Göre Renklendirilmiş)', fontsize=13, fontweight='bold')
plt.xlabel('Toplam Hesap ($)')
plt.ylabel('Bahşiş ($)')
save_figure('bill_tip_relationship.png')

plt.figure(figsize=(8, 5))
sns.violinplot(data=tips, x='day', y='total_bill', hue='sex', split=True, palette='Set2')
plt.title('Güne ve Cinsiyete Göre Hesap Tutarı Dağılımı', fontsize=13, fontweight='bold')
plt.xlabel('Gün')
plt.ylabel('Toplam Hesap ($)')
save_figure('bill_distribution_by_day.png')

sns.pairplot(tips, hue='time', vars=['total_bill', 'tip', 'size'], palette='husl')
plt.suptitle('Değişkenler Arası İlişki Matrisi (Pairplot)', y=1.02, fontsize=14, fontweight='bold')
save_figure('tips_pairplot.png')


# 11. Isı Haritası (Heatmap) ve Korelasyon Analizi

korelasyon = tips.select_dtypes(include='number').corr()
plt.figure(figsize=(6, 5))
sns.heatmap(korelasyon, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', linewidths=0.5)
plt.title('Korelasyon Matrisi (Isı Haritası)', fontsize=13, fontweight='bold')
save_figure('tips_correlation_matrix.png')


# 12. Plotly ile İnteraktif Görselleştirme

import plotly.express as px
fig = px.scatter(tips, x='total_bill', y='tip', color='day', size='size', hover_data=['sex', 'smoker'], title='İnteraktif Hesap Tutarı vs Bahşiş Grafiği')
fig.update_layout(template='plotly_white', width=800, height=500)
fig.write_html(FIGURES_DIR / 'interactive_scatter.html')

fig2 = px.line(df_satis, x='Ay', y=['Satis', 'Gider', 'Kar'], markers=True, title='İnteraktif Aylık Finansal Trend')
fig2.update_layout(template='plotly_white', xaxis_title='Ay', yaxis_title='Tutar (TL)')
fig2.write_html(FIGURES_DIR / 'interactive_trend.html')


# 13. Görselleştirme Tasarım İlkeleri (Best Practices)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
degerler = [420, 430, 445, 440]
etiketler = ['Ürün A', 'Ürün B', 'Ürün C', 'Ürün D']
axes[0].bar(etiketler, degerler, color='#E76F51')
axes[0].set_ylim(400, 450)
axes[0].set_title("YANLIŞ: Y Ekseni 0'dan Başlamıyor")
axes[1].bar(etiketler, degerler, color='#2A9D8F')
axes[1].set_ylim(0, 500)
axes[1].set_title("DOĞRU: Y Ekseni 0'dan Başlıyor")
plt.tight_layout()
save_figure('axis_design_comparison.png')


# 14. Gerçek Veri Seti Üzerinde Uçtan Uca Uygulama

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
gun_ortalama = tips.groupby('day', observed=True)['total_bill'].mean().sort_values()
axes[0, 0].barh(gun_ortalama.index, gun_ortalama.values, color='#457B9D')
axes[0, 0].set_title('Günlere Göre Ortalama Hesap')
axes[0, 0].set_xlabel('Ortalama Hesap ($)')
tips['bahsis_orani'] = tips['tip'] / tips['total_bill'] * 100
axes[0, 1].hist(tips['bahsis_orani'], bins=25, color='#F4A261', edgecolor='white')
axes[0, 1].set_title('Bahşiş Oranı Dağılımı (%)')
axes[0, 1].set_xlabel('Bahşiş Oranı (%)')
sns.boxplot(data=tips, x='sex', y='total_bill', hue='sex', legend=False, ax=axes[1, 0], palette='pastel')
axes[1, 0].set_title('Cinsiyete Göre Hesap Tutarı')
sns.regplot(data=tips, x='size', y='total_bill', ax=axes[1, 1], scatter_kws={'alpha': 0.5, 'color': '#264653'}, line_kws={'color': '#E63946'})
axes[1, 1].set_title('Kişi Sayısı vs Hesap Tutarı (Trend Çizgili)')
fig.suptitle('Restoran Bahşiş Verisi - Keşifsel Analiz Paneli', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
save_figure('tips_eda_dashboard.png')


# 15. Grafikleri Dosyaya Kaydetme

plt.figure(figsize=(8, 5))
plt.plot(df_satis['Ay'], df_satis['Kar'], marker='o', color='#2A9D8F', linewidth=2)
plt.title('Aylık Kâr Trendi')
plt.xlabel('Ay')
plt.ylabel('Kâr (TL)')
plt.savefig(FIGURES_DIR / 'monthly_profit_trend_300dpi.png', dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / 'monthly_profit_trend.png', bbox_inches='tight', dpi=150)
plt.close('all')
print("Aylık kâr trendi grafikleri kaydedildi.")

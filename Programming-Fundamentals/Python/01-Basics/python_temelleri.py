from pathlib import Path


# Değişkenler ve veri tipleri

course_name = "Veri Analitiği"
duration_weeks = 12
completion_rate = 0.92
is_completed = True

print(type(course_name).__name__, type(duration_weeks).__name__)
print(f"{course_name}: %{completion_rate * 100:.0f} tamamlandı")


# Metin ve sayısal işlemler

raw_name = "  istanbul veri analizi  "
clean_name = raw_name.strip().title()
scores = [78, 85, 91, 88, 94]

print(clean_name)
print(f"Ortalama: {sum(scores) / len(scores):.2f}")
print(f"Aralık: {min(scores)}-{max(scores)}")


# Koleksiyonlar

topics = ["Python", "NumPy", "Pandas", "SQL"]
topic_tuple = tuple(topics)
topic_set = {topic.lower() for topic in topics}
student = {
    "name": "Analyst",
    "topics": topics,
    "scores": scores,
}

topics.append("Power BI")
print(topic_tuple)
print(sorted(topic_set))
print(student["name"], student["scores"][-1])


# Koşullar ve döngüler

average_score = sum(scores) / len(scores)
if average_score >= 90:
    level = "ileri"
elif average_score >= 75:
    level = "orta"
else:
    level = "başlangıç"

for index, topic in enumerate(topics, start=1):
    print(f"{index:02d}. {topic}")

print(f"Başarı seviyesi: {level}")


# List comprehension ve sözlük üretimi

passed_scores = [score for score in scores if score >= 85]
score_map = {f"quiz_{index}": score for index, score in enumerate(scores, start=1)}
normalized = [(score - min(scores)) / (max(scores) - min(scores)) for score in scores]

print("Geçen skorlar:", passed_scores)
print("Skor sözlüğü:", score_map)
print("Normalize skorlar:", [round(value, 2) for value in normalized])


# Hata yönetimi

raw_values = ["42", "18", "hatalı", "27"]
valid_values = []

for value in raw_values:
    try:
        valid_values.append(int(value))
    except ValueError:
        print(f"Atlanan değer: {value!r}")

print("Geçerli değerler:", valid_values)


# Dosya işlemleri

output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(exist_ok=True)
summary_path = output_dir / "course_summary.txt"

summary_lines = [
    f"Konu sayısı: {len(topics)}",
    f"Ortalama skor: {average_score:.2f}",
    f"Seviye: {level}",
]
summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

print(summary_path.read_text(encoding="utf-8"))

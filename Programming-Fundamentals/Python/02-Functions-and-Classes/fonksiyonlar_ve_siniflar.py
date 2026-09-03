from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from statistics import mean


# Fonksiyonlar

def calculate_average(values: list[float]) -> float:
    if not values:
        raise ValueError("En az bir değer girilmelidir.")
    return mean(values)


def summarize_scores(name: str, *scores: float, threshold: float = 70) -> dict:
    average = calculate_average(list(scores))
    return {
        "name": name,
        "average": round(average, 2),
        "passed": average >= threshold,
    }


students = [
    summarize_scores("Ada", 82, 91, 87),
    summarize_scores("Ece", 68, 74, 71),
    summarize_scores("Mert", 55, 61, 58),
]

successful_students = list(filter(lambda item: item["passed"], students))
ranked_students = sorted(students, key=lambda item: item["average"], reverse=True)

print("Başarılı öğrenciler:", successful_students)
print("Sıralama:", ranked_students)


# Veri sınıfı ve kapsülleme

@dataclass
class Book:
    title: str
    author: str
    year: int
    _is_borrowed: bool = field(default=False, init=False, repr=False)

    @property
    def is_borrowed(self) -> bool:
        return self._is_borrowed

    def borrow(self) -> None:
        if self._is_borrowed:
            raise ValueError(f"{self.title} zaten ödünç verilmiş.")
        self._is_borrowed = True

    def return_book(self) -> None:
        self._is_borrowed = False

    def __str__(self) -> str:
        status = "ödünçte" if self.is_borrowed else "rafta"
        return f"{self.title} — {self.author} ({self.year}), {status}"


# Kalıtım ve soyutlama

class LibraryMember(ABC):
    def __init__(self, name: str) -> None:
        self.name = name
        self.borrowed_books: list[Book] = []

    @property
    @abstractmethod
    def borrowing_limit(self) -> int:
        raise NotImplementedError

    def borrow(self, book: Book) -> None:
        if len(self.borrowed_books) >= self.borrowing_limit:
            raise ValueError("Ödünç alma limiti doldu.")
        book.borrow()
        self.borrowed_books.append(book)

    def return_book(self, book: Book) -> None:
        if book not in self.borrowed_books:
            raise ValueError("Kitap bu üyeye ait değil.")
        book.return_book()
        self.borrowed_books.remove(book)


class StudentMember(LibraryMember):
    @property
    def borrowing_limit(self) -> int:
        return 3


class AcademicMember(LibraryMember):
    @property
    def borrowing_limit(self) -> int:
        return 5


# Sınıfların birlikte kullanımı

catalog = [
    Book("Python ile Veri Analizi", "Wes McKinney", 2022),
    Book("Hands-On Machine Learning", "Aurélien Géron", 2022),
    Book("Storytelling with Data", "Cole Nussbaumer Knaflic", 2015),
]

member = StudentMember("Ada")
member.borrow(catalog[0])
member.borrow(catalog[2])

for book in catalog:
    print(book)

member.return_book(catalog[0])
print("Ödünçteki kitap sayısı:", len(member.borrowed_books))

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
DATA_DIR = os.path.join(BASE_DIR, "data")

# ============================================================
# 1. СОЗДАНИЕ И ОБРАБОТКА МАССИВОВ
# ============================================================

def create_vector() -> np.ndarray:
    """
    Создать массив от 0 до 9.

    Returns:
        numpy.ndarray: Массив чисел от 0 до 9 включительно.
    """
    return np.arange(10)


def create_matrix() -> np.ndarray:
    """
    Создать матрицу 5x5 со случайными числами [0,1).

    Returns:
        numpy.ndarray: Матрица 5x5 со случайными значениями от 0 до 1.
    """
    return np.random.rand(5, 5)


def reshape_vector(vec: np.ndarray) -> np.ndarray:
    """
    Преобразовать (10,) -> (2,5).

    Args:
        vec: Входной массив формы (10,).

    Returns:
        numpy.ndarray: Преобразованный массив формы (2, 5).
    """
    return vec.reshape(2, 5)


def transpose_matrix(mat: np.ndarray) -> np.ndarray:
    """
    Транспонирование матрицы.

    Args:
        mat: Входная матрица.

    Returns:
        numpy.ndarray: Транспонированная матрица.
    """
    return mat.T


# ============================================================
# 2. ВЕКТОРНЫЕ ОПЕРАЦИИ
# ============================================================

def vector_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Сложение векторов одинаковой длины.

    Args:
        a: Первый вектор.
        b: Второй вектор.

    Returns:
        numpy.ndarray: Результат поэлементного сложения.
    """
    return a + b


def scalar_multiply(vec: np.ndarray, scalar: float | int) -> np.ndarray:
    """
    Умножение вектора на число.

    Args:
        vec: Входной вектор.
        scalar: Число для умножения.

    Returns:
        numpy.ndarray: Результат умножения вектора на скаляр.
    """
    return vec * scalar


def elementwise_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Поэлементное умножение.

    Args:
        a: Первый вектор или матрица.
        b: Второй вектор или матрица.

    Returns:
        numpy.ndarray: Результат поэлементного умножения.
    """
    return a * b


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """
    Скалярное произведение.

    Args:
        a: Первый вектор.
        b: Второй вектор.

    Returns:
        float: Скалярное произведение векторов.
    """
    return float(np.dot(a, b))


# ============================================================
# 3. МАТРИЧНЫЕ ОПЕРАЦИИ
# ============================================================

def matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Умножение матриц.

    Args:
        a: Первая матрица.
        b: Вторая матрица.

    Returns:
        numpy.ndarray: Результат умножения матриц.
    """
    return a @ b


def matrix_determinant(a: np.ndarray) -> float:
    """
    Определитель матрицы.

    Args:
        a: Квадратная матрица.

    Returns:
        float: Определитель матрицы.
    """
    return float(np.linalg.det(a))


def matrix_inverse(a: np.ndarray) -> np.ndarray:
    """
    Обратная матрица.

    Args:
        a: Квадратная матрица.

    Returns:
        numpy.ndarray: Обратная матрица.
    """
    return np.linalg.inv(a)


def solve_linear_system(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Решить систему Ax = b.

    Args:
        a: Матрица коэффициентов A.
        b: Вектор свободных членов b.

    Returns:
        numpy.ndarray: Решение системы x.
    """
    return np.linalg.solve(a, b)


# ============================================================
# 4. СТАТИСТИЧЕСКИЙ АНАЛИЗ
# ============================================================

def load_dataset(path: str = "data/students_scores.csv") -> np.ndarray:
    """
    Загрузить CSV и вернуть NumPy-массив.
    """
    if os.path.exists(path):
        real_path = path
    else:
        real_path = os.path.join(BASE_DIR, path)

    return pd.read_csv(real_path).to_numpy()


def statistical_analysis(data: np.ndarray) -> dict[str, Any]:
    """
    Вычислить основные статистические показатели.

    Args:
        data: Одномерный массив данных.

    Returns:
        dict: Словарь со статистическими показателями.
    """
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "percentile_25": float(np.percentile(data, 25)),
        "percentile_75": float(np.percentile(data, 75)),
    }


def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Min-Max нормализация.

    Args:
        data: Входной массив данных.

    Returns:
        numpy.ndarray: Нормализованный массив данных в диапазоне [0, 1].
    """
    data_min = np.min(data)
    data_max = np.max(data)

    if data_max == data_min:
        return np.zeros_like(data, dtype=float)

    return (data - data_min) / (data_max - data_min)


# ============================================================
# 5. ВИЗУАЛИЗАЦИЯ
# ============================================================

def _ensure_plots_dir() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_histogram(data: np.ndarray) -> None:
    """
    Построить гистограмму распределения оценок по математике.

    Args:
        data: Данные для гистограммы.
    """
    _ensure_plots_dir()
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=10, edgecolor="black")
    plt.title("Гистограмма оценок по математике")
    plt.xlabel("Оценка")
    plt.ylabel("Частота")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "histogram.png"))
    plt.close()


def plot_heatmap(matrix: np.ndarray) -> None:
    """
    Построить тепловую карту корреляции предметов.

    Args:
        matrix: Матрица корреляции.
    """
    _ensure_plots_dir()
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, cmap="viridis")
    plt.title("Тепловая карта корреляции")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "heatmap.png"))
    plt.close()


def plot_line(x: np.ndarray, y: np.ndarray) -> None:
    """
    Построить график зависимости: студент -> оценка по математике.

    Args:
        x: Номера студентов.
        y: Оценки студентов.
    """
    _ensure_plots_dir()
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")
    plt.title("Оценки студентов по математике")
    plt.xlabel("Номер студента")
    plt.ylabel("Оценка")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "line_plot.png"))
    plt.close()


# ============================================================
# ========================== ТЕСТЫ ===========================
# ============================================================

def test_create_vector():
    v = create_vector()
    assert isinstance(v, np.ndarray)
    assert v.shape == (10,)
    assert np.array_equal(v, np.arange(10))


def test_create_matrix():
    m = create_matrix()
    assert isinstance(m, np.ndarray)
    assert m.shape == (5, 5)
    assert np.all((m >= 0) & (m < 1))


def test_reshape_vector():
    v = np.arange(10)
    reshaped = reshape_vector(v)
    assert reshaped.shape == (2, 5)
    assert reshaped[0, 0] == 0
    assert reshaped[1, 4] == 9


def test_vector_add():
    assert np.array_equal(
        vector_add(np.array([1, 2, 3]), np.array([4, 5, 6])),
        np.array([5, 7, 9]),
    )
    assert np.array_equal(
        vector_add(np.array([0, 1]), np.array([1, 1])),
        np.array([1, 2]),
    )


def test_scalar_multiply():
    assert np.array_equal(
        scalar_multiply(np.array([1, 2, 3]), 2),
        np.array([2, 4, 6]),
    )


def test_elementwise_multiply():
    assert np.array_equal(
        elementwise_multiply(np.array([1, 2, 3]), np.array([4, 5, 6])),
        np.array([4, 10, 18]),
    )


def test_dot_product():
    assert dot_product(np.array([1, 2, 3]), np.array([4, 5, 6])) == 32
    assert dot_product(np.array([2, 0]), np.array([3, 5])) == 6


def test_matrix_multiply():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[2, 0], [1, 2]])
    assert np.array_equal(matrix_multiply(a, b), a @ b)


def test_matrix_determinant():
    a = np.array([[1, 2], [3, 4]])
    assert round(matrix_determinant(a), 5) == -2.0


def test_matrix_inverse():
    a = np.array([[1, 2], [3, 4]])
    inv_a = matrix_inverse(a)
    assert np.allclose(a @ inv_a, np.eye(2))


def test_solve_linear_system():
    a = np.array([[2, 1], [1, 3]])
    b = np.array([1, 2])
    x = solve_linear_system(a, b)
    assert np.allclose(a @ x, b)


def test_load_dataset():
    test_data = "math,physics,informatics\n78,81,90\n85,89,88"
    with open("test_data.csv", "w", encoding="utf-8") as f:
        f.write(test_data)
    try:
        data = load_dataset("test_data.csv")
        assert data.shape == (2, 3)
        assert np.array_equal(data[0], [78, 81, 90])
    finally:
        os.remove("test_data.csv")


def test_statistical_analysis():
    data = np.array([10, 20, 30])
    result = statistical_analysis(data)
    assert result["mean"] == 20
    assert result["min"] == 10
    assert result["max"] == 30


def test_normalization():
    data = np.array([0, 5, 10])
    norm = normalize_data(data)
    assert np.allclose(norm, np.array([0, 0.5, 1]))


def test_plot_histogram():
    data = np.array([1, 2, 3, 4, 5])
    plot_histogram(data)


def test_plot_heatmap():
    matrix = np.array([[1, 0.5], [0.5, 1]])
    plot_heatmap(matrix)


def test_plot_line():
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    plot_line(x, y)


if __name__ == "__main__":
    print("Запустите: python -m pytest LR-2-main.py -v")
import numpy as np
from collections import Counter

from pygame.transform import threshold


def find_best_split(feature_vector, target_vector):
    """
    Находит оптимальный порог для разбиения вектора признака по критерию Джини.

    Критерий Джини определяется следующим образом:
    .. math::
        Q(R) = -\\frac {|R_l|}{|R|}H(R_l) -\\frac {|R_r|}{|R|}H(R_r),

    где:
    * :math:`R` — множество всех объектов,
    * :math:`R_l` и :math:`R_r` — объекты, попавшие в левое и правое поддерево соответственно.

    Функция энтропии :math:`H(R)`:
    .. math::
        H(R) = 1 - p_1^2 - p_0^2,

    где:
    * :math:`p_1` и :math:`p_0` — доля объектов класса 1 и 0 соответственно.

    Указания:
    - Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    - В качестве порогов, нужно брать среднее двух соседних (при сортировке) значений признака.
    - Поведение функции в случае константного признака может быть любым.
    - При одинаковых приростах Джини нужно выбирать минимальный сплит.
    - Для оптимизации рекомендуется использовать векторизацию вместо циклов.

    Parameters
    ----------
    feature_vector : np.ndarray
        Вектор вещественнозначных значений признака.
    target_vector : np.ndarray
        Вектор классов объектов (0 или 1), длина `feature_vector` равна длине `target_vector`.

    Returns
    -------
    thresholds : np.ndarray
        Отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно разделить на
        два различных поддерева.
    ginis : np.ndarray
        Вектор со значениями критерия Джини для каждого порога в `thresholds`.
    threshold_best : float
        Оптимальный порог для разбиения.
    gini_best : float
        Оптимальное значение критерия Джини.

    """
    # ╰( ͡☉ ͜ʖ ͡☉ )つ──☆*:・ﾟ   ฅ^•ﻌ•^ฅ   ʕ•ᴥ•ʔ

    # преобразуем в numpy
    feature_values = np.array(feature_vector)
    target_classes = np.array(target_vector)

    # сортируем
    order = np.argsort(feature_values)
    sorted_features = feature_values[order]
    sorted_targets = target_classes[order]

    thresholds = (np.unique(sorted_features[1:]) + np.unique(sorted_features[:-1])) / 2 # проги
    unique_thresholds = np.unique(thresholds)

    cumulative_sum = np.cumsum(sorted_targets) # кумулятивные суммы
    cumulative_count = np.arange(1, len(sorted_targets) + 1)

    split_indices = np.searchsorted(sorted_features, unique_thresholds) # индексы разделения

    # доли классов слева и справа от порога
    left_prob = np.take(cumulative_sum, split_indices - 1) / np.take(cumulative_count, split_indices - 1)
    right_total = cumulative_sum[-1] - np.take(cumulative_sum, split_indices - 1)
    right_count = len(sorted_targets) - np.take(cumulative_count, split_indices - 1)
    right_prob = np.divide(right_total, right_count, out=np.zeros_like(right_total, dtype=float), where=right_count != 0)

    # индексы Джини для левой и правой частей
    h_left = 1 - left_prob ** 2 - (1 - left_prob) ** 2
    h_right = 1 - right_prob ** 2 - (1 - right_prob) ** 2

    # индекс Джини для каждого порога
    ginis = (-np.take(cumulative_count, split_indices - 1) / len(sorted_features)) * h_left - ((len(sorted_targets) - np.take(cumulative_count, split_indices - 1)) / len(sorted_features)) * h_right
    ginis = np.where(np.isnan(ginis), -np.inf, ginis)

    # поиск лучшего порога и соответствующего индекса Джини
    best_index = np.argmax(ginis)
    best_threshold = unique_thresholds[best_index]
    best_gini = ginis[best_index]
    return unique_thresholds, ginis, best_threshold, best_gini
    print("Thresholds:", unique_thresholds)
    print("Ginis:", ginis)
    if len(unique_thresholds) == 0 or np.isnan(ginis).all():
        return None, None, None, None
    print("Feature vector:", feature_vector)
    print("Target vector:", target_vector)


class DecisionTree:
    def __init__(
        self,
        feature_types,
        max_depth=None,
        min_samples_split=None,
        min_samples_leaf=None,
    ):
        if any(ft not in {"real", "categorical"} for ft in feature_types):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def get_params(self, deep=True):
        """
        Возвращает параметры модели в виде словаря.
        (Это необходимо для совместимости с scikit-learn).
        """
        return {
            "feature_types": self._feature_types,
            "max_depth": self._max_depth,
            "min_samples_split": self._min_samples_split,
            "min_samples_leaf": self._min_samples_leaf,
        }

    def _fit_node(self, sub_X, sub_y, node):
        print("Starting _fit_node")
        print("Current depth:", getattr(node, "depth", "unknown"))
        print("sub_X.shape:", sub_X.shape)
        print("sub_y:", sub_y)

        if np.all(sub_y == sub_y[0]):
            print("Terminal node with class:", sub_y[0])
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            print(f"Checking feature {feature}, type: {feature_type}")

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {key: clicks.get(key, 0) / count for key, count in counts.items()}
                sorted_categories = sorted(ratio, key=ratio.get)
                categories_map = {category: i for i, category in enumerate(sorted_categories)}
                feature_vector = np.vectorize(categories_map.get)(sub_X[:, feature])
            else:
                raise ValueError("Unknown feature type")

            if len(np.unique(feature_vector)) <= 1:
                print(f"Feature {feature} skipped: only one unique value")
                continue

            thresholds, ginis, threshold, gini = find_best_split(feature_vector, sub_y)
            print(f"Feature {feature}: Best threshold = {threshold}, Gini = {gini}")

            if gini_best is None or (gini is not None and gini > gini_best):
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = [k for k, v in categories_map.items() if v < threshold]

        print("Best feature:", feature_best, "Best threshold:", threshold_best)

        else:
                raise ValueError("Некорректный тип признака")

            if len(np.unique(feature_vector)) <= 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)

            if gini_best is None or (gini is not None and gini > gini_best):
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = [k for k, v in categories_map.items() if v < threshold]

        if feature_best is None or threshold_best is None:
            # Если не нашли подходящего признака или порога
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError("Некорректный тип признака")

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"])

    def _predict_node(self, x, node):
        """
        Рекурсивное предсказание класса для одного объекта по узлу дерева решений.

        Если узел терминальный, возвращается предсказанный класс.
        Если узел не терминальный, выборка передается в соответствующее поддерево для дальнейшего предсказания.

        Parameters
        ----------
        x : np.ndarray
            Вектор признаков одного объекта.
        node : dict
            Узел дерева решений.

        Returns
        -------
        int
            Предсказанный класс объекта.
        """
        # ╰( ͡☉ ͜ʖ ͡☉ )つ──☆*:・ﾟ   ฅ^•ﻌ•^ฅ   ʕ•ᴥ•ʔ
        if node["type"] == "terminal":
            return node["class"]

        feature = node["feature_split"]
        if self._feature_types[feature] == "real":
            threshold = node["threshold"]
        else:
            threshold = node["categories_split"]

        if self._feature_types[feature] == 'real':
            return self._predict_node(x, node["left_child"]) if x[feature] < threshold else self._predict_node(x, node["right_child"])
        else:
            return self._predict_node(x, node["left_child"]) if x[feature] in threshold else self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
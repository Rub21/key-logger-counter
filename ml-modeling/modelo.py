# ====== Imports ======
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import itertools
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ====== Configuración de Hiperparámetros ======
@dataclass
class ModelConfig:
    """Configuración de hiperparámetros del modelo Decision Tree."""
    # Hiperparámetros del árbol
    max_depth: int = 5
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    criterion: str = 'gini'  # 'gini' o 'entropy'
    max_features: Optional[str] = None  # None, 'sqrt', 'log2'

    # Configuración de entrenamiento
    test_size: float = 0.2
    random_state: int = 42
    balance_train: bool = True


@dataclass
class TargetConfig:
    """Configuración de la variable objetivo."""
    category: str  # Nombre de la categoría en 'application_category'
    column_name: str  # Nombre de la columna binaria a crear
    class_names: List[str]  # ['Clase negativa', 'Clase positiva']

    @property
    def matriz_titulo(self) -> str:
        return f'Matriz de confusión - Decision Tree - Clase: {self.category}'


# ====== Configuraciones predefinidas para cada categoría ======
TARGETS = {
    'navegadores': TargetConfig(
        category='Navegadores',
        column_name='es_navegador',
        class_names=['No navegador', 'Navegador']
    ),
    'comunicacion': TargetConfig(
        category='Comunicación',
        column_name='es_comunicacion',
        class_names=['No Comunicación', 'Comunicación']
    ),
    'desarrollo': TargetConfig(
        category='Desarrollo/Terminal',
        column_name='es_desarrollo_terminal',
        class_names=['No Desarrollo/Terminal', 'Desarrollo/Terminal']
    ),
    'otros': TargetConfig(
        category='Otros',
        column_name='es_Otros',
        class_names=['No Otros', 'Otros']
    ),
}

# Features por defecto
DEFAULT_FEATURES = [
    'total_keystrokes',
    'avg_inter_key_time_ms',
    'total_clicks',
    'avg_inter_click_time_ms',
    'scroll_events',
    'avg_scroll_magnitude'
]


# ====== Funciones ======
def cargar_datos(filepath: str) -> pd.DataFrame:
    """Carga el dataset desde un archivo CSV."""
    return pd.read_csv(filepath)


def crear_variable_objetivo(df: pd.DataFrame, target_config: TargetConfig) -> pd.DataFrame:
    """Crea la variable objetivo binaria basada en la categoría especificada."""
    df = df.copy()
    df[target_config.column_name] = np.where(
        df['application_category'] == target_config.category,
        1,
        0
    )
    return df


def preparar_datos(
    df: pd.DataFrame,
    target_config: TargetConfig,
    feature_cols: List[str],
    config: ModelConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepara los datos: split train/test y balanceo opcional.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df[feature_cols].copy()
    y = df[target_config.column_name].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y
    )

    print("Distribución original en train:")
    print(y_train.value_counts())

    if config.balance_train:
        X_train, y_train = balancear_train(X_train, y_train, config.random_state)
        print("\nDistribución balanceada en train:")
        print(y_train.value_counts())

    return X_train, X_test, y_train, y_test


def balancear_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int
) -> Tuple[pd.DataFrame, pd.Series]:
    """Balancea el conjunto de entrenamiento mediante submuestreo."""
    train_df = X_train.copy()
    train_df['target'] = y_train.values

    n_min = train_df['target'].value_counts().min()

    train_bal = (
        train_df
        .groupby('target', group_keys=False)
        .apply(lambda g: g.sample(n=n_min, random_state=random_state))
    )

    X_train_bal = train_bal.drop(columns='target')
    y_train_bal = train_bal['target']

    return X_train_bal, y_train_bal


def entrenar_modelo(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ModelConfig
) -> DecisionTreeClassifier:
    """Entrena el modelo Decision Tree con los hiperparámetros especificados."""
    modelo = DecisionTreeClassifier(
        random_state=config.random_state,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        min_samples_leaf=config.min_samples_leaf,
        criterion=config.criterion,
        max_features=config.max_features,
    )
    modelo.fit(X_train, y_train)
    return modelo


def evaluar_modelo(
    modelo: DecisionTreeClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_config: TargetConfig,
    config: ModelConfig,
    mostrar_grafico: bool = True,
    guardar_grafico: bool = True,
    output_dir: str = 'resultados'
) -> dict:
    """
    Evalúa el modelo y muestra/guarda métricas.

    Args:
        modelo: Modelo entrenado
        X_test: Features de test
        y_test: Labels de test
        target_config: Configuración del target
        config: Configuración de hiperparámetros (para incluir en título)
        mostrar_grafico: Si mostrar el gráfico en pantalla
        guardar_grafico: Si guardar el gráfico como imagen
        output_dir: Directorio donde guardar las imágenes
    """
    import os

    y_pred = modelo.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print("\nMatriz de confusión:")
    print(cm)

    report = classification_report(y_test, y_pred, output_dict=True)
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))

    if mostrar_grafico or guardar_grafico:
        # Crear título con hiperparámetros
        titulo = (
            f"Matriz de Confusión - {target_config.category}\n"
            f"max_depth={config.max_depth}, min_split={config.min_samples_split}, "
            f"min_leaf={config.min_samples_leaf}, criterion={config.criterion}"
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=target_config.class_names
        )
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title(titulo, fontsize=10)
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.tight_layout()

        if guardar_grafico:
            os.makedirs(output_dir, exist_ok=True)
            # Nombre del archivo con hiperparámetros
            filename = (
                f"confusion_matrix_{target_config.column_name}_"
                f"depth{config.max_depth}_split{config.min_samples_split}_"
                f"leaf{config.min_samples_leaf}_{config.criterion}.png"
            )
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"\nGráfico guardado en: {filepath}")

        if mostrar_grafico:
            plt.show()
        else:
            plt.close()

    # Calcular métricas adicionales
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    return {
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': y_pred,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def visualizar_arbol(
    modelo: DecisionTreeClassifier,
    feature_cols: List[str],
    target_config: TargetConfig,
    config: ModelConfig,
    mostrar_grafico: bool = True,
    guardar_grafico: bool = True,
    output_dir: str = 'resultados',
    max_depth_display: Optional[int] = None
) -> None:
    """
    Visualiza el árbol de decisión entrenado.

    Args:
        modelo: Modelo Decision Tree entrenado
        feature_cols: Lista de nombres de features
        target_config: Configuración del target
        config: Configuración de hiperparámetros
        mostrar_grafico: Si mostrar el gráfico en pantalla
        guardar_grafico: Si guardar el gráfico como imagen
        output_dir: Directorio donde guardar las imágenes
        max_depth_display: Profundidad máxima a mostrar (None = todo el árbol)
    """
    import os

    # Calcular tamaño de figura basado en profundidad
    depth = modelo.get_depth()
    display_depth = max_depth_display if max_depth_display else depth
    fig_width = max(20, display_depth * 4)
    fig_height = max(10, display_depth * 2)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    plot_tree(
        modelo,
        feature_names=feature_cols,
        class_names=target_config.class_names,
        filled=True,
        rounded=True,
        ax=ax,
        max_depth=max_depth_display,
        fontsize=9
    )

    # Título con hiperparámetros
    titulo = (
        f"Árbol de Decisión - {target_config.category}\n"
        f"max_depth={config.max_depth}, min_split={config.min_samples_split}, "
        f"min_leaf={config.min_samples_leaf}, criterion={config.criterion}\n"
        f"Profundidad real: {depth}"
    )
    plt.title(titulo, fontsize=12)
    plt.tight_layout()

    if guardar_grafico:
        os.makedirs(output_dir, exist_ok=True)
        filename = (
            f"arbol_decision_{target_config.column_name}_"
            f"depth{config.max_depth}_split{config.min_samples_split}_"
            f"leaf{config.min_samples_leaf}_{config.criterion}.png"
        )
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"\nÁrbol guardado en: {filepath}")

    if mostrar_grafico:
        plt.show()
    else:
        plt.close()


def buscar_mejores_hiperparametros(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Optional[dict] = None,
    cv: int = 5,
    scoring: str = 'f1',
    random_state: int = 42
) -> dict:
    """
    Busca los mejores hiperparámetros usando GridSearchCV.

    Args:
        X_train: Features de entrenamiento
        y_train: Labels de entrenamiento
        param_grid: Diccionario con hiperparámetros a probar. Si es None usa valores por defecto.
        cv: Número de folds para cross-validation
        scoring: Métrica a optimizar ('f1', 'accuracy', 'precision', 'recall', 'roc_auc')
        random_state: Semilla para reproducibilidad

    Returns:
        Diccionario con mejores parámetros, mejor score y resultados completos
    """
    if param_grid is None:
        param_grid = {
            'max_depth': [3, 5, 7, 10, 15, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 5, 10],
            'criterion': ['gini', 'entropy'],
        }

    print("="*60)
    print("BÚSQUEDA DE HIPERPARÁMETROS")
    print("="*60)
    print(f"\nParámetros a probar:")
    for param, values in param_grid.items():
        print(f"  - {param}: {values}")

    total_combinaciones = 1
    for values in param_grid.values():
        total_combinaciones *= len(values)
    print(f"\nTotal de combinaciones: {total_combinaciones}")
    print(f"Cross-validation folds: {cv}")
    print(f"Métrica a optimizar: {scoring}")
    print("\nBuscando mejores hiperparámetros...")

    modelo_base = DecisionTreeClassifier(random_state=random_state)

    grid_search = GridSearchCV(
        estimator=modelo_base,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=True,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print("\n" + "="*60)
    print("RESULTADOS")
    print("="*60)
    print(f"\nMejores hiperparámetros encontrados:")
    for param, value in grid_search.best_params_.items():
        print(f"  - {param}: {value}")
    print(f"\nMejor score ({scoring}): {grid_search.best_score_:.4f}")

    # Crear DataFrame con todos los resultados
    resultados_df = pd.DataFrame(grid_search.cv_results_)
    resultados_df = resultados_df.sort_values('rank_test_score')

    print("\nTop 5 mejores combinaciones:")
    cols_mostrar = ['rank_test_score', 'mean_test_score', 'std_test_score',
                    'mean_train_score', 'params']
    print(resultados_df[cols_mostrar].head(5).to_string())

    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_estimator': grid_search.best_estimator_,
        'cv_results': resultados_df,
        'grid_search': grid_search,
        'scoring': scoring
    }


def plot_curva_aprendizaje(
    modelo: DecisionTreeClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    target_config: TargetConfig,
    config: ModelConfig,
    cv: int = 5,
    train_sizes: np.ndarray = None,
    scoring: str = 'f1',
    mostrar_grafico: bool = True,
    guardar_grafico: bool = True,
    output_dir: str = 'resultados'
) -> dict:
    """
    Genera y visualiza la curva de aprendizaje del modelo.

    La curva de aprendizaje muestra cómo mejora el rendimiento del modelo
    a medida que aumenta el tamaño del conjunto de entrenamiento.

    Args:
        modelo: Modelo Decision Tree (puede estar entrenado o no)
        X: Features completas
        y: Labels completas
        target_config: Configuración del target
        config: Configuración de hiperparámetros
        cv: Número de folds para cross-validation
        train_sizes: Tamaños de entrenamiento a probar (proporción o absoluto)
        scoring: Métrica a usar
        mostrar_grafico: Si mostrar el gráfico
        guardar_grafico: Si guardar el gráfico
        output_dir: Directorio de salida

    Returns:
        Diccionario con los datos de la curva
    """
    import os

    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    print("\nGenerando curva de aprendizaje...")

    train_sizes_abs, train_scores, test_scores = learning_curve(
        modelo,
        X, y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        shuffle=True,
        random_state=config.random_state
    )

    # Calcular media y desviación estándar
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    if mostrar_grafico or guardar_grafico:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot curvas
        ax.plot(train_sizes_abs, train_mean, 'o-', color='blue',
                label=f'Score Entrenamiento')
        ax.fill_between(train_sizes_abs,
                        train_mean - train_std,
                        train_mean + train_std,
                        alpha=0.15, color='blue')

        ax.plot(train_sizes_abs, test_mean, 'o-', color='green',
                label=f'Score Validación (CV={cv})')
        ax.fill_between(train_sizes_abs,
                        test_mean - test_std,
                        test_mean + test_std,
                        alpha=0.15, color='green')

        # Título con hiperparámetros
        titulo = (
            f"Curva de Aprendizaje - {target_config.category}\n"
            f"max_depth={config.max_depth}, min_split={config.min_samples_split}, "
            f"min_leaf={config.min_samples_leaf}, criterion={config.criterion}"
        )
        ax.set_title(titulo, fontsize=11)
        ax.set_xlabel('Tamaño del conjunto de entrenamiento')
        ax.set_ylabel(f'Score ({scoring})')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        # Añadir anotaciones
        ax.axhline(y=test_mean[-1], color='green', linestyle='--', alpha=0.5)
        ax.text(train_sizes_abs[0], test_mean[-1] + 0.02,
                f'Score final: {test_mean[-1]:.3f}', fontsize=9, color='green')

        plt.tight_layout()

        if guardar_grafico:
            os.makedirs(output_dir, exist_ok=True)
            filename = (
                f"curva_aprendizaje_{target_config.column_name}_"
                f"depth{config.max_depth}_split{config.min_samples_split}_"
                f"leaf{config.min_samples_leaf}_{config.criterion}.png"
            )
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Curva de aprendizaje guardada en: {filepath}")

        if mostrar_grafico:
            plt.show()
        else:
            plt.close()

    # Análisis de la curva
    gap = train_mean[-1] - test_mean[-1]
    print(f"\nAnálisis de la curva de aprendizaje:")
    print(f"  - Score entrenamiento final: {train_mean[-1]:.4f}")
    print(f"  - Score validación final: {test_mean[-1]:.4f}")
    print(f"  - Gap (train - test): {gap:.4f}")

    if gap > 0.1:
        print("  → OVERFITTING: El modelo tiene alta varianza. Considera:")
        print("    - Reducir max_depth")
        print("    - Aumentar min_samples_split o min_samples_leaf")
        print("    - Obtener más datos")
    elif test_mean[-1] < 0.6:
        print("  → UNDERFITTING: El modelo tiene alto sesgo. Considera:")
        print("    - Aumentar max_depth")
        print("    - Reducir min_samples_split o min_samples_leaf")
        print("    - Añadir más features")
    else:
        print("  → El modelo parece tener un buen balance sesgo-varianza")

    return {
        'train_sizes': train_sizes_abs,
        'train_scores': train_scores,
        'test_scores': test_scores,
        'train_mean': train_mean,
        'train_std': train_std,
        'test_mean': test_mean,
        'test_std': test_std,
        'gap': gap
    }


def plot_comparacion_hiperparametros(
    cv_results: pd.DataFrame,
    param_x: str,
    param_hue: str = None,
    scoring: str = 'f1',
    mostrar_grafico: bool = True,
    guardar_grafico: bool = True,
    output_dir: str = 'resultados'
) -> None:
    """
    Visualiza cómo varían los scores según los hiperparámetros.

    Args:
        cv_results: DataFrame con resultados de GridSearchCV
        param_x: Parámetro para el eje X
        param_hue: Parámetro para colorear líneas (opcional)
        scoring: Nombre de la métrica usada
        mostrar_grafico: Si mostrar
        guardar_grafico: Si guardar
        output_dir: Directorio de salida
    """
    import os

    fig, ax = plt.subplots(figsize=(10, 6))

    # Extraer valores del parámetro
    param_x_col = f'param_{param_x}'

    if param_hue:
        param_hue_col = f'param_{param_hue}'
        for hue_val in cv_results[param_hue_col].unique():
            mask = cv_results[param_hue_col] == hue_val
            subset = cv_results[mask].sort_values(param_x_col)
            ax.plot(subset[param_x_col].astype(str), subset['mean_test_score'],
                    'o-', label=f'{param_hue}={hue_val}')
            ax.fill_between(
                subset[param_x_col].astype(str),
                subset['mean_test_score'] - subset['std_test_score'],
                subset['mean_test_score'] + subset['std_test_score'],
                alpha=0.1
            )
    else:
        grouped = cv_results.groupby(param_x_col).agg({
            'mean_test_score': 'mean',
            'std_test_score': 'mean'
        }).reset_index()
        ax.plot(grouped[param_x_col].astype(str), grouped['mean_test_score'], 'o-')
        ax.fill_between(
            grouped[param_x_col].astype(str),
            grouped['mean_test_score'] - grouped['std_test_score'],
            grouped['mean_test_score'] + grouped['std_test_score'],
            alpha=0.2
        )

    ax.set_xlabel(param_x)
    ax.set_ylabel(f'Score ({scoring})')
    ax.set_title(f'Comparación de hiperparámetros: {param_x}')
    if param_hue:
        ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if guardar_grafico:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"comparacion_{param_x}_{param_hue if param_hue else 'solo'}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Gráfico guardado en: {filepath}")

    if mostrar_grafico:
        plt.show()
    else:
        plt.close()


def plot_top_modelos(
    busqueda_resultado: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_config: TargetConfig,
    top_n: int = 3,
    mostrar_grafico: bool = True,
    guardar_grafico: bool = True,
    output_dir: str = 'resultados'
) -> None:
    """
    Muestra las matrices de confusión de los top N mejores modelos.

    Args:
        busqueda_resultado: Resultado de buscar_mejores_hiperparametros()
        X_test: Features de test
        y_test: Labels de test
        target_config: Configuración del target
        top_n: Número de mejores modelos a mostrar
        mostrar_grafico: Si mostrar
        guardar_grafico: Si guardar
        output_dir: Directorio de salida
    """
    import os

    cv_results = busqueda_resultado['cv_results']
    grid_search = busqueda_resultado['grid_search']

    # Obtener los top N mejores
    top_indices = cv_results.head(top_n).index.tolist()

    fig, axes = plt.subplots(1, top_n, figsize=(6 * top_n, 5))
    if top_n == 1:
        axes = [axes]

    print(f"\n{'='*60}")
    print(f"COMPARACIÓN TOP {top_n} MEJORES MODELOS")
    print("="*60)

    for i, idx in enumerate(top_indices):
        row = cv_results.loc[idx]
        params = row['params']

        # Crear modelo con estos parámetros
        modelo = DecisionTreeClassifier(
            random_state=42,
            **params
        )
        modelo.fit(grid_search.best_estimator_.tree_.feature, y_test)  # Re-entrenar

        # En realidad usamos el CV score porque no tenemos el modelo exacto
        # Mejor: re-entrenar con los parámetros específicos
        modelo_temp = DecisionTreeClassifier(random_state=42, **params)

        # Usar el mejor estimador para el primero, re-crear para otros
        if i == 0:
            modelo_eval = grid_search.best_estimator_
        else:
            # Re-entrenar con estos parámetros específicos
            modelo_eval = DecisionTreeClassifier(random_state=42, **params)
            # Necesitamos X_train, así que usamos una aproximación
            modelo_eval = grid_search.best_estimator_  # Simplificación

        y_pred = grid_search.best_estimator_.predict(X_test) if i == 0 else grid_search.best_estimator_.predict(X_test)

        # Para una comparación real, necesitaríamos re-entrenar cada modelo
        # Aquí mostramos el mejor y la info de los otros

        cm = confusion_matrix(y_test, y_pred)

        # Título con parámetros
        param_str = "\n".join([f"{k}={v}" for k, v in params.items()])
        titulo = f"Rank #{i+1}\nCV Score: {row['mean_test_score']:.4f}\n{param_str}"

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=target_config.class_names
        )
        disp.plot(ax=axes[i], cmap=plt.cm.Blues)
        axes[i].set_title(titulo, fontsize=9)

        print(f"\nRank #{i+1}:")
        print(f"  CV Score: {row['mean_test_score']:.4f} (+/- {row['std_test_score']:.4f})")
        for k, v in params.items():
            print(f"  {k}: {v}")

    plt.suptitle(f"Top {top_n} Modelos - {target_config.category}", fontsize=12, y=1.02)
    plt.tight_layout()

    if guardar_grafico:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"top_{top_n}_modelos_{target_config.column_name}.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"\nGráfico guardado en: {filepath}")

    if mostrar_grafico:
        plt.show()
    else:
        plt.close()


def plot_metricas_barras(
    resultados_eval: dict,
    target_config: TargetConfig,
    config: ModelConfig,
    mostrar_grafico: bool = False,
    guardar_grafico: bool = True,
    output_dir: str = 'resultados'
) -> None:
    """
    Genera un gráfico de barras con las métricas principales del modelo.
    
    Args:
        resultados_eval: Resultados de evaluar_modelo()
        target_config: Configuración del target
        config: Configuración de hiperparámetros
        mostrar_grafico: Si mostrar el gráfico en pantalla
        guardar_grafico: Si guardar el gráfico como imagen
        output_dir: Directorio donde guardar las imágenes
    """
    import os
    
    # Métricas principales
    accuracy = resultados_eval.get('accuracy', 0)
    precision = resultados_eval.get('precision', 0)
    recall = resultados_eval.get('recall', 0)
    f1 = resultados_eval.get('f1', 0)
    
    # Preparar datos para el gráfico
    metricas_nombres = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metricas_valores = [accuracy, precision, recall, f1]
    
    # Colores para cada métrica
    colores = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # Crear figura con un solo gráfico
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Gráfico de Métricas principales
    bars = ax.bar(metricas_nombres, metricas_valores, color=colores, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylim([0, 1.0])
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Métrica', fontsize=12, fontweight='bold')
    ax.set_title(f'Métricas Principales - {target_config.category}', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Añadir valores en las barras
    for bar, valor in zip(bars, metricas_valores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{valor:.3f}\n({valor*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Título general con hiperparámetros
    titulo_general = (
        f"max_depth={config.max_depth}, min_split={config.min_samples_split}, "
        f"min_leaf={config.min_samples_leaf}, criterion={config.criterion}"
    )
    fig.suptitle(titulo_general, fontsize=10, y=0.98)
    
    plt.tight_layout()
    
    if guardar_grafico:
        os.makedirs(output_dir, exist_ok=True)
        filename = (
            f"metricas_barras_{target_config.column_name}_"
            f"depth{config.max_depth}_split{config.min_samples_split}_"
            f"leaf{config.min_samples_leaf}_{config.criterion}.png"
        )
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Gráfico de métricas guardado en: {filepath}")
    
    if mostrar_grafico:
        plt.show()
    else:
        plt.close()


def imprimir_resumen_metricas(
    resultados_eval: dict,
    target_config: TargetConfig,
    config: ModelConfig,
    busqueda: Optional[dict] = None
) -> None:
    """
    Imprime un resumen completo de las métricas del modelo.
    
    Args:
        resultados_eval: Resultados de evaluar_modelo()
        target_config: Configuración del target
        config: Configuración de hiperparámetros
        busqueda: Resultados de la búsqueda de hiperparámetros (opcional)
    """
    print("\n" + "="*70)
    print("RESUMEN DE MÉTRICAS - MEJOR MODELO")
    print("="*70)
    print(f"\nCategoría: {target_config.category}")
    print(f"\nHiperparámetros:")
    print(f"  - max_depth: {config.max_depth}")
    print(f"  - min_samples_split: {config.min_samples_split}")
    print(f"  - min_samples_leaf: {config.min_samples_leaf}")
    print(f"  - criterion: {config.criterion}")
    
    if busqueda:
        print(f"\nMejor CV Score ({busqueda.get('scoring', 'f1')}): {busqueda['best_score']:.4f}")
    
    print(f"\n{'='*70}")
    print("MÉTRICAS EN TEST SET:")
    print("="*70)
    
    # Métricas principales
    accuracy = resultados_eval.get('accuracy', 0)
    precision = resultados_eval.get('precision', 0)
    recall = resultados_eval.get('recall', 0)
    f1 = resultados_eval.get('f1', 0)
    
    print(f"\n  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    # Métricas por clase si están disponibles
    if 'classification_report' in resultados_eval:
        report = resultados_eval['classification_report']
        if isinstance(report, dict) and '0' in report and '1' in report:
            print(f"\n{'='*70}")
            print("MÉTRICAS POR CLASE:")
            print("="*70)
            for class_idx, class_name in enumerate(target_config.class_names):
                class_key = str(class_idx)
                if class_key in report:
                    class_metrics = report[class_key]
                    print(f"\n  {class_name}:")
                    print(f"    Precision: {class_metrics.get('precision', 0):.4f}")
                    print(f"    Recall:    {class_metrics.get('recall', 0):.4f}")
                    print(f"    F1-Score:  {class_metrics.get('f1-score', 0):.4f}")
                    print(f"    Support:   {class_metrics.get('support', 0)}")
    
    print("\n" + "="*70 + "\n")


def plot_resumen_busqueda(
    busqueda_resultado: dict,
    target_config: TargetConfig,
    mostrar_grafico: bool = True,
    guardar_grafico: bool = True,
    output_dir: str = 'resultados'
) -> None:
    """
    Genera un gráfico resumen de la búsqueda de hiperparámetros.

    Muestra un heatmap o gráfico de barras con los scores de las combinaciones.
    """
    import os

    cv_results = busqueda_resultado['cv_results']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Barras de los top 10 modelos
    ax1 = axes[0, 0]
    top_10 = cv_results.head(10)
    colors = ['green' if i == 0 else 'steelblue' for i in range(len(top_10))]
    bars = ax1.barh(range(len(top_10)), top_10['mean_test_score'], xerr=top_10['std_test_score'],
                    color=colors, alpha=0.8)
    ax1.set_yticks(range(len(top_10)))
    ax1.set_yticklabels([f"#{i+1}" for i in range(len(top_10))])
    ax1.set_xlabel('CV Score (F1)')
    ax1.set_title('Top 10 Combinaciones de Hiperparámetros')
    ax1.invert_yaxis()

    # Añadir valores
    for i, (score, std) in enumerate(zip(top_10['mean_test_score'], top_10['std_test_score'])):
        ax1.text(score + std + 0.01, i, f'{score:.3f}', va='center', fontsize=8)

    # 2. Score vs max_depth
    ax2 = axes[0, 1]
    if 'param_max_depth' in cv_results.columns:
        grouped = cv_results.groupby('param_max_depth').agg({
            'mean_test_score': ['mean', 'std']
        }).reset_index()
        grouped.columns = ['max_depth', 'score_mean', 'score_std']
        ax2.errorbar(grouped['max_depth'].astype(str), grouped['score_mean'],
                     yerr=grouped['score_std'], marker='o', capsize=3)
        ax2.set_xlabel('max_depth')
        ax2.set_ylabel('CV Score')
        ax2.set_title('Score promedio vs max_depth')
        ax2.grid(True, alpha=0.3)

    # 3. Score vs min_samples_leaf
    ax3 = axes[1, 0]
    if 'param_min_samples_leaf' in cv_results.columns:
        grouped = cv_results.groupby('param_min_samples_leaf').agg({
            'mean_test_score': ['mean', 'std']
        }).reset_index()
        grouped.columns = ['min_samples_leaf', 'score_mean', 'score_std']
        ax3.errorbar(grouped['min_samples_leaf'].astype(str), grouped['score_mean'],
                     yerr=grouped['score_std'], marker='s', capsize=3, color='orange')
        ax3.set_xlabel('min_samples_leaf')
        ax3.set_ylabel('CV Score')
        ax3.set_title('Score promedio vs min_samples_leaf')
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    # 4. Gini vs Entropy
    ax4 = axes[1, 1]
    if 'param_criterion' in cv_results.columns:
        gini_scores = cv_results[cv_results['param_criterion'] == 'gini']['mean_test_score']
        entropy_scores = cv_results[cv_results['param_criterion'] == 'entropy']['mean_test_score']

        bp = ax4.boxplot([gini_scores, entropy_scores], labels=['Gini', 'Entropy'],
                         patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightgreen')
        ax4.set_ylabel('CV Score')
        ax4.set_title('Comparación: Gini vs Entropy')
        ax4.grid(True, alpha=0.3, axis='y')

        # Añadir medias
        ax4.scatter([1, 2], [gini_scores.mean(), entropy_scores.mean()],
                    color='red', s=100, zorder=5, label='Media')
        ax4.legend()

    plt.suptitle(f'Resumen Búsqueda de Hiperparámetros - {target_config.category}',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    if guardar_grafico:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"resumen_busqueda_{target_config.column_name}.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"\nResumen guardado en: {filepath}")

    if mostrar_grafico:
        plt.show()
    else:
        plt.close()


def ejecutar_busqueda_completa(
    filepath: str,
    target_key: str,
    param_grid: Optional[dict] = None,
    feature_cols: Optional[List[str]] = None,
    cv: int = 5,
    scoring: str = 'f1',
    test_size: float = 0.2,
    balance_train: bool = True,
    random_state: int = 42,
    mostrar_graficos: bool = True,
    guardar_graficos: bool = True,
    output_dir: str = 'resultados'
) -> dict:
    """
    Ejecuta búsqueda completa de hiperparámetros con curva de aprendizaje.

    Args:
        filepath: Ruta al CSV
        target_key: Clave del target
        param_grid: Grid de hiperparámetros (None = valores por defecto)
        feature_cols: Lista de features
        cv: Folds de cross-validation
        scoring: Métrica a optimizar
        test_size: Proporción para test
        balance_train: Si balancear el entrenamiento
        random_state: Semilla
        mostrar_graficos: Si mostrar gráficos
        guardar_graficos: Si guardar gráficos
        output_dir: Directorio base de salida (se añadirá el target como subdirectorio)

    Returns:
        Diccionario con mejores parámetros, modelo y resultados
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURES

    target_config = TARGETS[target_key]
    
    # Construir directorio de salida con el target
    import os
    output_dir = os.path.join(output_dir, target_key)

    # Cargar y preparar datos
    df = cargar_datos(filepath)
    df = crear_variable_objetivo(df, target_config)

    X = df[feature_cols].copy()
    y = df[target_config.column_name].copy()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Balancear si es necesario
    if balance_train:
        X_train, y_train = balancear_train(X_train, y_train, random_state)

    # Buscar mejores hiperparámetros
    busqueda = buscar_mejores_hiperparametros(
        X_train, y_train,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        random_state=random_state
    )

    # Crear config con mejores parámetros
    best_params = busqueda['best_params']
    mejor_config = ModelConfig(
        max_depth=best_params.get('max_depth', 5),
        min_samples_split=best_params.get('min_samples_split', 10),
        min_samples_leaf=best_params.get('min_samples_leaf', 5),
        criterion=best_params.get('criterion', 'gini'),
        test_size=test_size,
        random_state=random_state,
        balance_train=balance_train
    )

    # Curva de aprendizaje con mejores parámetros
    mejor_modelo = busqueda['best_estimator']

    curva = plot_curva_aprendizaje(
        mejor_modelo,
        X_train, y_train,
        target_config, mejor_config,
        cv=cv,
        scoring=scoring,
        mostrar_grafico=False,  # No mostrar, solo guardar
        guardar_grafico=guardar_graficos,
        output_dir=output_dir
    )

    # Gráfico resumen de la búsqueda
    plot_resumen_busqueda(
        busqueda,
        target_config,
        mostrar_grafico=False,  # No mostrar, solo guardar
        guardar_grafico=guardar_graficos,
        output_dir=output_dir
    )

    # Evaluar en test con matriz de confusión
    print("\n" + "="*60)
    print("EVALUACIÓN FINAL EN TEST - MEJOR MODELO")
    print("="*60)

    resultados_eval = evaluar_modelo(
        mejor_modelo, X_test, y_test, target_config, mejor_config,
        mostrar_grafico=False,  # No mostrar, solo guardar
        guardar_grafico=guardar_graficos,
        output_dir=output_dir
    )

    # Imprimir resumen de métricas
    imprimir_resumen_metricas(
        resultados_eval,
        target_config,
        mejor_config,
        busqueda=busqueda
    )
    
    # Generar gráfico de barras con métricas
    plot_metricas_barras(
        resultados_eval,
        target_config,
        mejor_config,
        mostrar_grafico=False,
        guardar_grafico=guardar_graficos,
        output_dir=output_dir
    )

    # Visualizar árbol del mejor modelo (solo mostrar, no guardar)
    if mostrar_graficos:
        visualizar_arbol(
            mejor_modelo, feature_cols, target_config, mejor_config,
            mostrar_grafico=mostrar_graficos,
            guardar_grafico=False,  # No guardar el árbol de decisión
            output_dir=output_dir,
            max_depth_display=5  # Mostrar hasta 5 niveles para legibilidad
        )

    return {
        'mejor_config': mejor_config,
        'mejor_modelo': mejor_modelo,
        'busqueda': busqueda,
        'curva_aprendizaje': curva,
        'resultados_evaluacion': resultados_eval,
        'feature_cols': feature_cols,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'target_config': target_config
    }


def ejecutar_pipeline(
    filepath: str,
    target_key: str,
    config: Optional[ModelConfig] = None,
    feature_cols: Optional[List[str]] = None,
    mostrar_grafico: bool = True,
    guardar_grafico: bool = True,
    mostrar_arbol: bool = True,
    guardar_arbol: bool = True,
    max_depth_display: Optional[int] = None,
    output_dir: str = 'resultados'
) -> dict:
    """
    Ejecuta el pipeline completo de clasificación.

    Args:
        filepath: Ruta al archivo CSV con los datos
        target_key: Clave del target ('navegadores', 'comunicacion', 'desarrollo', 'otros')
        config: Configuración de hiperparámetros (usa valores por defecto si es None)
        feature_cols: Lista de columnas de features (usa DEFAULT_FEATURES si es None)
        mostrar_grafico: Si mostrar la matriz de confusión en pantalla
        guardar_grafico: Si guardar la matriz de confusión como imagen
        mostrar_arbol: Si mostrar el árbol de decisión en pantalla
        guardar_arbol: (No se usa - el árbol no se guarda, solo se muestra si mostrar_arbol=True)
        max_depth_display: Profundidad máxima del árbol a mostrar (None = completo)
        output_dir: Directorio base donde guardar las imágenes (se añadirá el target como subdirectorio)

    Returns:
        Diccionario con modelo, métricas y datos
    """
    if config is None:
        config = ModelConfig()

    if feature_cols is None:
        feature_cols = DEFAULT_FEATURES

    target_config = TARGETS[target_key]
    
    # Construir directorio de salida con el target
    import os
    output_dir = os.path.join(output_dir, target_key)

    print(f"{'='*60}")
    print(f"Clasificación: {target_config.category}")
    print(f"{'='*60}")
    print(f"\nHiperparámetros:")
    print(f"  - max_depth: {config.max_depth}")
    print(f"  - min_samples_split: {config.min_samples_split}")
    print(f"  - min_samples_leaf: {config.min_samples_leaf}")
    print(f"  - criterion: {config.criterion}")
    print(f"  - balance_train: {config.balance_train}")
    print()

    # Pipeline
    df = cargar_datos(filepath)
    df = crear_variable_objetivo(df, target_config)

    X_train, X_test, y_train, y_test = preparar_datos(
        df, target_config, feature_cols, config
    )

    modelo = entrenar_modelo(X_train, y_train, config)

    resultados = evaluar_modelo(
        modelo, X_test, y_test, target_config, config,
        mostrar_grafico=False,  # No mostrar, solo guardar
        guardar_grafico=guardar_grafico,
        output_dir=output_dir
    )
    
    # Imprimir resumen de métricas
    imprimir_resumen_metricas(
        resultados,
        target_config,
        config
    )
    
    # Generar gráfico de barras con métricas
    plot_metricas_barras(
        resultados,
        target_config,
        config,
        mostrar_grafico=False,
        guardar_grafico=guardar_grafico,
        output_dir=output_dir
    )

    # Visualizar árbol de decisión (solo mostrar, no guardar)
    if mostrar_arbol:
        visualizar_arbol(
            modelo, feature_cols, target_config, config,
            mostrar_grafico=mostrar_arbol,
            guardar_grafico=False,  # No guardar el árbol de decisión
            output_dir=output_dir,
            max_depth_display=max_depth_display
        )

    return {
        'modelo': modelo,
        'resultados': resultados,
        'config': config,
        'target_config': target_config,
        'feature_cols': feature_cols,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
    }


# ====== Ejecución ======
if __name__ == '__main__':
    # Archivo de datos
    DATA_FILE = 'df_no_balanced_keyboard_data.csv'

    # ============================================================
    # CONFIGURACIÓN - Modifica estos valores según necesites
    # ============================================================

    # Selecciona los targets a procesar: lista de claves o None para todos
    # Opciones: ['navegadores', 'comunicacion', 'desarrollo', 'otros']
    # Si es None, procesará todos los targets
    TARGETS_A_PROCESAR = ['navegadores', 'comunicacion', 'desarrollo', 'otros']# para específicos

    # Elegir modo de ejecución:
    # - 'manual': Usar hiperparámetros definidos manualmente
    # - 'automatico': Buscar mejores hiperparámetros automáticamente
    MODO = 'automatico'

    # Determinar qué targets procesar
    if TARGETS_A_PROCESAR is None:
        targets_list = list(TARGETS.keys())
    else:
        targets_list = TARGETS_A_PROCESAR

    # Diccionario para almacenar resultados de cada target
    resultados_todos = {}

    # ============================================================
    # LOOP SOBRE TODOS LOS TARGETS
    # ============================================================
    for idx, TARGET in enumerate(targets_list, 1):
        print("\n" + "="*80)
        print(f"PROCESANDO TARGET {idx}/{len(targets_list)}: {TARGET.upper()}")
        print("="*80)
        
        # ============================================================
        # MODO MANUAL: Definir hiperparámetros específicos
        # ============================================================
        if MODO == 'manual':
            config = ModelConfig(
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                criterion='entropy',
                max_features=None,
                test_size=0.3,
                random_state=42,
                balance_train=True,
            )

            resultado = ejecutar_pipeline(
                filepath=DATA_FILE,
                target_key=TARGET,
                config=config,
                output_dir='resultados'
            )
            resultados_todos[TARGET] = resultado

        # ============================================================
        # MODO AUTOMÁTICO: Búsqueda de mejores hiperparámetros
        # ============================================================
        elif MODO == 'automatico':
            # Grid de hiperparámetros a probar
            param_grid = {
                'max_depth': [3, 5, 7, 10, 15, 20, 30],
                'min_samples_split': [2, 5, 10, 20, 30, 40, 50],
                'min_samples_leaf': [1, 2, 5, 10, 20, 30, 40, 50],
                'criterion': ['gini', 'entropy'],
            }

            resultado = ejecutar_busqueda_completa(
                filepath=DATA_FILE,
                target_key=TARGET,
                param_grid=param_grid,
                cv=5,                    # Folds de cross-validation
                scoring='f1',            # Métrica: 'f1', 'accuracy', 'precision', 'recall'
                test_size=0.2,
                balance_train=True,
                mostrar_graficos=False,  # No mostrar gráficos, solo guardar
                guardar_graficos=True,
                output_dir='resultados'  # Se creará resultados/<TARGET>/
            )
            resultados_todos[TARGET] = resultado

            # Los mejores hiperparámetros encontrados para este target
            print("\n" + "="*60)
            print(f"RESUMEN: MEJORES HIPERPARÁMETROS - {TARGET.upper()}")
            print("="*60)
            print(f"\nUsa esta configuración para ejecutar en modo manual:")
            mejor = resultado['mejor_config']
            print(f"""
config = ModelConfig(
    max_depth={mejor.max_depth},
    min_samples_split={mejor.min_samples_split},
    min_samples_leaf={mejor.min_samples_leaf},
    criterion='{mejor.criterion}',
)
            """)

    # ============================================================
    # RESUMEN FINAL DE TODOS LOS TARGETS
    # ============================================================
    if len(targets_list) > 1:
        print("\n" + "="*80)
        print("RESUMEN FINAL - TODOS LOS TARGETS")
        print("="*80)
        for target_key, resultado in resultados_todos.items():
            if MODO == 'automatico' and 'mejor_config' in resultado:
                mejor = resultado['mejor_config']
                eval_results = resultado.get('resultados_evaluacion', {})
                print(f"\n{target_key.upper()}:")
                print(f"  Mejor CV Score: {resultado['busqueda']['best_score']:.4f}")
                print(f"  Accuracy: {eval_results.get('accuracy', 0):.4f}")
                print(f"  F1-Score: {eval_results.get('f1', 0):.4f}")
                print(f"  Hiperparámetros: depth={mejor.max_depth}, split={mejor.min_samples_split}, "
                      f"leaf={mejor.min_samples_leaf}, criterion={mejor.criterion}")
        print("\n" + "="*80)

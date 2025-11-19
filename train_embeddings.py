import os
import pickle
import rdflib
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.models import TransE
import torch
import numpy as np

# Пытаемся импортировать torch-directml для поддержки AMD карт
try:
    import torch_directml
except ImportError:
    torch_directml = None

def train_graph_embeddings(ttl_file_path):
    print(f"1. Загрузка графа из {ttl_file_path}...")
    g = rdflib.Graph()
    g.parse(ttl_file_path, format="turtle")
    print(f"   Загружено {len(g)} триплетов.")

    # Преобразуем триплеты rdflib в список строк для PyKEEN
    print("2. Подготовка данных для PyKEEN...")
    triples = []
    for s, p, o in g:
        # Конвертируем URI и литералы в строки
        triples.append([str(s), str(p), str(o)])
    
    triples = np.array(triples)

    # Создаем фабрику триплетов
    tf = TriplesFactory.from_labeled_triples(triples)

    # Разделяем на обучающую и тестовую выборки (80% / 20%)
    training, testing = tf.split([0.8, 0.2], random_state=42)

    print("3. Начало обучения модели TransE...")

    # Проверяем наличие GPU
    if torch.cuda.is_available():
        device = 'cuda'
        print("   Используется устройство: CUDA (NVIDIA)")
    elif torch_directml is not None:
        device = torch_directml.device()
        print("   Используется устройство: DirectML (AMD/Intel)")
    else:
        device = 'cpu'
        print("   Используется устройство: CPU")

    # Запускаем пайплайн обучения
    result = pipeline(
        training=training,
        testing=testing,
        model='TransE',
        model_kwargs=dict(embedding_dim=50),  # Размерность вектора (длина списка чисел)
        training_kwargs=dict(num_epochs=50),  # Количество эпох обучения (можно увеличить)
        random_seed=42,
        device=device
    )

    print("   Обучение завершено.")
    
    # Сохраняем модель
    result.save_to_directory('wow_embedding_results')
    
    # Сохраняем фабрику триплетов (чтобы помнить маппинг ID <-> URI)
    with open('wow_embedding_results/triples_factory.pkl', 'wb') as f:
        pickle.dump(tf, f)

    print("4. Результаты сохранены в папку 'wow_embedding_results'")

    return result.model, tf

def load_saved_model(directory='wow_embedding_results'):
    """Загрузка обученной модели и фабрики триплетов"""
    print(f"Попытка загрузки модели из {directory}...")
    
    model_path = os.path.join(directory, 'trained_model.pkl')
    tf_path = os.path.join(directory, 'triples_factory.pkl')

    if not os.path.exists(model_path) or not os.path.exists(tf_path):
        print("  Сохраненная модель не найдена.")
        return None, None

    try:
        # Загружаем модель (на CPU для совместимости)
        model = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Загружаем фабрику
        with open(tf_path, 'rb') as f:
            tf = pickle.load(f)
            
        print("  Модель и данные успешно загружены!")
        return model, tf
    except Exception as e:
        print(f"  Ошибка при загрузке: {e}")
        return None, None


def get_embedding_for_entity(model, triples_factory, entity_uri):
    """Получить вектор для конкретной сущности"""
    # PyKEEN использует внутренние ID, нужно найти ID по URI
    if entity_uri in triples_factory.entity_to_id:
        entity_id = triples_factory.entity_to_id[entity_uri]
        # Получаем эмбеддинг (нужен torch.tensor)
        entity_id_tensor = torch.as_tensor([entity_id])
        embedding = model.entity_representations[0](entity_id_tensor)
        return embedding.detach().numpy()[0]
    else:
        print(f"Сущность {entity_uri} не найдена в словаре.")
        return None

def find_similar_entities(model, triples_factory, entity_uri, top_k=5):
    """Найти топ-K похожих сущностей"""
    if entity_uri not in triples_factory.entity_to_id:
        print(f"Сущность {entity_uri} не найдена.")
        return

    target_id = triples_factory.entity_to_id[entity_uri]
    
    # Получаем все эмбеддинги как тензор
    all_embeddings = model.entity_representations[0](indices=None).detach()
    target_embedding = all_embeddings[target_id].unsqueeze(0)

    # Вычисляем расстояния (чем меньше, тем более похожи)
    # TransE оптимизирует расстояния, поэтому используем Евклидову метрику
    distances = torch.norm(all_embeddings - target_embedding, dim=1, p=2)
    
    # Получаем индексы с наименьшим расстоянием
    # topk с largest=False даст нам наименьшие значения
    values, indices = torch.topk(distances, k=top_k+1, largest=False)
    
    print(f"\nСущности, похожие на {entity_uri}:")
    for value, idx in zip(values, indices):
        idx = idx.item()
        if idx == target_id:
            continue
            
        entity_label = triples_factory.entity_id_to_label[idx]
        print(f"  - {entity_label} (дистанция: {value.item():.4f})")

if __name__ == "__main__":
    # Путь к онтологии
    ttl_file = "wow_items_graph.ttl"
    save_dir = 'wow_embedding_results'
    
    # Пробуем загрузить существующую модель
    model, tf = load_saved_model(save_dir)
    
    # Если не нашли или не смогли загрузить - тренируем заново
    if model is None:
        print("Запускаем обучение с нуля...")
        model, tf = train_graph_embeddings(ttl_file)

    # Пример 1: Получить вектор
    example_entity = "http://example.org/wowkg#Item" 
    vector = get_embedding_for_entity(model, tf, example_entity)
    
    if vector is not None:
        print(f"\nПример вектора для {example_entity}:")
        print(vector[:5], "... (показаны первые 5 чисел)")

    # Пример 2: Найти похожие сущности
    # Попробуем найти что-то похожее на "Меч" (Sword) или "Item"
    # Замените URI на те, которые точно есть в вашем графе
    print("\n--- Проверка похожих сущностей ---")
    find_similar_entities(model, tf, "http://example.org/wowkg#Sword")
    find_similar_entities(model, tf, "http://example.org/wowkg#Warrior")

# AGENTS.md

Этот файл для новых пользователей/агентов проекта `GraphEcho`.
Цель: быстро понять, что уже сделано в репозитории и в каком порядке запускать скрипты.

## Что уже стандартизировано

- Основной рабочий датасет: `export_points_lv_800x600/`
- Аннотации для LV:
  - класс: `0`
  - координаты: нормализованные (`x/width`, `y/height`)
- Основная точка валидации: `tools/runs/validation_runs/validation_summary.csv`
- Документация по скриптам вынесена в: `tools/docs/`

## Где смотреть документацию

- Общая последовательность: `tools/docs/README.md`
- Экспорт аннотаций: `tools/docs/EXPORT_POINTS_ANNOTATIONS.md`
- Фильтр LV: `tools/docs/FILTER_LEFT_VENTRICLE_POINTS.md`
- Resize + пересчет аннотаций: `tools/docs/RESIZE_EXPORTED_POINTS.md`
- Обучение YOLO: `tools/docs/YOLO_LEARNING.md`
- Single-image inference: `tools/docs/YOLO_SINGLE_TEST.md`
- Валидация всех run-ов: `tools/docs/VALIDATE_YOLO_RUNS.md`
- График из training logs: `tools/docs/PLOT_YOLO_METRICS.md`
- График из post-validation: `tools/docs/PLOT_VALIDATION_METRICS.md`
- Видео -> GIF: `tools/docs/INFER_VIDEOS_TO_GIF.md`

## Практический порядок (если датасет уже подготовлен)

1. Обучение:
   - `C:\conda\python.exe tools\yolo_learning.py`
2. Быстрый тест на изображении:
   - `C:\conda\python.exe tools\yolo_single_test.py --run yolo11x-seg_100_x`
3. Полная валидация моделей:
   - `C:\conda\python.exe tools\validate_yolo_runs.py --runs yolo11n-seg_100_n yolo11s-seg_100_s yolo11m-seg_100_m yolo11l-seg_100_l yolo11x-seg_100_x --batch 8 --workers 0 --device 0`
4. Построение графика по новой валидации:
   - `C:\conda\python.exe tools\plot_validation_metrics.py --runs yolo11n-seg_100_n yolo11s-seg_100_s yolo11m-seg_100_m yolo11l-seg_100_l yolo11x-seg_100_x`
5. Инференс видео в GIF:
   - `C:\conda\python.exe tools\infer_videos_to_gif.py --input-dir videos --output-dir videos/gif_results --run auto`

## Ключевые артефакты

- Обучения: `tools/runs/<run_name>/`
- Валидация: `tools/runs/validation_runs/<run_name>_val/`
- Сводка валидации: `tools/runs/validation_runs/validation_summary.csv`
- Графики валидации:
  - `tools/runs/validation_runs/validation_metrics_grouped_bar.png`
  - `tools/runs/validation_runs/validation_metrics_all_images_grouped_bar.png`
- GIF результаты:
  - `videos/gif_results/*.gif`

## Частые ошибки

- Опечатка в именах run-папок:
  - правильно: `yolo11l-seg_100_l`, `yolo11x-seg_100_x`
  - неправильно: `yolo100l-...`, `yolo100x-...`
- Путаница метрик:
  - `plot_yolo_metrics.py` -> метрики из training `results.csv`
  - `plot_validation_metrics.py` -> метрики из `model.val()` (`validation_summary.csv`)
- Windows + stdin при `model.val()`:
  - не запускать длинные `val`-пайплайны через `python -`
  - использовать `tools/validate_yolo_runs.py` (там это уже учтено)

## Текущее правило для графика валидации

В `tools/plot_validation_metrics.py` ось Y зафиксирована:
- минимум `0.6`
- максимум `1.2`

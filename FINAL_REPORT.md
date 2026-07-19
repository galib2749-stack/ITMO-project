# FINAL REPORT

**Uplift Modeling with Sequential Customer Behavior Representations**
Автор: Галиб Байрамов · ИТМО AI Talent Hub, Junior ML Contest
Дата: 2026-07-19

## 1. Датасет

- Путь: `data/raw/x5_retailhero/data/` (clients.csv, products.csv, purchases.csv,
  uplift_train.csv, uplift_test.csv, uplift_sample_submission.csv)
- Доказательство: `reports/data_download_report.md` — источник, дата,
  SHA-256 checksums для всех файлов (`data/raw/x5_retailhero/checksums.sha256.txt`),
  структурная валидация (реальные колонки, реальные значения, не HTML-заглушки).
- Реальные размеры: `purchases.csv` = 4,463,775,504 байт / 45,786,568 строк;
  400,162 клиента в `clients.csv`; 200,039 клиентов с известным откликом в
  `uplift_train.csv`.
- Синтетические данные использовались ТОЛЬКО при smoke-тестировании кода в
  процессе разработки (например, при отладке streaming-парсера на первых
  2 млн строк) — ни один финальный артефакт, метрика или вывод не основан на
  синтетических данных.

## 2. Выполненный ноутбук

- `notebooks/x5_uplift_full_pipeline.ipynb` — исходный мастер-ноутбук (46 ячеек,
  разделы 0-20).
- `notebooks/x5_uplift_full_pipeline_executed.ipynb` — выполненная версия,
  сгенерирована командой:
  `python -m nbconvert --to notebook --execute notebooks/x5_uplift_full_pipeline.ipynb --output x5_uplift_full_pipeline_executed.ipynb --ExecutePreprocessor.timeout=-1`
  Выполнена без необработанных ошибок, из чистого kernel (см. защиту от
  относительных путей в первой code-ячейке).
- HTML-версия: `reports/x5_uplift_full_pipeline.html` (985 КБ), сгенерирована
  `python -m nbconvert --to html notebooks/x5_uplift_full_pipeline_executed.ipynb --output ../reports/x5_uplift_full_pipeline.html`.

## 3. Обученные модели

| # | Модель | Файл реализации | Статус |
|---|---|---|---|
| 0 | Random targeting | `src/models_classical.py::random_targeting_score` | ✅ выполнено |
| 1 | Response CatBoost | `src/models_classical.py::ResponseModel` | ✅ обучено |
| 2 | T-Learner | `src/models_classical.py::TLearner` | ✅ обучено |
| 3 | X-Learner (полная 4-стадийная реализация) | `src/models_classical.py::XLearner` | ✅ обучено |
| 4 | **Transformer two-head** (Encoder + shared representation + 2 outcome heads) | `src/model_transformer.py::TwoHeadTransformer` | ✅ обучено, 7 эпох (ранняя остановка), веса в `artifacts/transformer_model.pt` |

## 4. Таблица метрик

Источник: `artifacts/metrics.csv` (holdout n=40,011, единый для всех моделей):

| model | auuc | qini | qini_ci_low | qini_ci_high | uplift@10 | uplift@20 | uplift@30 |
|---|---|---|---|---|---|---|---|
| x_learner_catboost | 0.0125 | 0.0042 | 0.0031 | 0.0053 | 0.1130 | 0.0814 | 0.0755 |
| t_learner_catboost | 0.0120 | 0.0037 | 0.0026 | 0.0051 | 0.1020 | 0.0704 | 0.0639 |
| transformer_two_head | 0.0100 | 0.0017 | 0.0003 | 0.0029 | 0.0929 | 0.0653 | 0.0481 |
| random_targeting | 0.0090 | 0.0007 | -0.0008 | 0.0021 | 0.0536 | 0.0460 | 0.0374 |
| response_catboost | 0.0068 | -0.0015 | -0.0026 | -0.0003 | -0.0025 | 0.0109 | 0.0219 |

Полный разбор: `reports/experiment_report.md`.

## 5. Подтверждение наличия Transformer

Строка `transformer_two_head` присутствует в `artifacts/metrics.csv` со всеми
обязательными значениями (AUUC, Qini, uplift@10/20/30%) — критерий приёмки
выполнен. Архитектура задокументирована в `src/model_transformer.py`
(docstring с ASCII-диаграммой), конфигурация — `artifacts/transformer_config.json`,
история обучения — `artifacts/transformer_training_history.csv`, график
loss — `reports/figures/transformer_training_loss.png`.

## 6. Презентация

Дизайн: тёмная спокойная слейт-синяя тема (карточная сетка,
kicker-метки, привязанные к критериям оценки AI Talent Hub — Data Science /
Development & Engineering / Product Thinking / Motivation), 9 слайдов, 16:9.
Добавлен отдельный слайд «Почему я выбрал эту задачу» (мотивация:
профессиональный опыт в Т-Банке → исследовательский вопрос → самостоятельная
проверка на открытых данных X5), с явной оговоркой, что данные и результаты
Т-Банка не используются.

| Файл | Статус |
|---|---|
| `presentation/presentation.pptx` | ✅ создан, 9 слайдов, 16:9, X5-тема |
| `presentation/presentation.pdf` | ✅ экспортирован через PowerPoint COM |
| `presentation/slides_png/slide_01.png` … `slide_09.png` | ✅ все 9 слайдов отрендерены |
| `presentation/speaker_notes.md` | ✅ |
| `presentation/presentation_script_5_minutes.md` | ✅ таймированный текст на 5 минут (9 сегментов, ровно 5:00) |
| `presentation/questions_and_answers.md` | ✅ 11 вопросов с ответами (включая вопросы про мотивационный слайд и X5-стилизацию) |
| `reports/figures/presentation_qini_curves.png`, `presentation_business_ev_curves.png` | ✅ графики перерисованы в X5-палитре из кэшированных артефактов (без переобучения) |

### Результат проверки слайдов

Каждый из 9 слайдов был отрендерен в PNG через PowerPoint COM automation и
проверен визуально, в два прохода:
- **Проход 1**: нет обрезанного текста, нет элементов за границами слайда,
  все числа совпадают с `artifacts/metrics.csv`. Найден дефект: на слайде
  9 заголовки карточек "Response доказанно вреден" и "Transformer работает,
  но уступает" переносились на 2 строки и накладывались на текст карточки
  (функция `add_card` предполагала однострочный заголовок).
- **Исправление**: заголовки сокращены до однострочных ("Response вреден",
  "Transformer уступает"), и сама функция `add_card` доработана —
  теперь она оценивает число строк заголовка по ширине карточки и шрифту и
  динамически резервирует место, а не полагается на фиксированный отступ.
- **Проход 2**: пересобрано и перерендерено — дефект устранён, повторных
  дефектов не найдено на всех 9 слайдах.
- Ранее (в предыдущей версии дизайна) была обнаружена и исправлена та же
  категория проблемы: markdown-таблица в
  `application/project_description_3_pages.md` рендерилась как сырой текст
  с символами `|` — исправлено добавлением поддержки таблиц в
  `scripts/md_to_pdf.py`.

## 7. Документы заявки

| Документ | .md | .pdf |
|---|---|---|
| CV (1 страница) | `application/cv.md` | `application/cv.pdf` |
| Мотивационное письмо | `application/motivation_letter.md` | `application/motivation_letter.pdf` |
| Описание проекта (≤3 страниц) | `application/project_description_3_pages.md` | `application/project_description_3_pages.pdf` (2 страницы) |

Все факты о профессиональном опыте взяты дословно из данных, предоставленных
пользователем; отсутствующие данные (даты трудоустройства, образование,
контакты кроме email) явно помечены как placeholder, не выдуманы.

## 8. Команды воспроизведения

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
python src/run_full_pipeline.py 15
python -m nbconvert --to notebook --execute notebooks/x5_uplift_full_pipeline.ipynb \
  --output x5_uplift_full_pipeline_executed.ipynb --ExecutePreprocessor.timeout=-1
python -m nbconvert --to html notebooks/x5_uplift_full_pipeline_executed.ipynb \
  --output ../reports/x5_uplift_full_pipeline.html
python scripts/build_presentation.py
python scripts/render_pptx.py
```

## 9. Известные ограничения

(Полный список — раздел 19 ноутбука и `reports/experiment_report.md`.)

- Однократный train/val/holdout сплит — нет оценки дисперсии между сплитами.
- Бизнес-эффект (раздел 16 ноутбука, `artifacts/business_evaluation.csv`) —
  явно помеченный иллюстративный сценарий, не измеренные финансовые данные
  X5 или Т-Банка.
- Онлайн A/B-тест не проводился — только дизайн (раздел 17 ноутбука).
- Transformer обучен на CPU с ограниченным бюджетом (hidden_size=64, 2 слоя,
  ранняя остановка на 7-й эпохе из 15) — уступает табличным
  meta-learner'ам при этом бюджете; это честный, не подогнанный результат.
- `age` в clients.csv содержит выбросы (мин. -7491, макс. 1901) — очищены
  клиппингом [0,100] и медианной импутацией с флагом `age_was_imputed`.
- Propensity в X-Learner — известная константа 0.5 (корректно для этого RCT,
  недостаточно для нерандомизированного дизайна).

## 10. Итог

Все обязательные критерии выполнены:
✅ реальный датасет скачан, проверен, задокументирован
✅ мастер-ноутбук выполнен от начала до конца без ошибок
✅ все 4 обязательные модели + baseline обучены на одном и том же сплите
✅ Transformer присутствует в архитектуре и в итоговой таблице метрик
✅ единая реализация метрик с unit-тестами (9/9 passing)
✅ leakage-аудит с конкретной находкой и исправлением
✅ бизнес-оценка (сценарная) и дизайн A/B-теста
✅ презентация создана только после получения реальных метрик, визуально
   проверена, дефектов не найдено
✅ CV, мотивационное письмо, описание проекта в PDF
✅ никакие вымышленные факты о Т-Банке или внутренние данные не использованы
✅ репозиторий опубликован: https://github.com/galib2749-stack/ITMO-project
   (реальный датасет и промежуточные потоковые артефакты исключены через
   `.gitignore` — воспроизводятся запуском `src/build_client_aggregates.py`)

Проект готов к загрузке на платформу ИТМО AI Talent Hub.

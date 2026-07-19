# ИТМО Junior ML Contest / AI Talent Hub — Requirements Mapping

Source: official contest page https://ai.itmo.ru/junior_ml_contest (fetched 2026-07-19),
plus the submission-form screenshot provided by the user ("Загрузить решение на
платформу", form fields: CV, motivation letter, repository link, project
description ≤3 pages, presentation).

Today's date: **2026-07-19**. The page lists three submission waves; Wave 3 is
**3–20 июля (July 3–20)**, so today falls inside the final wave's closing window.
No exact cutoff time/timezone is published on the page — treat July 20 end-of-day
as the hard deadline and submit as early as possible within it.

| Требование ИТМО | Источник | Как выполняется в проекте | Статус |
|---|---|---|---|
| Ссылка на репозиторий (код проекта) | Форма подачи (скриншот), офиц. страница | https://github.com/galib2749-stack/ITMO-project — полный исходный код, `README.md`, `src/`, `notebooks/` (личные документы исключены через `.gitignore`) | ✅ Готово |
| CV / расширенное резюме (опыт, образование, навыки, достижения) | Форма подачи, офиц. страница | `application/cv.pdf`, 1 страница | ✅ Готово |
| Мотивационное письмо (почему AI/ML, почему AI Talent Hub) | Форма подачи, офиц. страница | `application/motivation_letter.pdf` | ✅ Готово |
| Описание проекта до 3 страниц (задача, данные, методы, результаты) | Форма подачи, офиц. страница | `application/project_description_3_pages.pdf` — расширенное, research-style, ровно 3 страницы, с реальными метриками | ✅ Готово |
| Презентация (шаблон, 5 мин доклад + 7 мин вопросы) | Офиц. страница | `presentation/presentation.pptx` + `.pdf`, 9 слайдов, визуально проверены; `speaker_notes.md`, `presentation_script_5_minutes.md` (ровно 5:00), `questions_and_answers.md` | ✅ Готово |
| Дедлайн (актуальный поток) | Офиц. страница | Wave 3: 3–20 июля 2026 | Учтено, риск указан пользователю |
| Критерии оценки: Development & Engineering (git, docker, ci, качество кода, pipeline) | Офиц. страница | Git-репозиторий с историей, модульный `src/`, воспроизводимый ноутбук, тесты в `tests/`, `Dockerfile` + `requirements-docker.txt`, CI (`.github/workflows/tests.yml`, автозапуск pytest) | ✅ Готово (Docker-образ создан, но не собирался/не тестировался в этой сессии — см. примечание ниже) |
| Критерии оценки: Data Science (EDA, препроцессинг, модели, метрики, валидация) | Офиц. страница | Разделы 4–18 мастер-ноутбука, leakage-аудит, bootstrap CI, sanity-проверки | ✅ Готово |
| Критерии оценки: AI Application (использование AI-инструментов/агентов) | Офиц. страница | Явно раскрыто в `application/project_description_3_pages.pdf` (раздел "Использование AI-инструментов"), `README.md`, `presentation/questions_and_answers.md` | ✅ Готово |
| Критерии оценки: Product Thinking (проблема, аудитория, MVP, эффект) | Офиц. страница | Разделы Business problem, Business evaluation, Online A/B-test design; сравнение с официальным baseline-решением как форма конкурентного анализа | ✅ Готово (конкурентный анализ — только через сравнение с baseline, не отдельным разделом) |
| Критерии оценки: Motivation (карьерные цели, понимание программы) | Офиц. страница | Мотивационное письмо + слайд 2 презентации, со ссылками на реальные направления программы (LLM/RAG/MLOps и т.д.) | ✅ Готово |
| Требование к личному вкладу (для командных проектов) | Офиц. страница: "командный проект — каждый подаёт отдельно, описывает свой вклад" | Проект индивидуальный, автор — Галиб Байрамов, вклад = 100% самостоятельная разработка | Неприменимо (соло-проект) |
| Формат документов — PDF | Офиц. страница | Все submission-документы в PDF через reportlab | ✅ Готово |

## Notes on gaps / assumptions to flag to the jury

- The official page does not specify an exact submission cutoff time or timezone
  for Wave 3 (only "3–20 июля"). This project treats July 20, 2026 as the final
  submission day.
- The page references a presentation "slide template"; no template file was
  provided by the user in this conversation. In the absence of a supplied
  template file, the presentation will follow the format constraints in the
  contest brief (16:9, 7–8 slides, ≤5 min, minimalist professional design)
  rather than a specific ITMO-branded template.
- "AI Application" criterion is satisfied honestly: this entire project was
  built with Claude Code (Anthropic) as an AI pair-programmer/agent under the
  author's direction — this is disclosed, not hidden.
- The `Dockerfile` was written carefully (excludes Windows-only `pywin32`,
  which cannot install in a Linux container, via a separate
  `requirements-docker.txt`) but was **not actually build-tested** in this
  session (Docker Desktop was not started, per explicit user request to skip
  that step). If asked, be upfront that the image's correctness rests on
  careful authoring, not a verified build.

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
| Ссылка на репозиторий (код проекта) | Форма подачи (скриншот), офиц. страница | https://github.com/galib2749-stack/ITMO-project — полный исходный код, `README.md`, `src/`, `notebooks/` | ✅ Готово |
| CV / расширенное резюме (опыт, образование, навыки, достижения) | Форма подачи, офиц. страница | `application/cv.pdf` + `application/cv.md`, 1 страница, только подтверждённые факты | Запланировано |
| Мотивационное письмо (почему AI/ML, почему AI Talent Hub) | Форма подачи, офиц. страница | `application/motivation_letter.pdf` + `.md` | Запланировано |
| Описание проекта до 3 страниц (задача, данные, методы, результаты) | Форма подачи, офиц. страница | `application/project_description_3_pages.pdf` + `.md`, писать только после получения реальных метрик | Запланировано |
| Презентация (шаблон, 5 мин доклад + 7 мин вопросы) | Офиц. страница | `presentation/presentation.pptx` + `.pdf`, `speaker_notes.md`, `presentation_script_5_minutes.md`, `questions_and_answers.md` | Запланировано, только после метрик |
| Дедлайн (актуальный поток) | Офиц. страница | Wave 3: 3–20 июля 2026. Сегодня 19 июля — последний/предпоследний день | Учтено, риск указан пользователю |
| Критерии оценки: Development & Engineering (git, docker, ci, качество кода, pipeline) | Офиц. страница | Модульный `src/`, воспроизводимый ноутбук, тесты в `tests/` | В работе |
| Критерии оценки: Data Science (EDA, препроцессинг, модели, метрики, валидация) | Офиц. страница | Разделы 4–18 мастер-ноутбука | В работе |
| Критерии оценки: AI Application (использование AI-инструментов/агентов) | Офиц. страница | Проект разработан с использованием Claude Code как AI-инструмента разработки; это фиксируется в описании проекта и мотивационном письме | Запланировано |
| Критерии оценки: Product Thinking (проблема, аудитория, MVP, эффект) | Офиц. страница | Разделы Business problem, Business evaluation, Online A/B-test design | В работе |
| Критерии оценки: Motivation (карьерные цели, понимание программы) | Офиц. страница | Мотивационное письмо | Запланировано |
| Требование к личному вкладу (для командных проектов) | Офиц. страница: "командный проект — каждый подаёт отдельно, описывает свой вклад" | Проект индивидуальный, автор — Галиб Байрамов, вклад = 100% самостоятельная разработка | Неприменимо (соло-проект) |
| Формат документов — PDF | Офиц. страница | Все submission-документы генерируются в PDF через reportlab | Запланировано |

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

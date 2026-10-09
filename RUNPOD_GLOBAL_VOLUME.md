# Studio: переход на Runpod Global Volume

Global Volume уже создан. Дальше: обновить образ → загрузить модели через временный GPU Pod → подключить том к Serverless → проверить запуск → подключить Studio.

Подготовлен режим `MODEL_STORAGE=global`: воркер только проверяет готовность моделей и читает их, без блокировок, переименований и записи на Global Volume. Обычные Network Volumes сохраняют прежний режим `MODEL_STORAGE=network`.

## 1. Обновить GitHub и собрать образ

1. Распакуй свежий `Runpod/OFM_Runpod_Serverless_GitHub.zip` в отдельную папку.
2. Открой существующий репозиторий `sypex6/ofm-studio-runpod` → Code → Add file → Upload files.
3. Загрузи папки `studio` и `tests` из архива, сохраняя структуру, а также обе инструкции. GitHub заменит совпадающие файлы. Commit changes в `main`.
4. Отдельно обнови `.github/workflows/runpod-image.yml` содержимым файла из архива, если скрытая папка не попала в загрузку. Новый workflow запускает также тесты Global Volume.
5. Actions → Build Runpod Serverless image → дождись последнего запуска с зелёной галочкой. Если не запустилось, Run workflow → main.
6. Открой завершённый запуск → Summary → скопируй точный адрес из блока Runpod image: `ghcr.io/sypex6/ofm-studio-runpod:ПОЛНЫЙ_COMMIT_SHA`.
7. Используй этот новый тег. Старый образ не имеет режима Global Volume. Публичность GHCR package сохраняется; если package приватный, переведи его в Public.

`hf_token.txt`, `.env`, пользовательские данные и cookies на GitHub не загружай. Публичный архив их не содержит.

## 2. Создать временный Pod для подготовки моделей

Этот Pod нужен только для переноса файлов, генерацию на нём не запускаем. Подойдёт доступная недорогая GPU; RTX PRO 6000 для копирования не требуется. Работа Pod оплачивается.

1. Storage → созданный Global Volume → Configure Pod with volume.
2. Выбери GPU и образ **Runpod PyTorch** с Python и доступом к терминалу. Наш Serverless-образ сюда не ставь: он сам запускает воркер.
3. Если старый Network Volume содержит готовые модели, добавь его в Storage → Persistent storage → Add volume. Pod должен запускаться в датацентре старого Network Volume.
4. При стандартных путях с двумя томами: Network Volume находится в `/workspace`, Global Volume — в `/workspace-global`. С одним Global Volume: он находится в `/workspace`.
5. Если модели надо скачать заново, выстави Container Disk **100 GB**, чтобы временные 56,05 GiB моделей поместились. С готовым Network Volume такой запас не нужен.
6. Запусти Pod → Connect → терминал / Jupyter Terminal. Проверь фактические mount paths в Console и Storage перед командами ниже.

Официальные инструкции: [Global Volumes для Pods](https://docs.runpod.io/storage/globalvolume/globalvolume-pods).

## 3. Получить скрипты в терминале Pod

Все команды ниже выполняются **в Linux-терминале Pod**, а не в PowerShell на компьютере.

```bash
git clone https://github.com/sypex6/ofm-studio-runpod.git /tmp/ofm-runpod
cd /tmp/ofm-runpod/studio/cuda128/serverless
python -m pip install requests filelock
```

Если репозиторий приватный, скачай архив исходников с GitHub на компьютер и загрузи его в Pod через Jupyter, затем распакуй и перейди в `studio/cuda128/serverless`. Публичность GHCR package не делает репозиторий публичным.

## 4А. Перенести готовые модели со старого Network Volume

Используй этот вариант, если на старом томе есть `/workspace/models` с нашими моделями.

```bash
ls /workspace/models
ls /workspace-global
python publish_global.py --source /workspace/models --destination /workspace-global/models
```

Скрипт проверяет SHA-256 источников, копирует обязательные модели и проверяет загруженные файлы. Маркер готовности записывается после проверки каждого файла. Он не использует переименование, блокировки или hard links. Источник не изменяется.

Если исходные модели неполные, сначала докачай на обычный Network Volume:

```bash
read -r -s -p 'HF token: ' HF_TOKEN; echo
export HF_TOKEN
python prepare_models.py --root /workspace/models
unset HF_TOKEN
python publish_global.py --source /workspace/models --destination /workspace-global/models
```

Вставь токен из локального `Runpod/hf_token.txt` в скрытый ввод. В команду или GitHub его не вставляй.

## 4Б. Если старого тома с моделями нет: скачать заново

С одним подключённым Global Volume стандартный путь — `/workspace`. Скачиваем сначала на обычный временный диск `/tmp`, затем копируем на Global Volume.

```bash
df -h /tmp
read -r -s -p 'HF token: ' HF_TOKEN; echo
export HF_TOKEN
python prepare_models.py --root /tmp/ofm-models
unset HF_TOKEN
python publish_global.py --source /tmp/ofm-models --destination /workspace/models
```

Если Global Volume у тебя смонтирован в другом месте, замени `/workspace/models` на его фактический путь плюс `/models`. Не указывай Global Volume как `--root` обычного загрузчика: ему нужен диск с полной поддержкой файловых операций.

## 5. Проверить перенос

Вариант с двумя томами:

```bash
python prepare_models.py --root /workspace-global/models --check --readonly
```

Вариант с одним Global Volume:

```bash
python prepare_models.py --root /workspace/models --check --readonly
```

Ожидаемые сообщения:

```text
Global Volume ready: 16 models. Start NEW workers now.
Models ready: 16
```

Проверка в Pod не заменяет проверку нового Serverless-воркера: запущенные воркеры читают состояние Global Volume на момент монтирования. Создавай/перезапускай их после завершения загрузки.

## 6. Подключить Global Volume к Serverless

1. Serverless → свой endpoint → Manage → Edit Endpoint.
2. Container image: вставь **новый** адрес образа с полным commit SHA из шага 1.
3. Advanced → отсоедини старые Network Volumes у этого endpoint, затем выбери созданный Global Volume. Одновременно оба типа к endpoint не подключаются.
4. GPU: **96 GB RTX PRO 6000**. GPU count: **1**. Сними ограничения датацентров, если они были выставлены вручную.
5. Active workers: **0**, Max workers: **1**. Container Disk: **40 GB**. Модели лежат отдельно на Global Volume.
6. Execution timeout: **3600 секунд**. Для прогрева выстави worker initialization timeout 3600, если настройка есть в интерфейсе.
7. Environment variables:

| Имя | Значение |
|---|---|
| `MODEL_STORAGE` | `global` |
| `DOWNLOAD_MODELS` | `0` |
| `WORKFLOW_PROFILES` | `animate-ki,animate-wrapper` |
| `STUDIO_HOST` | `ofm-109-71-252-204.sslip.io` |
| `GENERATION_TIMEOUT` | `3600` |
| `RUNPOD_INIT_TIMEOUT` | `3600` |

HF_TOKEN для рабочего Serverless-воркера не нужен: он не скачивает модели. Удали переменную, если она осталась от старого способа подготовки.

8. Save Endpoint. Смена тома вызывает новый release. Для проверки старта можно временно выставить Active workers **1**, после проверки вернуть **0**. Активный воркер оплачивается и во время ожидания задач.

Global Volume открывает доступ к GPU разных датацентров, но не гарантирует свободную RTX PRO 6000. В Serverless выбирается пул GPU, поэтому фактическую карту и 96 GB VRAM проверь в логах.

[Подключение Global Volume к Serverless](https://docs.runpod.io/storage/globalvolume/globalvolume-serverless).

## 7. Проверить логи и завершить временный Pod

У нового воркера должны появиться:

```text
Model Volume: /runpod-volume/models, readonly=True; ...
Models ready: 16
Graph schema validated: animate-ki 28 nodes
Graph schema validated: animate-wrapper 25 nodes
Starting Serverless Worker
All fitness checks passed
```

Если появляется `Missing/incomplete models`, проверь, что модели находятся непосредственно в `<Global Volume>/models`, выбран правильный том и перенос завершён до старта воркера. После исправления создай новые воркеры.

Если сообщение про FileLock/atomic rename — используется старый образ или не задан `MODEL_STORAGE=global`.

Если воркер остаётся Pending без логов — модели ещё не проверялись: Runpod не выделил GPU или не начал запуск.

После успешного запуска заверши временный Pod через **Terminate**, сохранив сам Global Volume. Не удаляй старый Network Volume до первой успешной генерации; после неё его можно отдельно удалить, если он больше не нужен. Оба хранилища тарифицируются, пока существуют.

## 8. Подключить Studio и проверить настоящий ролик

Для Studio нужны **Runpod API key** и **endpoint ID** (ID берётся на странице endpoint). Это другие данные, не Hugging Face token. Их нужно сохранить в `/opt/ofm-studio/.env` на VPS в переменных `RUNPOD_API_KEY` и `RUNPOD_ENDPOINT_ID`, затем перезапустить `ofm-studio`. Endpoint ID указывается без URL. Если Studio уже подключена к этому же endpoint, менять ключ и ID не нужно.

После подключения: короткий видеореференс и фото → Kiara Animate → проверить появление MP4 в личной библиотеке. Затем отдельно проверить SSIBAL Animate. Проверки графов и fitness checks подтверждают запуск, но не успешную генерацию и достаточность VRAM.

Изменения исходников и локальные проверки подготовлены. Перенос на реальный Global Volume, новый Docker build, старт в Runpod и GPU-генерация выполняются по этому гайду и пока не подтверждены.

# Studio → Runpod Serverless

Обновлено по `Runpod/Workflows`: **Kiara Animate** и **SSIBAL Animate**, включая присланные API-версии. Подготовлены исходники образа. Сборку нужно выполнить в GitHub Actions; новый образ и GPU-генерация ещё должны пройти проверку на Runpod.

Цепочка Studio: фото модели + первые кадры + видеореференсы → опциональный Gemini 3 Flash через Siray по теме пользователя → Nano Banana 2 → готовое фото → Runpod / ComfyUI → MP4 в личной библиотеке. Автопромптер и подготовку фото можно отключать.

## 1. Обновить существующий GitHub repository

1. Распакуй свежий `OFM_Runpod_Serverless_GitHub.zip`.
2. Загрузи содержимое в существующий репозиторий, сохраняя пути `studio/cuda128/`, `tests/`, `.github/workflows/`. Замени Dockerfile, start.sh и файлы serverless на новые. Старые workflow JSON можно удалить: теперь worker использует конкретные новые API-файлы.
3. Папка `.github` может быть скрыта. Обнови `.github/workflows/runpod-image.yml` через редактирование существующего файла, если загрузка папок её пропустила.
4. Открой **Actions → Build Runpod Serverless image**. Если commit уже запустил сборку, дождись её; иначе нажми **Run workflow**.
5. После зелёной сборки скопируй из Summary адрес `ghcr.io/ТВОЙ_ЛОГИН/ofm-studio-runpod:COMMIT_SHA`.
6. Сохрани GHCR package публичным: **Packages → ofm-studio-runpod → Package settings → Change visibility → Public**.

Workflow проверяет графы, HTTP-докачку и реестр установленного ComfyUI при сборке. Использует встроенный `GITHUB_TOKEN` с `packages: write`; Docker Hub не требуется. [GitHub Actions / GHCR](https://docs.github.com/en/actions/tutorials/publish-packages/publish-docker-images).

**`hf_token.txt` не загружать на GitHub.** Архив содержит только worker, тесты и инструкцию. В нём нет токена Hugging Face, `.env`, cookies, базы Studio или пользовательских файлов.

## 2. Network Volume и модели

Подключи Network Volume **150 GB**. Обязательные модели занимают **56,05 GiB**; дополнительные отключённые модели сегментации — ещё **0,47 GiB**. Container Disk и Network Volume — разные диски. [Runpod Network Volumes](https://docs.runpod.io/storage/network-volumes).

Модели скачиваются напрямую в `/runpod-volume/models/<категория>/<имя>.part`. После проверки размера и SHA-256 файл атомарно получает окончательное имя. HTTP Range позволяет продолжить загрузку после обрыва. Hub/Xet-кеш для скачивания моделей не используется. Worker печатает размер/свободное место Volume и останавливается с понятной ошибкой, если для следующего файла места недостаточно.

Исходные ссылки закреплены на commit SHA в `serverless/models.json`. Совпадающие CLIP/VAE/text encoder/Lightx2v/YOLO используются обоими профилями без повторного скачивания.

**Основные модели имеют одинаковые имена, но разные SHA-256:**

- SSIBAL: `diffusion_models/Wan2_2-Animate-14B_fp8_scaled_e4m3fn_KJ_v2.safetensors`, источник Kijai.
- Kiara: `diffusion_models/kiara/Wan2_2-Animate-14B_fp8_scaled_e4m3fn_KJ_v2.safetensors`, источник GerbyHorty76.

Подпапка Kiara предотвращает перезапись. LoRA Seko получает имя, ожидаемое workflow. Для ViTPose H добавлен обязательный `vitpose_h_wholebody_data.bin` из того же репозитория: ONNX ссылается на него.

SAM и дополнительные YOLO включены в manifest с `optional: true`: они не участвуют в присланном API-графе. Скачать их отдельно можно командой в контейнере:

```bash
cd /opt/studio-worker/serverless
python prepare_models.py --root /runpod-volume/models --include-optional
```

Это скачивает файлы; ручная отключённая сегментация автоматически не включается.

Полные файлы с совпадающим SHA-256 из старого `.hf-cache` могут быть проверены и переиспользованы через hard link. Незавершённые загрузки Xet не считаются готовыми моделями. Старые файлы и кеши автоматически не удаляются. Проверка диска:

```bash
df -h /runpod-volume /tmp
du -sh /runpod-volume/models /runpod-volume/huggingface 2>/dev/null
```

При `Disk quota exceeded` проверь подключение, размер и занятое место Volume. Новые ссылки не увеличивают квоту. Не удаляй кеш до проверки, какие файлы удалось переиспользовать.

## 3. Обновить Runpod endpoint

Выбери **Queue-based**, новый Docker image из Summary и одну GPU на worker. Volume монтируется в `/runpod-volume` и ограничивает доступные датацентры.

| Настройка | Значение |
| --- | --- |
| Active / Min workers | 0 |
| Max workers | 1 |
| Execution timeout | 3600 секунд |
| Container Disk | 40 GB; увеличить, если образ не помещается |
| GPU для первого теста | A100 80 GB; потребление памяти ещё проверить генерацией |

Для первоначальной загрузки можно временно выбрать дешёвую GPU в датацентре Volume. Загрузчик и проверка схем не запускают генерацию.

Переменные окружения:

| Имя | Значение |
| --- | --- |
| `STUDIO_HOST` | `ofm-109-71-252-204.sslip.io` без https:// и слеша |
| `WORKFLOW_PROFILES` | `animate-ki,animate-wrapper` |
| `DOWNLOAD_MODELS` | `1` для первого запуска нового manifest, потом `0` |
| `GENERATION_TIMEOUT` | `3600` |
| `RUNPOD_INIT_TIMEOUT` | `3600` на первоначальную загрузку |
| `HF_TOKEN` | Значение из локального `Runpod/hf_token.txt`; только в настройках Runpod |

Токен применяется к запросу Hugging Face; при перенаправлении на другой хост Authorization снимается. Он не передаётся в задания Studio/ComfyUI и не печатается в логах. Авторизация повышает лимиты, но не гарантирует конкретную скорость сети.

1. Сохрани новый образ, Volume и переменные.
2. Временно поставь **Active workers = 1**, **Max workers = 1**.
3. Дождись в Workers → Logs обеих строк после `Downloading:` / `Ready:`:

```text
Graph schema validated: animate-ki
Graph schema validated: animate-wrapper
```

4. Поставь **Active workers = 0**, `DOWNLOAD_MODELS=0`. Перед генерацией выбери GPU с достаточной памятью.

**Active workers = 1 оплачивается и в простое; загрузка моделей расходует оплачиваемое время.** Volume оплачивается отдельно. [Настройки Runpod](https://docs.runpod.io/serverless/endpoints/endpoint-configurations). Увеличенный init timeout нужен для длительного первоначального скачивания. [Runpod initialization](https://docs.runpod.io/serverless/development/optimization).

## 4. Подключение к Studio

В серверном `.env`:

```dotenv
RUNPOD_API_KEY=ключ_Runpod
RUNPOD_ENDPOINT_ID=ID_endpoint_без_URL
```

После изменения перезапусти `ofm-studio`. `SIRAY_API_KEY` используется для Nano Banana 2 и Gemini; `STUDIO_PUBLIC_URL` должен вести на Studio. Ключи остаются на VPS/в настройках endpoint.

В Studio доступны **Kiara Animate** и **SSIBAL Animate · Uni3C**. Внутренние ID сохранены: `runpod/animate-ki` и `runpod/animate-wrapper`.

С подготовкой фото: первое изображение — модель, следующие — первые кадры; видеореференсы идут в том же порядке. Для трёх роликов нужны 4 изображения и 3 видео. Без подготовки фото: одно готовое фото на один видеореференс. До 20 роликов в партии.

### Длительность Kiara

В присланном API `TOTAL DURATION` (439) умножается на 25 FPS (441), а загрузчик использует 30 FPS (52). Поэтому 10 даёт приблизительно 8,4 секунды; фиксированное +2 не универсально.

Worker считает кадры по **длительности и FPS из Studio**, без рассогласования и без пропуска первого кадра. Видео не длиннее референса. Для Wan число кадров округляется вниз до 4n+1; разница с заданной длительностью — менее 4/FPS секунды (менее 0,134 с при 30 FPS). Перекрытия блоков Kiara удаляются перед финальной сборкой.

Первый тест: один ролик, 480×832, 5 секунд, 4 шага, сначала готовое фото без автопромптинга/подготовки фото. Затем проверить полную цепочку. Это реальные платные задачи; новый образ и сквозная GPU-генерация ещё не проверены.

## 5. Ноды и адаптации

Kiara: исходные LoRA 0,95/0,3, новые модели, pose/face detection, WanAnimateToVideo. UI-цикл развёрнут в блоки до 77 кадров с продолжением на 5 кадрах. SSIBAL: WanVideoWrapper, исходные LoRA и силы, Uni3C, TSPoseDataSmoother, TSColorMatch, финальный MP4 со звуком референса.

UI/preview/Seed/Set/Get/Easy Use заменены параметрами API. torch.compile/SageAttention и отключённая ручная сегментация не включаются; используется SDPA. Совпадение пиксель-в-пиксель с интерактивным запуском не обещается.

TSColorMatch в закреплённой версии сохраняет дополнительный MP4 и видео каждого окна. Проверяемый build-time patch `patch_nodes.py` отключает эти записи, сохраняя вычисления и возвращаемые кадры. Итоговый MP4 сохраняет только выходная нода. Кадры SSIBAL дополнительно ограничиваются запросом перед финальной сборкой.

Файлы аудита:

- `nodes.lock.json`: пять необходимых runtime-пакетов, закреплённых commit SHA.
- `provided-nodes.json`: все 16 уникальных репозиториев из списка пользователя и необходимость установки.
- `workflow-inventory.json`: ноды обоих UI-workflow, включая отключённые.
- `models.json`: закреплённые ссылки, размер, SHA-256, профили и опциональные файлы.

Все четыре исходных workflow JSON включены в образ. Ноды устанавливаются при сборке, модели остаются на Volume. Старый Pod-шаблон сохранён отдельно локально и не входит в архив.

## 6. Результат и восстановление

MP4 загружается напрямую на VPS по подписанному адресу задачи, без base64/S3. Studio проверяет подпись, владельца и личную квоту; повторная доставка не создаёт дубликат. Администратор видит файлы пользователя через существующую админку.

Состояние шагов и ID платных запросов сохраняются в базе. После перезапуска продолжается проверка отправленного задания. Неоднозначная отправка автоматически не повторяется. Доставка MP4 повторяется до трёх раз; постоянный сбой требует проверки логов и может потребовать новой генерации. Успешно доставленный файл остаётся в библиотеке независимо от срока хранения ответа Runpod.

Gemini подключён через Siray `google/gemini-3-flash-preview`; мультимодальный запрос и полный платный запуск ещё требуют проверки на этом аккаунте. [Siray Gemini](https://docs.siray.ai/api-reference/model-api/gemini-3-flash-preview).

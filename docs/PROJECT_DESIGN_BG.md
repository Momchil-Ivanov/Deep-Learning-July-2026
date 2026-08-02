# TrialCriteriaNER: подробно описание на проекта

## 1. Идея и цел

**TrialCriteriaNER** е Deep Learning проект за автоматично разпознаване на
медицински понятия в критериите за включване и изключване на участници в
клинични изпитвания.

Критериите обикновено са написани като свободен текст. Например:

> Patients with diabetes and HbA1c above 7% receiving metformin.

Желаният структуриран резултат е:

- `diabetes` → `Condition`
- `HbA1c` → `Measurement`
- `7%` → `Value`
- `metformin` → `Drug`

Основният изследователски въпрос е:

> Може ли компактен Transformer модел да разпознава медицинските понятия в
> eligibility criteria по-точно от класически CRF модел?

Проектът извлича информация от текст. Той няма да решава дали конкретен пациент
трябва да участва в изпитване и не представлява медицинско средство.

## 2. Реалният проблем

Подборът на подходящи участници е важна и трудоемка част от клиничните
изпитвания. Критериите са разнообразни, съдържат медицински термини, числови
граници, времеви условия и отрицания. Ръчното им превръщане в структурирани
заявки е бавно и трудно за мащабиране.

Named Entity Recognition (NER) е първата стъпка към структуриране на този текст.
Разпознатите понятия впоследствие могат да подпомогнат:

- търсене и филтриране на клинични изпитвания;
- подготовка на формални eligibility правила;
- електронен prescreening под контрола на медицински специалист;
- анализ на сложността и честотата на различни видове критерии.

## 3. Защо използваме невронна мрежа

Използването на невронна мрежа е обосновано от естеството на данните:

1. **Значението зависи от контекста.** Една дума може да има различна роля
   според околните думи.
2. **Медицинският език има много парафрази.** Речник или regular expressions
   не могат надеждно да изброят всички възможни формулировки.
3. **Трябва да се открият и границата, и класът на понятието.** Например
   моделът трябва да различи цялото `body mass index` от отделните думи.
4. **Transformer-ът използва двупосочен контекст.** Представянето на всеки
   token зависи от текста преди и след него.
5. **Transfer learning намалява нуждата от огромен набор.** Предварително
   обученият DistilBERT вече съдържа езикови представяния и се дообучава за
   конкретните CHIA етикети.

Твърдението, че невронната мрежа е полезна, няма да се приема на доверие.
DistilBERT ще бъде сравнен с класически Conditional Random Field (CRF) върху
един и същ заключен test set.

## 4. Защо тази тема е подходяща за изпитния проект

- Решава ясно дефиниран реален проблем, свързан директно с клинични изпитвания.
- Deep Learning моделът е съществена част от решението, а не формално допълнение.
- Има публичен, експертно анотиран набор без лични пациентски данни.
- Има измерими gold-standard етикети и ясни precision, recall и F1 метрики.
- Има публикувани резултати, с които могат да се сравнят нашите експерименти.
- Обхватът е реалистичен за срока и наличната GTX 1050 Ti с 4 GB VRAM.
- Позволява смислени визуализации, тестове и подробен error analysis.

Други разгледани теми са по-рискови:

- При prediction на прекратяване `Completed` не означава клиничен успех, а
  registry данните могат да съдържат информация, добавена след резултата.
- При patient-to-trial matching липсват леснодостъпни реални patient profiles
  и надеждни relevance labels за честна оценка.
- Reinforcement learning няма естествен agent, environment и reward в тази
  задача и би добавил сложност без научна необходимост.

## 5. Данни: CHIA

Ще използваме публичния корпус
[CHIA](https://doi.org/10.1038/s41597-020-00620-0), създаден от критерии в
ClinicalTrials.gov:

- 1 000 Phase IV интервенционни клинични изпитвания;
- 12 409 анотирани eligibility criteria;
- 44 616 raw text-bound annotations в използвания архив;
- 30 raw annotation типа, включително model entities и quality/error tags.

Този проект се фокусира върху entity recognition. Relationship extraction не е
част от основния обхват.

Официалният `bigbio/chia` Hugging Face dataset използва Python loading script.
`datasets==5.0.0` вече не поддържа такива scripts, затова опитът за директно
`load_dataset` правилно завърши с incompatibility error. Вместо да понижаваме
библиотеката, loader-ът изтегля revision-pinned
`data/chia_without_scope.zip` и парсва оригиналните BRAT `.txt/.ann` файлове.
Това прави acquisition и parsing процеса явен и възпроизводим.

За директно сравнение с публикуваното Transformer изследване ще използваме
същите 11 major entity типа:

- `Condition`, `Device`, `Drug`, `Measurement`;
- `Mood`, `Observation`, `Person`;
- `Pregnancy_considerations`, `Procedure`, `Temporal`, `Value`.

Тези класове съдържат 34 396 entities преди разрешаване на overlaps. Останалите
10 220 raw annotations са construct или quality/error категории и няма да се
използват като model labels.

Всеки запис трябва да запази:

- `NCT ID`;
- inclusion или exclusion тип;
- оригиналния текст;
- character offsets;
- entity type.

Преди моделиране ще проверим празни текстове, невалидни offsets, дублирани
annotations, неизвестни labels, class imbalance и nested/overlapping spans.

Изпълненият audit потвърди:

- 2 000 inclusion/exclusion документа от точно 1 000 NCT IDs;
- 12 409 непразни criteria lines;
- 1 000 inclusion и 1 000 exclusion документа;
- нула дублирани entity IDs;
- нула невалидни spans и нула text mismatches за 11-те model labels;
- 2 поправени offsets, при които annotation текстът има точно едно срещане;
- 2 523 припокриващи се model entity двойки, от които 2 515 nested;
- 3 празни документа и 66 документа без entity от избраните 11 типа.

## 6. Представяне с BIO етикети

Character spans ще бъдат преобразувани в token labels:

- `B-Type` — начало на entity;
- `I-Type` — продължение на entity;
- `O` — token извън entity.

Пример:

| Token | BIO label |
|---|---|
| HbA1c | B-Measurement |
| above | O |
| 7 | B-Value |
| % | I-Value |

Обикновеният BIO формат не може да пази две припокриващи се entities върху един
token. Затова ще използваме предварително определена и документирана политика,
например избор на по-дългия span. Ще отчетем колко annotations се губят поради
тази трансформация.

Реализираната политика подрежда entities по envelope length и covered length,
запазва най-дългия при реален character overlap и не слива само допиращи се
spans. Така 34 396 model entities стават 32 434 и не остава нито една
припокриваща се двойка. Официалният research notebook слива и съседни spans и
третира discontinuous entity като един envelope; приложен върху текущия архив,
неговият код дава 31 996 entities спрямо 31 944 в статията. Нашата политика е
по-близка до стандартната NER дефиниция, а разликата остава явно документирана.

WordPiece token понякога пресича границата между два съседни character spans,
например един `[UNK]` token може да съдържа края на `Measurement` и началото на
`Value`. Тогава token-ът се присвоява на entity-то с по-голямо character
припокриване; tie-break правилото е детерминирано. Всички 12 409 criteria
преминаха token alignment без грешка.

Разделянето на train, validation и test ще бъде по цели `NCT ID` групи, а не по
отделни criteria. Това предотвратява попадане на текстове от едно изпитване в
различни подмножества.

Използваме seed `42` и split 800/100/100 NCT IDs. Получените размери са:

- train: 9 806 criteria и 25 625 entities;
- validation: 1 293 criteria и 3 285 entities;
- test: 1 310 criteria и 3 524 entities;
- NCT leakage: 0.

DistilBERT token audit показа median 15, p95 56, p99 96 и максимум 507 tokens.
Само 48 criteria са над 128 tokens и 5 са над 256. Затова training
`max_length=128` остава подходящ за 4 GB VRAM, но дългите criteria ще се
обработват с overflow windows, а не чрез тихо изрязване.

## 7. Модели

### 7.1. CRF baseline

Реализираният CRF използва regex tokenization и локални характеристики:

- lowercase форма;
- prefixes и suffixes с дължина 2 и 3;
- token shape, дължина, главни букви и цифри;
- начална и крайна позиция в criterion;
- lowercase, shape, title case и uppercase характеристики на предходния и
  следващия token.

Това е класически sequence-labeling baseline. Той показва какво може да се
постигне с ръчно зададени локални зависимости без Transformer.

Преди пълното обучение беше изпълнен smoke test с 300 training criteria и 10
L-BFGS итерации. Той потвърди, че feature extraction, BIO prediction,
entity-level evaluation и strict/relaxed quality checks работят от край до
край.

Пълният модел беше обучен само върху 800-те training NCT IDs: 9 806 criteria и
141 141 regex tokens. Използваните фиксирани настройки са `c1=0.1`, `c2=0.1`,
100 L-BFGS итерации и `all_possible_transitions=True`. Не е правено настройване
по test резултата.

Резултати:

- validation strict precision `0.6710`, recall `0.6008`, F1 `0.6339`;
- validation relaxed precision `0.7980`, recall `0.7145`, F1 `0.7540`;
- test strict precision `0.6343`, recall `0.5486`, F1 `0.5884`;
- test relaxed precision `0.7844`, recall `0.6783`, F1 `0.7275`;
- test strict F1 по criteria type: `0.6167` за exclusion и `0.5410` за
  inclusion.

Обучението на CPU отне приблизително 82.1 секунди. Pickle моделът е около 3.27
MB и се записва в Git-ignored `checkpoints/crf_baseline.pkl`. Подробният JSON
отчет с резултати по entity type е в
`artifacts/crf_baseline_metrics.json`. Най-силните test strict резултати са за
`Person` и `Value`, а редките или по-нееднозначни `Device`, `Mood`,
`Observation` и `Pregnancy_considerations` остават значително по-трудни.

### 7.2. DistilBERT

Pipeline:

1. Текстът се разделя на WordPiece tokens.
2. Character offsets се подравняват към token positions.
3. DistilBERT създава contextual embedding за всеки token.
4. Linear classification head извежда вероятности за BIO класовете.
5. Cross-entropy loss обучава модела; padding и special tokens получават label
   `-100` и не участват в loss.
6. Предсказаните BIO етикети се сглобяват обратно в текстови spans.

Началните настройки за 4 GB VRAM са:

- `max_length=128`;
- physical batch size 1–2;
- gradient accumulation до effective batch около 16;
- AdamW с learning rate около `2e-5`;
- weight decay `0.01`;
- warmup около 10%;
- 3–5 epochs с early stopping;
- gradient checkpointing при необходимост;
- checkpoint и evaluation след всяка epoch.

FP16 ще се използва само ако кратък benchmark покаже стабилност и полза. GTX
1050 Ti няма Tensor Cores, затова FP16 не е автоматично по-бърз.

## 8. Поетапна работа и quality gates

Не преминаваме към следващ етап, докато текущият не е проверен.

1. **Среда и GPU** — imports, dependency check и реална CUDA tensor операция.
2. **CHIA loader** — schema, размери, липсващи данни и ръчна проверка на записи.
3. **BIO alignment** — unit tests и round-trip примери.
4. **Data split** — автоматична проверка за общи NCT IDs.
5. **CRF** — mini-run преди пълното обучение.
6. **DistilBERT smoke test** — loss, gradients, VRAM, evaluation и checkpoint
   save/resume върху малка извадка.
7. **Пълно обучение** — следене на loss, F1, време и видеопамет.
8. **Test evaluation** — еднократно след избор на модел по validation set.
9. **Reproducibility** — чисто изпълнение на notebook, тестове и README.

При грешка спираме на съответния етап и документираме:

- точната грешка или неправилно поведение;
- доказаната или най-вероятната причина;
- какво е проверено и изключено;
- приложената корекция;
- резултата от повторната и regression проверката;
- безопасен fallback, ако проблемът остане.

## 9. Оценяване

Основната метрика ще бъде entity-level strict micro-F1. Ще отчетем:

- strict precision, recall и F1;
- relaxed overlap precision, recall и F1;
- F1 по entity тип;
- резултати за inclusion и exclusion criteria;
- CRF срещу DistilBERT върху един и същ test set.

Token accuracy няма да бъде основна метрика, защото многобройният `O` клас може
да даде подвеждащо висока стойност.

Error analysis ще групира:

- грешна граница;
- грешен entity тип;
- пропуснат entity;
- неправилно добавен entity;
- truncation;
- проблем от nested annotations.

## 10. Предишни изследвания

Минималните основни източници са:

1. Kury et al.,
   [Chia, a large annotated corpus of clinical trial eligibility criteria](https://doi.org/10.1038/s41597-020-00620-0).
2. Zhang et al.,
   [Transformer-Based Named Entity Recognition for Parsing Clinical Trial Eligibility Criteria](https://pmc.ncbi.nlm.nih.gov/articles/PMC8373041/).

Второто изследване избира 11 major CHIA типа, запазва най-дългия span при
overlap и разделя 1 000 trials на 800/100/100. То отчита strict F1 `0.6339` за
BERT и `0.6578` за най-добрия RoBERTa-MIMIC-Trial модел. Ще следваме същия
висок-level protocol, но нашият split seed и scientifically stricter overlap
policy не възпроизвеждат напълно оригиналния preprocessing. Затова сравнението
с published F1 ще бъде ориентировъчно, докато CRF срещу DistilBERT върху нашия
общ test set ще бъде директното експериментално сравнение. DistilBERT е по-малък
модел, избран за 4 GB VRAM, затова не се приема предварително, че ще достигне
най-добрия публикуван резултат.

## 11. Checkpoints и работа от друго място

Физическото местоположение не влияе на обучението. Checkpoint-ът трябва да
съдържа model weights, optimizer state, scheduler state, trainer state и random
state. `resume_from_checkpoint` позволява продължаване от последната запазена
стъпка.

Кодът използва относителни пътища. Средата се възстановява от
`requirements.txt`. Checkpoints не се commit-ват в GitHub, защото обикновено
надвишават лимита от 100 MB. Те могат да се синхронизират отделно чрез MEGA,
Google Drive или Hugging Face Hub.

Преди пътуване ще направим кратък save/resume тест. Ако се използва друга
NVIDIA машина, checkpoint-ът остава преносим, но първото resume изпълнение ще
запази същите training настройки.

## 12. Потвърдена локална среда

На 27 юли 2026 г. преминаха следните проверки:

- Python 3.11.9, 64-bit;
- NVIDIA driver 582.53;
- NVIDIA GeForce GTX 1050 Ti, 4 GB, compute capability 6.1;
- PyTorch 2.13.0 с CUDA 12.6 runtime;
- `torch.cuda.is_available() == True`;
- успешна матрична CUDA операция;
- успешен import на всички директни зависимости;
- `pip check` без dependency conflicts.

Използва се CUDA 12.6 PyTorch wheel, защото Pascal `sm_61` не е включен в
актуалните CUDA 13 wheels.

## 13. Ограничения и етика

- CHIA съдържа Phase IV изпитвания и резултатите може да не се обобщават към
  всички фази и терапевтични области.
- BIO преобразуването губи част от nested структурата.
- Редките entity класове вероятно ще имат по-нисък F1.
- Registry текстът не е равностоен на електронно здравно досие.
- Моделът не е валидиран за clinical decision-making.
- Предсказанията трябва да бъдат преглеждани от квалифициран специалист при
  реална употреба.

## 14. Съответствие с изпитните критерии

- **Problem statement:** реален и ясно ограничен clinical-trial NLP проблем.
- **Layout:** структуриран Jupyter notebook и отделни source модули.
- **Code quality:** функции, type hints, docstrings и тестове.
- **Previous research:** поне двата основни източника и сравнение на резултати.
- **Data:** възпроизводимо зареждане, cleaning, BIO formatting и data audit.
- **Testing:** unit tests, leakage checks, validation/test split и smoke tests.
- **Visualization:** class distributions, training curves, per-class F1 и
  маркирани prediction примери.
- **Communication:** английски notebook и README, ясни ограничения и
  изследователска история от проблема до заключението.

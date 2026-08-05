# SQL - аналитические задачи

В разделе собраны запросы, решающие бизнес-задачи исследования рынков, отбора инвестиционных инструментов, подготовки биржевых данных и анализа телеком-инфраструктуры.

## Что демонстрирует код

- PostgreSQL: CTE, LAG, CASE, JOIN, группировки, оконные функции и строковые операции.
- PostgreSQL DDL/DML: создание и изменение таблиц, составной первичный ключ, INSERT и UPDATE.
- ClickHouse: агрегации и фильтрация больших массивов данных.

Файл [`business_analysis.sql`](business_analysis.sql) содержит очищенные версии запросов. Повторяющийся кейс по акциям объединён в один сценарий, а расчёты объёма торгов названы нейтрально: исходное поле `volume` требует проверки единицы измерения перед интерпретацией как денежного оборота.

[business_analysis.sql](https://github.com/user-attachments/files/30756844/business_analysis.sql)
-- 1. Healthcare insurance market data preparation (PostgreSQL)
SELECT
    s.security,
    CASE
        WHEN LOWER(s.sub_industry) LIKE '%insurance%' THEN 'Insurance'
        WHEN s.sector = 'Health Care' THEN 'Health Care'
        ELSE 'Other'
    END AS category,
    s.sector,
    s.sub_industry,
    SPLIT_PART(s.adress, ',', 1) AS city
FROM securities AS s
WHERE SPLIT_PART(s.adress, ',', 1) = 'New York'
  AND s.sector IN ('Financials', 'Health Care')
  AND (
      LOWER(s.sub_industry) LIKE '%insurance%'
      OR s.sector = 'Health Care'
  )
ORDER BY s.sector, s.sub_industry, s.security;


-- 2. Volatile US equity screening (PostgreSQL)
WITH daily_changes AS (
    SELECT
        p.symbol,
        p.date,
        p.close - LAG(p.close) OVER (
            PARTITION BY p.symbol
            ORDER BY p.date
        ) AS daily_change
    FROM prices AS p
),
stock_metrics AS (
    SELECT
        p.symbol,
        MAX(p.high) AS max_price,
        MIN(p.low) AS min_price,
        SUM(p.volume) AS total_volume,
        COUNT(*) AS trading_days,
        AVG(d.daily_change) AS average_daily_change
    FROM prices AS p
    LEFT JOIN daily_changes AS d
        ON p.symbol = d.symbol
       AND p.date = d.date
    GROUP BY p.symbol
)
SELECT
    s.symbol,
    s.security,
    s.adress,
    m.average_daily_change,
    m.max_price,
    m.min_price,
    m.total_volume,
    m.trading_days
FROM stock_metrics AS m
JOIN securities AS s
    ON s.symbol = m.symbol
WHERE m.max_price > 200
  AND m.min_price < 30
  AND m.average_daily_change > 0
  AND m.total_volume > 5000000
  AND m.trading_days > 504
ORDER BY m.average_daily_change DESC;


-- 3. Exchange price table lifecycle (PostgreSQL)
CREATE TABLE sandbox.prices_stock_ilya_kostkin (
    date DATE,
    symbol VARCHAR(10),
    open DOUBLE PRECISION,
    close DOUBLE PRECISION,
    low DOUBLE PRECISION,
    high DOUBLE PRECISION,
    volume DOUBLE PRECISION
);

ALTER TABLE sandbox.prices_stock_ilya_kostkin
RENAME TO prices_new_ilya_kostkin;

ALTER TABLE sandbox.prices_new_ilya_kostkin
ADD COLUMN in_portfolio BOOLEAN;

ALTER TABLE sandbox.prices_new_ilya_kostkin
ADD CONSTRAINT prices_new_ilya_kostkin_pk
PRIMARY KEY (date, symbol);

INSERT INTO sandbox.prices_new_ilya_kostkin
    (date, symbol, open, close, low, high, volume, in_portfolio)
VALUES
    ('2024-05-05', 'RUSS', 124, 125, 121, 129, 4000000, TRUE);

UPDATE sandbox.prices_new_ilya_kostkin
SET high = 130
WHERE symbol = 'RUSS'
  AND date = '2024-05-05';

INSERT INTO sandbox.prices_new_ilya_kostkin
    (date, symbol, open, close, low, high, volume, in_portfolio)
SELECT
    '2024-05-04',
    symbol,
    0,
    0,
    0,
    0,
    volume,
    in_portfolio
FROM sandbox.prices_new_ilya_kostkin
WHERE symbol = 'RUSS'
  AND date = '2024-05-05';


-- 4. Telecom tower summary (ClickHouse)
SELECT count() AS dataset_rows
FROM cell_towers;

SELECT radio
FROM cell_towers
WHERE mcc = 250
  AND area = 30440
  AND cell = 1041;

SELECT
    radio,
    count() AS towers
FROM cell_towers
WHERE lat BETWEEN 54 AND 56
  AND lon BETWEEN 54 AND 56
GROUP BY radio
ORDER BY towers DESC;

SELECT
    cell,
    count() AS towers
FROM cell_towers
WHERE lat BETWEEN 54 AND 56
  AND lon BETWEEN 54 AND 56
GROUP BY cell
ORDER BY towers DESC
LIMIT 1;

SELECT
    radio,
    avg(samples) AS average_samples,
    min(created) AS first_created,
    max(created) AS last_created,
    min(updated) AS first_updated,
    max(updated) AS last_updated,
    countIf(updated = created) AS never_updated_after_creation
FROM cell_towers
WHERE lat BETWEEN 54 AND 56
  AND lon BETWEEN 54 AND 56
GROUP BY radio
ORDER BY radio;


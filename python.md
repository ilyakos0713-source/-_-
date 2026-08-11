# Python - аналитика и моделирование

## Оценка маркетинговой акции

Построен контрфактический прогноз выручки на 92 дня акции. Использованы обработка выбросов по IQR, семидневное сглаживание, тест Дики-Фуллера и Prophet.

- качество модели на валидации: R2 = 0,96;
- верхняя граница прогноза: 156 393,55 доллара;
- фактическая выручка: 184 435 долларов;
- превышение прогноза: 28 041,45 доллара, или 17,93%.

## Прогноз отмены бронирований

Обработаны 119 390 бронирований: заполнены пропуски, удалены дубликаты, подготовлены признаки и визуализации. Baseline-модель логистической регрессии показала accuracy 0,72. Перед публикацией будут добавлены stratified split, confusion matrix, precision, recall, F1 и ROC-AUC.

## Рынок аренды недвижимости

На 5 000 объявлениях построена baseline-модель справедливой арендной ставки по площади, этажу, числу комнат и региону. R2 = 0,543. Модель используется как отправная точка для дальнейшего feature engineering и сравнения алгоритмов.

## Рынок подержанных автомобилей Индии

Проанализированы 16 747 объявлений. После отбора автомобилей 2017 года и новее:

- 9 188 автомобилей вошли в возрастной фильтр;
- 8 478 дополнительно соответствуют ограничению двигателя до 2 000 куб. см - 92,3%;
- 99,95% имеют пробег не выше 200 000 км;
- найдено 286 моделей, впервые появившихся в данных в 2022-2023 годах;
- средний индекс качества равен 24,98 при пороге перспективности 20.

Курс 1,16 рубля за рупию в исходной работе является фиксированным допущением и должен сопровождаться датой расчёта.

## Геоаналитика экспансии

В командной работе объединены юнит-экономика и пространственный анализ 527 конкурентов в 8 городах США. Использованы DBSCAN, Haversine, сетка 2 x 2 км и Folium. Устойчивый рейтинг при нескольких системах весов вывел Portland на первое место; сформировано 13 рекомендуемых зон в пяти городах.

## Код
[python_analytics_portfolio.py](https://github.com/user-attachments/files/30939086/python_analytics_portfolio.py)

#!/usr/bin/env python3

Projects
--------
1. Rental price prediction.
2. Hotel booking cancellation prediction.
3. Travel marketing campaign evaluation with a counterfactual forecast.
4. Indian used-car market analysis.
5. Bank marketing response prediction.
6. Restaurant-chain expansion and geospatial competitor analysis.

# ---------------------------------------------------------------------------
# Project 1: rental price prediction
# ---------------------------------------------------------------------------


RENTAL_COLUMNS = [
    "id",
    "create_date",
    "rent_price",
    "deposit",
    "floor",
    "area_total",
    "rooms",
    "region",
    "repair_type",
    "agent_fee",
]


def _normalise_rental_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Support both a clean schema and the unnamed-column source export."""
    data = frame.copy()
    required_model_columns = {"rent_price", "floor", "area_total", "rooms", "region"}

    has_usable_named_schema = required_model_columns.issubset(data.columns) and all(
        data[column].notna().any() for column in required_model_columns
    )
    if has_usable_named_schema:
        return data

    unnamed_columns = [column for column in data.columns if str(column).startswith("Unnamed")]
    if len(unnamed_columns) >= 7 and data.shape[1] >= 10:
        data = data.iloc[:, :10].copy()
        data.columns = RENTAL_COLUMNS
        return data

    _require_columns(data, required_model_columns, "Rental price prediction")
    return data


def run_rental_price_project(
    input_csv: str | Path,
    output_dir: str | Path,
    test_size: float = 0.30,
) -> dict[str, Any]:
    """Clean rental listings and train a linear-regression baseline."""
    output = _ensure_output_dir(output_dir)
    raw = pd.read_csv(input_csv)
    data = _normalise_rental_schema(raw)
    _require_columns(
        data,
        ["rent_price", "floor", "area_total", "rooms", "region"],
        "Rental price prediction",
    )

    if "create_date" in data:
        data["create_date"] = pd.to_datetime(data["create_date"], errors="coerce")

    data["area_total"] = (
        data["area_total"]
        .astype(str)
        .str.extract(r"(-?\d+(?:[.,]\d+)?)", expand=False)
        .str.replace(",", ".", regex=False)
    )
    numeric_columns = [
        "rent_price",
        "deposit",
        "floor",
        "area_total",
        "rooms",
        "region",
        "agent_fee",
    ]
    for column in numeric_columns:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    if "repair_type" in data:
        data["repair_type"] = data["repair_type"].fillna("unknown")
    for column in ["deposit", "agent_fee"]:
        if column in data:
            data[column] = data[column].fillna(0)

    rows_raw = len(data)
    data = data.drop_duplicates().copy()
    data = data[
        data["rent_price"].gt(0)
        & data["area_total"].gt(0)
        & data["area_total"].lt(45_000)
        & data["floor"].gt(0)
        & data["rooms"].ge(0)
    ].dropna(subset=["rent_price", "floor", "area_total", "rooms", "region"])

    if len(data) < 20:
        raise ValueError("Rental price prediction: at least 20 valid rows are required.")

    features = ["floor", "area_total", "rooms", "region"]
    x_train, x_test, y_train, y_test = train_test_split(
        data[features],
        data["rent_price"],
        test_size=test_size,
        random_state=RANDOM_STATE,
    )
    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    prediction_table = x_test.copy()
    prediction_table["actual_rent_price"] = y_test
    prediction_table["predicted_rent_price"] = predictions
    prediction_table["absolute_error"] = np.abs(
        prediction_table["actual_rent_price"] - prediction_table["predicted_rent_price"]
    )
    prediction_table.to_csv(output / "predictions.csv", index=False)
    data.to_csv(output / "cleaned_rent_listings.csv", index=False)

    coefficients = pd.DataFrame(
        {"feature": features, "coefficient": model.coef_}
    ).sort_values("coefficient", key=np.abs, ascending=False)
    coefficients.to_csv(output / "coefficients.csv", index=False)

    metrics = {
        "project": "rental_price_prediction",
        "rows_raw": rows_raw,
        "rows_after_cleaning": len(data),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "r2": r2_score(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": mean_squared_error(y_test, predictions) ** 0.5,
        "intercept": model.intercept_,
    }
    _save_metrics(metrics, output)
    return metrics


# ---------------------------------------------------------------------------
# Project 2: hotel booking cancellations
# ---------------------------------------------------------------------------


HOTEL_BASE_FEATURES = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
]


def run_hotel_cancellation_project(
    input_csv: str | Path,
    output_dir: str | Path,
    test_size: float = 0.33,
) -> dict[str, Any]:
    """Prepare hotel bookings and predict cancellation probability."""
    output = _ensure_output_dir(output_dir)
    data = pd.read_csv(input_csv)
    _require_columns(
        data,
        ["is_canceled", *HOTEL_BASE_FEATURES],
        "Hotel cancellation prediction",
    )
    rows_raw = len(data)

    fill_values = {"children": 0, "country": "NA", "agent": -1, "company": -1}
    for column, value in fill_values.items():
        if column in data:
            data[column] = data[column].fillna(value)
    data = data.drop_duplicates().copy()

    date_parts = {
        "arrival_date_year",
        "arrival_date_month",
        "arrival_date_day_of_month",
    }
    if date_parts.issubset(data.columns):
        data["arrival_date"] = pd.to_datetime(
            data["arrival_date_year"].astype(str)
            + "-"
            + data["arrival_date_month"].astype(str)
            + "-"
            + data["arrival_date_day_of_month"].astype(str),
            errors="coerce",
        )

    for column in HOTEL_BASE_FEATURES + ["is_canceled"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["is_canceled", *HOTEL_BASE_FEATURES]).copy()
    data["total_nights"] = (
        data["stays_in_weekend_nights"] + data["stays_in_week_nights"]
    )
    data["total_guests"] = data["adults"] + data["children"]
    data["is_family"] = (data["children"] > 0).astype(int)

    model_features = [*HOTEL_BASE_FEATURES, "total_nights", "total_guests", "is_family"]
    x = data[model_features]
    y = data["is_canceled"].astype(int)
    if y.nunique() < 2:
        raise ValueError("Hotel cancellation prediction: target must contain two classes.")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    prediction_table = x_test.copy()
    prediction_table["actual_is_canceled"] = y_test
    prediction_table["predicted_is_canceled"] = predictions
    prediction_table["cancellation_probability"] = probabilities
    prediction_table.to_csv(output / "predictions.csv", index=False)

    coefficients = pd.DataFrame(
        {
            "feature": model_features,
            "coefficient": model.named_steps["model"].coef_[0],
        }
    ).sort_values("coefficient", key=np.abs, ascending=False)
    coefficients.to_csv(output / "coefficients.csv", index=False)
    data[model_features + ["is_canceled"]].describe().T.to_csv(
        output / "descriptive_statistics.csv"
    )

    metrics = {
        "project": "hotel_cancellation_prediction",
        "rows_raw": rows_raw,
        "rows_after_cleaning": len(data),
        "duplicate_rows_removed": rows_raw - len(data),
        "cancellation_rate": y.mean(),
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": _safe_roc_auc(y_test, probabilities),
    }
    _save_metrics(metrics, output)
    return metrics


# ---------------------------------------------------------------------------
# Project 3: Travel promotion evaluation
# ---------------------------------------------------------------------------


def _prepare_revenue_history(
    frame: pd.DataFrame,
    date_column: str,
    revenue_column: str,
    iqr_multiplier: float,
) -> pd.DataFrame:
    data = frame[[date_column, revenue_column]].copy()
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data[revenue_column] = pd.to_numeric(data[revenue_column], errors="coerce")
    data = data.dropna(subset=[date_column]).sort_values(date_column).drop_duplicates(date_column)
    data[revenue_column] = data[revenue_column].interpolate().bfill().ffill()

    q1, q3 = data[revenue_column].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr
    previous_rolling_mean = data[revenue_column].shift(1).rolling(7, min_periods=1).mean()
    outlier_mask = ~data[revenue_column].between(lower, upper)
    data["processed_revenue"] = data[revenue_column].where(
        ~outlier_mask,
        previous_rolling_mean,
    )
    data["processed_revenue"] = data["processed_revenue"].fillna(data[revenue_column])
    data["was_outlier"] = outlier_mask
    return data


def run_travel_promo_project(
    historical_csv: str | Path,
    promo_csv: str | Path,
    output_dir: str | Path,
    date_column: str = "dates",
    revenue_column: str = "Revenue",
    iqr_multiplier: float = 1.30,
) -> dict[str, Any]:
    """Build a no-promotion forecast and compare it with actual revenue."""
    try:
        from prophet import Prophet
    except ImportError as error:
        raise ImportError(
            "Travel promotion evaluation requires Prophet: pip install prophet"
        ) from error

    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError as error:
        raise ImportError(
            "Travel promotion evaluation requires statsmodels: pip install statsmodels"
        ) from error

    output = _ensure_output_dir(output_dir)
    historical_raw = pd.read_csv(historical_csv)
    promo = pd.read_csv(promo_csv)
    _require_columns(
        historical_raw,
        [date_column, revenue_column],
        "Travel promotion evaluation — historical data",
    )
    _require_columns(
        promo,
        [revenue_column],
        "Travel promotion evaluation — promotion data",
    )

    history = _prepare_revenue_history(
        historical_raw,
        date_column=date_column,
        revenue_column=revenue_column,
        iqr_multiplier=iqr_multiplier,
    )
    prophet_data = history.rename(
        columns={date_column: "ds", "processed_revenue": "y"}
    )[["ds", "y"]]
    holdout_size = max(1, int(len(prophet_data) * 0.30))
    if len(prophet_data) - holdout_size < 30:
        raise ValueError("Travel promotion evaluation: at least 45 historical rows are required.")

    train = prophet_data.iloc[:-holdout_size]
    test = prophet_data.iloc[-holdout_size:]
    validation_model = Prophet()
    validation_model.fit(train)
    validation_forecast = validation_model.predict(test[["ds"]])
    validation_r2 = r2_score(test["y"], validation_forecast["yhat"])
    validation_rmse = mean_squared_error(test["y"], validation_forecast["yhat"]) ** 0.5

    promo[revenue_column] = pd.to_numeric(promo[revenue_column], errors="coerce")
    promo = promo.dropna(subset=[revenue_column]).copy()
    forecast_days = len(promo)
    if forecast_days == 0:
        raise ValueError("Travel promotion evaluation: promotion data contains no revenue values.")

    final_model = Prophet()
    final_model.fit(prophet_data)
    future = final_model.make_future_dataframe(periods=forecast_days)
    forecast = final_model.predict(future).tail(forecast_days)[
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ]
    forecast.to_csv(output / "counterfactual_forecast.csv", index=False)
    history.to_csv(output / "processed_history.csv", index=False)

    actual_revenue = promo[revenue_column].sum()
    expected_revenue = forecast["yhat"].sum()
    upper_revenue = forecast["yhat_upper"].sum()
    effect_vs_expected_pct = (actual_revenue / expected_revenue - 1) * 100
    effect_vs_upper_pct = (actual_revenue / upper_revenue - 1) * 100

    adf_result = adfuller(prophet_data["y"])
    metrics = {
        "project": "travel_marketing_campaign_evaluation",
        "historical_days": len(history),
        "promo_days": forecast_days,
        "outliers_replaced": history["was_outlier"].sum(),
        "adf_p_value": adf_result[1],
        "is_stationary_at_5pct": adf_result[1] <= 0.05,
        "validation_r2": validation_r2,
        "validation_rmse": validation_rmse,
        "forecast_revenue": expected_revenue,
        "forecast_upper_revenue": upper_revenue,
        "actual_revenue": actual_revenue,
        "effect_vs_forecast_pct": effect_vs_expected_pct,
        "effect_vs_upper_bound_pct": effect_vs_upper_pct,
    }
    _save_metrics(metrics, output)
    return metrics


# ---------------------------------------------------------------------------
# Project 4: Indian used-car market
# ---------------------------------------------------------------------------


def _parse_indian_price_to_rupees(value: Any) -> float:
    text = str(value).strip()
    number = _first_number(text)
    if math.isnan(number):
        return math.nan
    lowered = text.lower()
    if "crore" in lowered:
        return number * 10_000_000
    if "lakh" in lowered:
        return number * 100_000
    return number


def run_india_cars_project(
    input_csv: str | Path,
    output_dir: str | Path,
    rub_per_rupee: float = 1.16,
    minimum_year: int = 2017,
    maximum_engine_cc: int = 2000,
) -> dict[str, Any]:
    """Analyse the used-car market and calculate the custom quality index."""
    output = _ensure_output_dir(output_dir)
    data = pd.read_csv(input_csv)
    required = [
        "full_name",
        "engine_capacity",
        "resale_price",
        "kms_driven",
        "max_power",
    ]
    _require_columns(data, required, "Indian used-car market analysis")
    rows_raw = len(data)

    data["year_car"] = pd.to_numeric(
        data["full_name"].astype(str).str.extract(r"(\d{4})", expand=False),
        errors="coerce",
    )
    data["engine_capacity_cc"] = data["engine_capacity"].map(_first_number)
    data["resale_price_rupees"] = data["resale_price"].map(_parse_indian_price_to_rupees)
    data["resale_price_rub"] = data["resale_price_rupees"] * rub_per_rupee
    data["kms_driven_numeric"] = data["kms_driven"].map(_first_number)
    data["max_power_numeric"] = data["max_power"].map(_first_number)
    data["model_name"] = data["full_name"].astype(str).str.replace(
        r"^\s*\d{4}\s*", "", regex=True
    )

    eligible_by_year = data[data["year_car"].ge(minimum_year)].copy()
    eligible = eligible_by_year[
        eligible_by_year["engine_capacity_cc"].le(maximum_engine_cc)
    ].copy()
    share_engine_eligible = (
        len(eligible) / len(eligible_by_year) if len(eligible_by_year) else math.nan
    )

    valid_quality = eligible[
        eligible["resale_price_rub"].gt(0)
        & eligible["max_power_numeric"].gt(0)
        & eligible["kms_driven_numeric"].gt(100)
    ].copy()
    valid_quality["quality_index"] = (
        np.log2(valid_quality["resale_price_rub"])
        * np.sqrt(valid_quality["max_power_numeric"])
        / np.log2(valid_quality["kms_driven_numeric"] / 100)
    ).round(2)

    current_models = set(
        eligible_by_year.loc[eligible_by_year["year_car"].ge(2022), "model_name"]
    )
    older_models = set(
        eligible_by_year.loc[eligible_by_year["year_car"].lt(2022), "model_name"]
    )
    new_models_2022_2023 = current_models - older_models
    mileage_within_200k_share = (
        eligible_by_year["kms_driven_numeric"].le(200_000).mean()
        if len(eligible_by_year)
        else math.nan
    )

    valid_quality.to_csv(output / "eligible_cars_with_quality_index.csv", index=False)
    (
        eligible_by_year.groupby("year_car", dropna=False)["resale_price_rub"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .to_csv(output / "price_by_year.csv", index=False)
    )

    average_quality_index = valid_quality["quality_index"].mean()
    metrics = {
        "project": "india_used_car_market_analysis",
        "rows_raw": rows_raw,
        "cars_from_minimum_year": len(eligible_by_year),
        "cars_meeting_year_and_engine_constraints": len(eligible),
        "share_meeting_engine_constraint": share_engine_eligible,
        "share_with_mileage_up_to_200k": mileage_within_200k_share,
        "new_models_since_2022": len(new_models_2022_2023),
        "average_quality_index": average_quality_index,
        "market_is_promising": bool(average_quality_index > 20)
        if not pd.isna(average_quality_index)
        else False,
        "rub_per_rupee": rub_per_rupee,
    }
    _save_metrics(metrics, output)
    return metrics


# ---------------------------------------------------------------------------
# Project 5: bank marketing response
# ---------------------------------------------------------------------------


def _binary_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        target = pd.to_numeric(series, errors="coerce")
    else:
        normalised = series.astype(str).str.strip().str.lower()
        mapping = {
            "yes": 1,
            "y": 1,
            "true": 1,
            "1": 1,
            "no": 0,
            "n": 0,
            "false": 0,
            "0": 0,
        }
        target = normalised.map(mapping)
    return target


def run_bank_marketing_project(
    input_csv: str | Path,
    output_dir: str | Path,
    target_column: str = "y",
    test_size: float = 0.30,
) -> dict[str, Any]:
    """Prepare bank campaign data and predict customer response."""
    output = _ensure_output_dir(output_dir)
    data = pd.read_csv(input_csv)
    _require_columns(data, [target_column], "Bank marketing response prediction")
    rows_raw = len(data)

    if "age" in data:
        data["age"] = pd.to_numeric(data["age"], errors="coerce")
        data["age"] = data["age"].fillna(data["age"].median())
        data = data[data["age"].between(15, 104)].copy()
    if "balance" in data:
        data["balance"] = pd.to_numeric(data["balance"], errors="coerce").fillna(0)
    data = data.drop_duplicates().copy()
    data[target_column] = _binary_target(data[target_column])
    data = data.dropna(subset=[target_column]).copy()
    data[target_column] = data[target_column].astype(int)

    if data[target_column].nunique() < 2:
        raise ValueError("Bank marketing response prediction: target must contain two classes.")

    drop_columns = [target_column]
    for identifier in ["Id", "id"]:
        if identifier in data:
            drop_columns.append(identifier)
    x = data.drop(columns=drop_columns)
    y = data[target_column]
    numeric_features = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in x.columns if column not in numeric_features]

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    prediction_table = pd.DataFrame(
        {
            "actual_response": y_test,
            "predicted_response": predictions,
            "response_probability": probabilities,
        },
        index=y_test.index,
    )
    prediction_table.to_csv(output / "predictions.csv", index_label="source_index")

    feature_names = model.named_steps["preprocessing"].get_feature_names_out()
    coefficients = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": model.named_steps["model"].coef_[0],
        }
    ).sort_values("coefficient", key=np.abs, ascending=False)
    coefficients.to_csv(output / "coefficients.csv", index=False)

    metrics = {
        "project": "bank_marketing_response_prediction",
        "rows_raw": rows_raw,
        "rows_after_cleaning": len(data),
        "response_rate": y.mean(),
        "numeric_features": len(numeric_features),
        "categorical_features": len(categorical_features),
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": _safe_roc_auc(y_test, probabilities),
    }
    _save_metrics(metrics, output)
    return metrics


# ---------------------------------------------------------------------------
# Project 6: restaurant-chain expansion
# ---------------------------------------------------------------------------


def _score_cities(
    cities: pd.DataFrame,
    benefit_columns: Sequence[str],
    cost_columns: Sequence[str],
    weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    _require_columns(
        cities,
        ["city", *benefit_columns, *cost_columns],
        "Restaurant expansion — city scoring",
    )
    criteria = [*benefit_columns, *cost_columns]
    if not criteria:
        raise ValueError(
            "Restaurant expansion: provide at least one benefit or cost column."
        )

    if weights:
        unknown = set(weights) - set(criteria)
        if unknown:
            raise ValueError(f"Restaurant expansion: weights contain unknown criteria {unknown}.")
        raw_weights = {criterion: float(weights.get(criterion, 0.0)) for criterion in criteria}
    else:
        raw_weights = {criterion: 1.0 for criterion in criteria}
    total_weight = sum(raw_weights.values())
    if total_weight <= 0:
        raise ValueError("Restaurant expansion: the sum of criterion weights must be positive.")
    normalised_weights = {
        criterion: weight / total_weight for criterion, weight in raw_weights.items()
    }

    result = cities.copy()
    score = pd.Series(0.0, index=result.index)
    for criterion in benefit_columns:
        component = _min_max(result[criterion], higher_is_better=True)
        result[f"score_{criterion}"] = component
        score += component * normalised_weights[criterion]
    for criterion in cost_columns:
        component = _min_max(result[criterion], higher_is_better=False)
        result[f"score_{criterion}"] = component
        score += component * normalised_weights[criterion]
    result["city_score"] = score
    result["city_rank"] = result["city_score"].rank(method="dense", ascending=False).astype(int)
    return result.sort_values(["city_rank", "city"])


def _calculate_product_unit_economics(products: pd.DataFrame) -> pd.DataFrame:
    required = ["product", "price", "variable_cost", "expected_monthly_sales"]
    _require_columns(products, required, "Restaurant expansion — product economics")
    result = products.copy()
    for column in ["price", "variable_cost", "expected_monthly_sales"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=required).copy()
    result["contribution_margin"] = result["price"] - result["variable_cost"]
    result["contribution_margin_pct"] = np.where(
        result["price"].ne(0),
        result["contribution_margin"] / result["price"],
        np.nan,
    )
    result["expected_monthly_revenue"] = (
        result["price"] * result["expected_monthly_sales"]
    )
    result["expected_monthly_contribution"] = (
        result["contribution_margin"] * result["expected_monthly_sales"]
    )
    return result.sort_values("expected_monthly_contribution", ascending=False)


def _cluster_competitors(
    competitors: pd.DataFrame,
    eps_km: float,
    min_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = ["city", "latitude", "longitude"]
    _require_columns(competitors, required, "Restaurant expansion — competitors")
    result_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    for city, group in competitors.groupby("city", dropna=False):
        local = group.copy()
        local["latitude"] = pd.to_numeric(local["latitude"], errors="coerce")
        local["longitude"] = pd.to_numeric(local["longitude"], errors="coerce")
        local = local.dropna(subset=["latitude", "longitude"])
        if local.empty:
            continue
        coordinates = np.radians(local[["latitude", "longitude"]].to_numpy())
        labels = DBSCAN(
            eps=eps_km / EARTH_RADIUS_KM,
            min_samples=min_samples,
            metric="haversine",
            algorithm="ball_tree",
        ).fit_predict(coordinates)
        local["cluster"] = labels
        result_parts.append(local)

        for cluster_id, cluster_group in local[local["cluster"].ge(0)].groupby("cluster"):
            summaries.append(
                {
                    "city": city,
                    "cluster": int(cluster_id),
                    "competitors": len(cluster_group),
                    "center_latitude": cluster_group["latitude"].mean(),
                    "center_longitude": cluster_group["longitude"].mean(),
                }
            )

    clustered = pd.concat(result_parts, ignore_index=True) if result_parts else pd.DataFrame()
    summary = pd.DataFrame(summaries)
    return clustered, summary


def _score_candidate_zones(
    candidates: pd.DataFrame,
    competitors: pd.DataFrame,
    radius_km: float,
) -> pd.DataFrame:
    required = ["city", "latitude", "longitude"]
    _require_columns(candidates, required, "Restaurant expansion — candidate zones")
    _require_columns(competitors, required, "Restaurant expansion — competitors")
    result_parts: list[pd.DataFrame] = []

    for city, group in candidates.groupby("city", dropna=False):
        local_candidates = group.copy()
        local_competitors = competitors[competitors["city"].eq(city)].copy()
        for frame in [local_candidates, local_competitors]:
            frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
            frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
            frame.dropna(subset=["latitude", "longitude"], inplace=True)
        if local_candidates.empty:
            continue

        if local_competitors.empty:
            local_candidates["nearest_competitor_km"] = np.nan
            local_candidates["competitors_within_radius"] = 0
        else:
            competitor_coordinates = np.radians(
                local_competitors[["latitude", "longitude"]].to_numpy()
            )
            candidate_coordinates = np.radians(
                local_candidates[["latitude", "longitude"]].to_numpy()
            )
            tree = BallTree(competitor_coordinates, metric="haversine")
            nearest_distance, _ = tree.query(candidate_coordinates, k=1)
            local_candidates["nearest_competitor_km"] = (
                nearest_distance[:, 0] * EARTH_RADIUS_KM
            )
            nearby = tree.query_radius(
                candidate_coordinates,
                r=radius_km / EARTH_RADIUS_KM,
                count_only=True,
            )
            local_candidates["competitors_within_radius"] = nearby

        score = _min_max(local_candidates["nearest_competitor_km"], True) * 0.45
        score += _min_max(local_candidates["competitors_within_radius"], False) * 0.35
        if "market_potential" in local_candidates:
            score += _min_max(local_candidates["market_potential"], True) * 0.20
        else:
            score += 0.10
        local_candidates["zone_score"] = score
        result_parts.append(local_candidates)

    if not result_parts:
        return pd.DataFrame()
    result = pd.concat(result_parts, ignore_index=True)
    result["zone_rank_in_city"] = result.groupby("city")["zone_score"].rank(
        method="first", ascending=False
    ).astype(int)
    return result.sort_values(["city", "zone_rank_in_city"])


def _build_restaurant_map(
    competitors: pd.DataFrame,
    candidate_zones: pd.DataFrame,
    output_path: Path,
) -> bool:
    try:
        import folium
    except ImportError:
        return False

    coordinates = []
    for frame in [competitors, candidate_zones]:
        if not frame.empty:
            coordinates.extend(
                frame[["latitude", "longitude"]].dropna().to_numpy().tolist()
            )
    if not coordinates:
        return False
    center = np.mean(np.asarray(coordinates, dtype=float), axis=0)
    map_object = folium.Map(location=center.tolist(), zoom_start=5, tiles="CartoDB positron")

    for _, row in competitors.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3,
            color="#d62728",
            fill=True,
            fill_opacity=0.65,
            tooltip=f"Competitor — {row.get('city', '')}",
        ).add_to(map_object)
    for _, row in candidate_zones.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=6,
            color="#2ca02c",
            fill=True,
            fill_opacity=0.85,
            tooltip=(
                f"Candidate — {row.get('city', '')}; "
                f"score={row.get('zone_score', math.nan):.3f}"
            ),
        ).add_to(map_object)
    map_object.save(str(output_path))
    return True


def run_restaurant_expansion_project(
    cities_csv: str | Path,
    products_csv: str | Path,
    competitors_csv: str | Path,
    output_dir: str | Path,
    benefit_columns: Sequence[str],
    cost_columns: Sequence[str],
    weights: Mapping[str, float] | None = None,
    candidates_csv: str | Path | None = None,
    eps_km: float = 2.0,
    min_samples: int = 3,
    top_cities: int = 5,
    top_zones_per_city: int = 3,
) -> dict[str, Any]:
    """Rank cities, calculate unit economics and analyse competitor geography."""
    output = _ensure_output_dir(output_dir)
    cities = pd.read_csv(cities_csv)
    products = pd.read_csv(products_csv)
    competitors = pd.read_csv(competitors_csv)

    city_ranking = _score_cities(cities, benefit_columns, cost_columns, weights)
    product_economics = _calculate_product_unit_economics(products)
    clustered_competitors, cluster_summary = _cluster_competitors(
        competitors,
        eps_km=eps_km,
        min_samples=min_samples,
    )

    city_ranking.to_csv(output / "city_ranking.csv", index=False)
    product_economics.to_csv(output / "product_unit_economics.csv", index=False)
    clustered_competitors.to_csv(output / "competitor_clusters.csv", index=False)
    cluster_summary.to_csv(output / "competitor_cluster_summary.csv", index=False)

    candidate_scores = pd.DataFrame()
    recommendations = pd.DataFrame()
    if candidates_csv:
        candidates = pd.read_csv(candidates_csv)
        candidate_scores = _score_candidate_zones(candidates, competitors, radius_km=eps_km)
        selected_cities = set(city_ranking.head(top_cities)["city"])
        recommendations = candidate_scores[
            candidate_scores["city"].isin(selected_cities)
            & candidate_scores["zone_rank_in_city"].le(top_zones_per_city)
        ].copy()
        candidate_scores.to_csv(output / "candidate_zone_scores.csv", index=False)
        recommendations.to_csv(output / "recommended_zones.csv", index=False)

    map_created = _build_restaurant_map(
        clustered_competitors if not clustered_competitors.empty else competitors,
        recommendations,
        output / "restaurant_expansion_map.html",
    )

    metrics = {
        "project": "restaurant_chain_expansion",
        "cities_analysed": len(cities),
        "products_analysed": len(product_economics),
        "competitors_analysed": len(clustered_competitors),
        "competitor_clusters": len(cluster_summary),
        "recommended_cities": city_ranking.head(top_cities)["city"].tolist(),
        "recommended_zones": len(recommendations),
        "expected_monthly_product_revenue": product_economics[
            "expected_monthly_revenue"
        ].sum(),
        "expected_monthly_contribution": product_economics[
            "expected_monthly_contribution"
        ].sum(),
        "interactive_map_created": map_created,
    }
    _save_metrics(metrics, output)
    return metrics


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def _add_output_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", required=True, help="Directory for metrics and tables.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one of the data analytics portfolio projects."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    rent = subparsers.add_parser("rent", help="Predict apartment rental prices.")
    rent.add_argument("--input", required=True)
    rent.add_argument("--test-size", type=float, default=0.30)
    _add_output_argument(rent)

    hotel = subparsers.add_parser(
        "hotel-cancellations", help="Predict hotel booking cancellations."
    )
    hotel.add_argument("--input", required=True)
    hotel.add_argument("--test-size", type=float, default=0.33)
    _add_output_argument(hotel)

    travel = subparsers.add_parser(
        "travel-promo", help="Evaluate a Travel marketing campaign."
    )
    travel.add_argument("--historical", required=True)
    travel.add_argument("--promo", required=True)
    travel.add_argument("--date-column", default="dates")
    travel.add_argument("--revenue-column", default="Revenue")
    travel.add_argument("--iqr-multiplier", type=float, default=1.30)
    _add_output_argument(travel)

    cars = subparsers.add_parser("cars-india", help="Analyse Indian used cars.")
    cars.add_argument("--input", required=True)
    cars.add_argument("--rub-per-rupee", type=float, default=1.16)
    cars.add_argument("--minimum-year", type=int, default=2017)
    cars.add_argument("--maximum-engine-cc", type=int, default=2000)
    _add_output_argument(cars)

    bank = subparsers.add_parser(
        "bank-marketing", help="Predict response to a bank campaign."
    )
    bank.add_argument("--input", required=True)
    bank.add_argument("--target-column", default="y")
    bank.add_argument("--test-size", type=float, default=0.30)
    _add_output_argument(bank)

    restaurants = subparsers.add_parser(
        "restaurant-expansion", help="Rank cities and analyse restaurant locations."
    )
    restaurants.add_argument("--cities", required=True)
    restaurants.add_argument("--products", required=True)
    restaurants.add_argument("--competitors", required=True)
    restaurants.add_argument("--candidates")
    restaurants.add_argument("--benefit-columns", required=True)
    restaurants.add_argument("--cost-columns", default="")
    restaurants.add_argument(
        "--weights",
        help='Optional JSON mapping, e.g. \'{"population": 0.4, "avg_rent": 0.2}\'.',
    )
    restaurants.add_argument("--eps-km", type=float, default=2.0)
    restaurants.add_argument("--min-samples", type=int, default=3)
    restaurants.add_argument("--top-cities", type=int, default=5)
    restaurants.add_argument("--top-zones-per-city", type=int, default=3)
    _add_output_argument(restaurants)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)

    if args.command == "rent":
        metrics = run_rental_price_project(
            args.input,
            args.output_dir,
            test_size=args.test_size,
        )
    elif args.command == "hotel-cancellations":
        metrics = run_hotel_cancellation_project(
            args.input,
            args.output_dir,
            test_size=args.test_size,
        )
    elif args.command == "travel-promo":
        metrics = run_travel_promo_project(
            args.historical,
            args.promo,
            args.output_dir,
            date_column=args.date_column,
            revenue_column=args.revenue_column,
            iqr_multiplier=args.iqr_multiplier,
        )
    elif args.command == "cars-india":
        metrics = run_india_cars_project(
            args.input,
            args.output_dir,
            rub_per_rupee=args.rub_per_rupee,
            minimum_year=args.minimum_year,
            maximum_engine_cc=args.maximum_engine_cc,
        )
    elif args.command == "bank-marketing":
        metrics = run_bank_marketing_project(
            args.input,
            args.output_dir,
            target_column=args.target_column,
            test_size=args.test_size,
        )
    elif args.command == "restaurant-expansion":
        weights = json.loads(args.weights) if args.weights else None
        metrics = run_restaurant_expansion_project(
            cities_csv=args.cities,
            products_csv=args.products,
            competitors_csv=args.competitors,
            candidates_csv=args.candidates,
            output_dir=args.output_dir,
            benefit_columns=_parse_csv_list(args.benefit_columns),
            cost_columns=_parse_csv_list(args.cost_columns),
            weights=weights,
            eps_km=args.eps_km,
            min_samples=args.min_samples,
            top_cities=args.top_cities,
            top_zones_per_city=args.top_zones_per_city,
        )
    else:
        raise AssertionError(f"Unsupported command: {args.command}")

    _print_metrics(metrics)
    return metrics


if __name__ == "__main__":
    main()

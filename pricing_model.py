from math import sqrt

FEATURES = ("reviews", "review_score", "distance_km")
RIDGE_ALPHA = 1.0


def _fit_ridge(features, prices):
    feature_count = len(FEATURES)
    sample_count = len(features)
    means = [
        sum(row[index] for row in features) / sample_count
        for index in range(feature_count)
    ]
    scales = [
        sqrt(sum((row[index] - means[index]) ** 2 for row in features) / sample_count)
        or 1.0
        for index in range(feature_count)
    ]
    price_mean = sum(prices) / sample_count
    scaled_features = [
        [(row[index] - means[index]) / scales[index] for index in range(feature_count)]
        for row in features
    ]

    gram = [
        [
            sum(row[left] * row[right] for row in scaled_features)
            + (RIDGE_ALPHA if left == right else 0.0)
            for right in range(feature_count)
        ]
        for left in range(feature_count)
    ]
    target = [
        sum(
            row[index] * (price - price_mean)
            for row, price in zip(scaled_features, prices)
        )
        for index in range(feature_count)
    ]
    coefficients = _solve(gram, target)
    return means, scales, price_mean, coefficients


def _solve(matrix, target):
    augmented = [row[:] + [value] for row, value in zip(matrix, target)]
    size = len(target)
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("Unable to solve Ridge regression coefficients.")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]

        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[index][-1] for index in range(size)]


def _predict(model, features):
    means, scales, intercept, coefficients = model
    return intercept + sum(
        coefficient * (value - mean) / scale
        for value, mean, scale, coefficient in zip(features, means, scales, coefficients)
    )


def train_pricing_model(source_data):
    listings = source_data["listings"]
    if len(listings) < 2:
        raise ValueError("At least two listings are required to train the pricing model.")

    try:
        features = [
            [float(listing[name]) for name in FEATURES]
            for listing in listings
        ]
        prices = [float(listing["price"]) for listing in listings]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Each listing must have numeric price and model features.") from error

    loo_predictions = []
    for omitted_index in range(len(listings)):
        training_features = features[:omitted_index] + features[omitted_index + 1 :]
        training_prices = prices[:omitted_index] + prices[omitted_index + 1 :]
        loo_model = _fit_ridge(training_features, training_prices)
        loo_predictions.append(_predict(loo_model, features[omitted_index]))

    loo_mae = sum(
        abs(price - prediction)
        for price, prediction in zip(prices, loo_predictions)
    ) / len(prices)
    flag_band = round(loo_mae, 1)

    scored_listings = []
    for listing, prediction in zip(listings, loo_predictions):
        fair_price = max(1, round(prediction))
        gap = round(float(listing["price"]) - fair_price)
        if gap < -flag_band:
            flag = "Under-priced"
        elif gap > flag_band:
            flag = "Over-priced"
        else:
            flag = "Fair"

        scored_listing = dict(listing)
        scored_listing.update(
            fair_price=fair_price,
            gap=gap,
            flag=flag,
            gap_pct=round(gap / fair_price * 100, 1),
        )
        scored_listings.append(scored_listing)

    model = _fit_ridge(features, prices)
    means, scales, price_mean, scaled_coefficients = model
    coefficients = [
        coefficient / scale
        for coefficient, scale in zip(scaled_coefficients, scales)
    ]
    intercept = price_mean - sum(
        coefficient * mean for coefficient, mean in zip(coefficients, means)
    )

    result = dict(source_data)
    result["listings"] = scored_listings
    result["meta"] = dict(source_data["meta"])
    result["meta"].update(
        best_model="Ridge, scaled (reviews, score, distance)",
        n_listings=len(listings),
        loo_mae_bwp=flag_band,
        flag_band_bwp=flag_band,
        regression={
            "features": list(FEATURES),
            "alpha": RIDGE_ALPHA,
            "intercept": round(intercept, 4),
            "coefficients": {
                name: round(coefficient, 4)
                for name, coefficient in zip(FEATURES, coefficients)
            },
        },
    )
    return result, model

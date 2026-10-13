"""Location features and the Member 1 popularity model applied to a proposed restaurant.

`engineer_location_features` in notebook 01 builds features for restaurants already in the
catalog. This module rebuilds the same feature definitions for a *hypothetical* restaurant at
any point, so the trained regression pipeline can give a what-if estimate. All competitor and
activity features use the saved Philadelphia restaurant catalog from the model bundle.
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0088
CITY_HALL = (39.9526, -75.1636)
BOOLEAN_ATTRIBUTES = ['RestaurantsTakeOut', 'RestaurantsDelivery', 'OutdoorSeating', 'RestaurantsReservations',
                      'RestaurantsGoodForGroups', 'GoodForKids', 'BusinessAcceptsCreditCards', 'HasTV',
                      'RestaurantsTableService', 'Caters', 'BikeParking', 'WheelchairAccessible', 'HappyHour']


class PopularityModel:
    """Wraps the saved Member 1 bundle (`outputs/member1/member1_model.joblib`)."""

    def __init__(self, bundle):
        self.bundle = bundle
        self.model = bundle['model']
        self.columns = list(bundle['feature_columns'])
        self.tags = list(bundle['category_tags'])
        catalog = bundle['restaurant_catalog'].reset_index(drop=True)
        self.catalog = catalog
        self.tree = BallTree(np.radians(catalog[['latitude', 'longitude']].to_numpy(float)), metric='haversine')
        self.is_open = catalog['is_open'].to_numpy(float)
        self.activity = np.log1p(catalog['checkin_count'].to_numpy(float)) if 'checkin_count' in catalog else None
        attr_counts = catalog['attributes'].map(lambda d: len(d) if isinstance(d, dict) else 0)
        self.typical_attribute_count = int(attr_counts[catalog['is_open'] == 1].median())

    def location_features(self, lat, lon):
        """Competition and activity around a point (the focal business is new, so nothing is excluded)."""
        point = np.radians([[lat, lon]])
        out = {}
        for radius in [.5, 1., 2.]:
            idx = self.tree.query_radius(point, r=radius / EARTH_RADIUS_KM)[0]
            suffix = str(radius).replace('.', 'p')
            out[f'competitors_{suffix}km'] = len(idx)
            if radius == 1.:
                out['open_competitors_1km'] = int(np.nansum(self.is_open[idx]))
            if self.activity is not None and radius in (.5, 1.):
                out[f'nearby_checkin_mean_{suffix}km'] = float(self.activity[idx].mean()) if len(idx) else np.nan
        out['competitor_density_1km'] = out['competitors_1p0km'] / np.pi
        lat1, lon1, lat2, lon2 = np.radians([lat, lon, *CITY_HALL])
        a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        out['distance_city_hall_km'] = float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))
        return out

    def feature_row(self, lat, lon, stars, price_range, categories, years_on_yelp, attributes=None,
                    alcohol=None, noise=None, wifi=None, any_parking=None, days_open=7, hours_per_week=70,
                    late_night_days=0, attribute_count=None):
        attributes = attributes or {}
        row = {'latitude': lat, 'longitude': lon, 'stars': stars, 'is_open': 1.0, 'price_range': price_range,
               'category_count': len(set(categories) | {'Restaurants'}),
               'years_since_first_checkin': years_on_yelp, 'checkin_active_years': years_on_yelp,
               'attribute_count': attribute_count if attribute_count is not None else self.typical_attribute_count,
               'level_Alcohol': alcohol, 'level_NoiseLevel': noise, 'level_WiFi': wifi, 'any_parking': any_parking,
               'days_open': days_open, 'hours_per_week': hours_per_week, 'late_night_days': late_night_days}
        row.update(self.location_features(lat, lon))
        for name in BOOLEAN_ATTRIBUTES:
            value = attributes.get(name)
            row[f'attr_{name}'] = np.nan if value is None else float(bool(value))
        for tag in self.tags:
            row[f'tag__{tag}'] = int(tag in categories)
        # Columns the model does not use are dropped; missing ones become NaN and are imputed by the pipeline.
        return pd.DataFrame([{c: row.get(c, np.nan) for c in self.columns}], dtype=float)

    def estimate(self, years=(1, 2, 3, 5), **kwargs):
        """Estimated cumulative Yelp reviews after each number of years (a popularity proxy, not revenue).

        When the bundle has quantile models, `low` / `high` give the 10th-90th percentile range.
        """
        rows = []
        for y in years:
            features = self.feature_row(years_on_yelp=float(y), **kwargs)
            row = {'years_on_yelp': y, 'estimated_reviews': float(max(self.model.predict(features)[0], 0))}
            if 'model_q10' in self.bundle:
                # Conformally calibrated 80% interval: widen the quantile range on the log scale (see notebook M1-7b).
                adj = float(self.bundle.get('interval_log_adjustment', 0.0))
                low = max(self.bundle['model_q10'].predict(features)[0], 0)
                high = max(self.bundle['model_q90'].predict(features)[0], low)
                row['low'] = float(max(np.expm1(np.log1p(low) - adj), 0))
                row['high'] = float(np.expm1(np.log1p(high) + adj))
            rows.append(row)
        return pd.DataFrame(rows)

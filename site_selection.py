import os
import json
import re
import argparse
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from pyproj import Transformer
import matplotlib.pyplot as plt

EARTH_R_KM = 6371.0

# Utils

def haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1); lat1 = np.radians(lat1)
    lon2 = np.radians(lon2); lat2 = np.radians(lat2)
    dlon = lon2 - lon1; dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return EARTH_R_KM * 2.0 * np.arcsin(np.sqrt(a))

def normalize(s):
    s = pd.to_numeric(s, errors='coerce')
    mn, mx = s.min(), s.max()
    return (s - mn)/(mx - mn) if (pd.notna(mn) and pd.notna(mx) and mx>mn) else pd.Series(np.zeros(len(s)), index=s.index)

def _weights_or_equal(n, cfg_list):
    """Return normalized weights from cfg or equal weights if none"""
    if isinstance(cfg_list, (list, tuple)) and len(cfg_list) == n:
        w = np.array(cfg_list, dtype=float)
    else:
        w = np.ones(n, dtype=float)
    s = w.sum()
    return (w / s) if s > 0 else np.ones(n, dtype=float) / float(n)

def to_float_safe(x):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        xs = x.replace(' ','').replace(' ','').replace(',', '.')
        try:
            return float(xs)
        except Exception:
            return np.nan
    return np.nan

def _normalize_name(s: str) -> str:
    if s is None:
        return ""
    s = str(s).casefold() 
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^\w\s]", "", s)
    return s

def cells_to_geojson(df, id_col=None):
    ddeg = 1.0 / 120.0   # 30 arcsecs
    half = ddeg / 2.0
    feats = []
    for _, r in df.iterrows():
        lon = float(r['lon']); lat = float(r['lat'])
        cell_id = str(r[id_col]) if (id_col and id_col in df.columns) else f"{lon:.6f}_{lat:.6f}" #for pbi visuals linking

        ring = [
            [lon - half, lat - half],
            [lon + half, lat - half],
            [lon + half, lat + half],
            [lon - half, lat + half],
            [lon - half, lat - half]
        ]
        feats.append({
            "type": "Feature",
            "properties": {"cell_id": cell_id},
            "geometry": {"type": "Polygon", "coordinates": [ring]}
        })
    return {"type": "FeatureCollection", "features": feats}

def aggregate_poi_by_radius_m(centers_lat, centers_lon, poi_df, t_ll_to_utm, value_col=None, radii_m=(100,), mode='sum'):
    n = len(centers_lat)
    res = {}
    if poi_df is None or len(poi_df)==0:
        for r in radii_m:
            res[f"m_{int(r)}"] = np.zeros(n)
        return res
    px, py = t_ll_to_utm.transform(poi_df['lon'].to_numpy(), poi_df['lat'].to_numpy())
    if value_col and value_col in poi_df.columns:
        pv = pd.to_numeric(poi_df[value_col], errors='coerce').to_numpy().astype(float)
        pv[np.isnan(pv)] = 0.0
    else:
        pv = np.ones(len(poi_df), dtype=float)
    cx, cy = t_ll_to_utm.transform(centers_lon, centers_lat)
    for r in radii_m:
        R2 = (float(r) ** 2)
        vals = np.zeros(n, dtype=float)
        for ci in range(n):
            dx = px - cx[ci]; dy = py - cy[ci]
            inside = (dx*dx + dy*dy) <= R2
            vals[ci] = float(np.sum(pv[inside])) if mode=='sum' else float(np.sum(inside))
        res[f"m_{int(r)}"] = vals
    return res

# Load data

def load_users(path):
    df = pd.read_csv(path, sep=';')
    df.columns = [c.strip().lower().replace(' ','_') for c in df.columns]
    metric_col = next((c for c in df.columns if 'metric' in c), None)
    if metric_col is None:
        raise ValueError('Users: metric column not found')
    df = df.rename(columns={metric_col:'user_metric'})
    return df.dropna(subset=['lat','lon','user_metric'])

def load_stores(path):
    df = pd.read_csv(path, sep=';')
    df = df[[c for c in df.columns if str(c).strip()!='']]
    df.columns = [c.strip().lower().replace(' ','_') for c in df.columns]
    metric_col = next((c for c in df.columns if 'metric' in c), None)
    df = df.rename(columns={metric_col:'store_metric','long':'lon'})
    return df.dropna(subset=['lat','lon','store_metric'])

def load_kyiv_polygon(geojson_path):
    with open(geojson_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    geoms = []
    feats = data.get('features') or []
    for feat in feats:
        geom = feat.get('geometry')
        if geom and geom.get('type') in ('Polygon','MultiPolygon'):
            coords = geom.get('coordinates')
            if geom['type']=='Polygon':
                geoms.append(Polygon(coords[0]))
            else:
                geoms.append(MultiPolygon([Polygon(poly[0]) for poly in coords]))
    if not geoms:
        raise ValueError('No Kyiv polygon found')
    return unary_union(geoms)

def load_population_in_kyiv(csv_path, kyiv_poly):
    chunks = []                                     #idk maybe no need for chunking? pretty small dataset
    for ch in pd.read_csv(csv_path, names=['lon','lat','pop_density'], header=0, chunksize=500000):
        minx, miny, maxx, maxy = kyiv_poly.bounds
        mask = (ch['lon']>=minx) & (ch['lon']<=maxx) & (ch['lat']>=miny) & (ch['lat']<=maxy)
        ch2 = ch.loc[mask].copy()
        if ch2.empty:
            continue
        pts = [Point(xy) for xy in zip(ch2['lon'], ch2['lat'])]
        inside = [kyiv_poly.contains(pt) for pt in pts]
        sub = ch2.loc[inside]
        if not sub.empty:
            chunks.append(sub)
    if not chunks:
        return pd.DataFrame(columns=['lon','lat','pop_density'])
    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates(subset=['lon','lat'])
    return df

# Points Of Interest

def load_parking_arcgis(path_or_url):
    data = None
    if isinstance(path_or_url, str) and path_or_url.startswith('http') and requests is not None:
        try:
            r = requests.get(path_or_url, timeout=60)
            r.raise_for_status()
            data = r.json()
        except Exception:
            data = None
    if data is None:
        with open(path_or_url, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception:
                return pd.DataFrame(columns=['lon','lat','cars'])
    feats = data.get('features') if isinstance(data, dict) else (data if isinstance(data, list) else [])
    rows = []
    for f in feats:
        attrs = f.get('attributes', {}) or {}
        cars = to_float_safe(attrs.get('cars'))
        geom = f.get('geometry', {}) or {}
        rings = geom.get('rings')
        coords = geom.get('coordinates')
        poly = None
        try:
            if isinstance(rings, list) and len(rings)>0:
                poly = Polygon(rings[0])
            elif isinstance(coords, list) and len(coords)>0:
                poly = Polygon(coords[0])
        except Exception:
            poly = None
        if poly is None:
            continue
        try:
            cen = poly.centroid
            rows.append({'lon': float(cen.x), 'lat': float(cen.y), 'cars': float(cars) if not np.isnan(cars) else 0.0})
        except Exception:
            continue
    return pd.DataFrame(rows)

def load_transit_aboveground(path_or_url, mapping=None):
    """Load simple list/dict JSON for above-ground stops. Type is set to 'other'."""
    data = None
    if isinstance(path_or_url, str) and path_or_url.startswith('http') and requests is not None:
        try:
            r = requests.get(path_or_url, timeout=60)
            r.raise_for_status()
            data = r.json()
        except Exception:
            data = None
    if data is None:
        with open(path_or_url, 'r', encoding='utf-8') as f:
            data = json.load(f)
    it = []
    if isinstance(data, list):
        it = data
    elif isinstance(data, dict):
        it = data.get('data') or data.get('features') or data.get('records') or data.get('items') or []
    rows = []
    for rec in it:
        lat = lon = None
        if isinstance(rec, dict):
            if mapping:
                lat = rec.get(mapping.get('lat')); lon = rec.get(mapping.get('lon'))
            lat = lat if lat is not None else rec.get('lat') or rec.get('Latitude') or rec.get('Y') or rec.get('y')
            lon = lon if lon is not None else rec.get('lon') or rec.get('Longitude') or rec.get('X') or rec.get('x')
            geom = rec.get('geometry') or rec.get('geo') or rec.get('location')
            if (lat is None or lon is None) and isinstance(geom, dict):
                coords = geom.get('coordinates')
                if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                    lon = lon or coords[0]; lat = lat or coords[1]
        if lat is not None and lon is not None:
            rows.append({'lon': float(lon), 'lat': float(lat), 'type': 'other'})
    return pd.DataFrame(rows)

def load_overpass_supermarkets(src):
    data = None
    if isinstance(src, str):
        if src.lower().startswith(("http")) and requests is not None:
            r = requests.get(src, timeout=60)
            r.raise_for_status()
            data = r.json()
        else:
            with open(src, "r", encoding="utf-8") as f:
                data = json.load(f)
    else:
        raise TypeError("src must be URL string or local path string")
    features = data.get("features") or []
    rows = []

    for ft in features:
        if not isinstance(ft, dict):
            continue

        props = ft.get("properties") or {}
        geom  = ft.get("geometry") or {}
        gtype = (geom.get("type") or "").lower()
        coords = geom.get("coordinates")

        name = props.get("name")                # to filter our existing stores

        lon, lat = None, None

        try:
            if gtype == "point":
                if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                    lon, lat = float(coords[0]), float(coords[1])

            elif gtype == "polygon":
                if isinstance(coords, list) and len(coords) > 0:
                    poly = Polygon(coords[0])
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    c = poly.centroid
                    lon, lat = float(c.x), float(c.y)

            elif gtype == "multipolygon":
                if isinstance(coords, list) and len(coords) > 0:
                    polys = []
                    for p in coords:
                        if isinstance(p, list) and len(p) > 0:
                            try:
                                pp = Polygon(p[0])
                                if not pp.is_valid:
                                    pp = pp.buffer(0)
                                polys.append(pp)
                            except Exception:
                                continue
                    if polys:
                        mpoly = MultiPolygon(polys)
                        if not mpoly.is_valid:
                            mpoly = mpoly.buffer(0)
                        c = mpoly.centroid
                        lon, lat = float(c.x), float(c.y)

        except Exception: #bad geometry
            raise(ValueError, f"Bad Geometry found on {name}, {coords}")
            lon, lat = None, None

        if lon is not None and lat is not None:
            rows.append({"lon": lon, "lat": lat, "name": name})

    if not rows:
        return pd.DataFrame(columns=["lon", "lat", "name"])

    return pd.DataFrame(rows)

def load_metro_exits_geojson(path_or_url):
    data = None
    if isinstance(path_or_url, str) and path_or_url.startswith('http') and requests is not None:
        try:
            r = requests.get(path_or_url, timeout=60)
            r.raise_for_status()
            data = r.json()
        except Exception:
            data = None
    if data is None:
        with open(path_or_url, 'r', encoding='utf-8') as f:
            data = json.load(f)
    if not isinstance(data, dict) or data.get('type') != 'FeatureCollection':
        return pd.DataFrame(columns=['lon','lat','type','name','exit_number','line_id'])
    feats = data.get('features', [])
    rows = []
    for ft in feats:
        geom = ft.get('geometry') or {}
        props = ft.get('properties') or {}
        coords = geom.get('coordinates') if isinstance(geom, dict) else None
        lon = lat = None
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            lon, lat = float(coords[0]), float(coords[1])
        line_id = props.get('line_id') or props.get('LINE_ID') or props.get('line') or props.get('Line') or props.get('linia_id')
        if (lon is not None) and (lat is not None):
            rows.append({'lon': lon, 'lat': lat, 'type': 'metro_exit', 'line_id': line_id})
    return pd.DataFrame(rows)

# Grid build

def grid_from_population_30arc(pop_df, kyiv_poly, t_ll_to_utm):
    ddeg = 1.0/120.0
    half = ddeg/2.0
    cells_utm, centers_lon, centers_lat, areas_km2, pop_vals = [], [], [], [], []
    for _, r in pop_df.iterrows():
        lon = float(r['lon']); lat = float(r['lat']); z = float(r['pop_density'])
        poly_ll = Polygon([(lon-half, lat-half), (lon+half, lat-half), (lon+half, lat+half), (lon-half, lat+half)])
        if not kyiv_poly.contains(Point(lon, lat)):
            continue
        xs, ys = [], []
        for (xll, yll) in poly_ll.exterior.coords:
            # IMPORTANT: lon,lat order with always_xy=True
            xutm, yutm = t_ll_to_utm.transform(xll, yll)
            xs.append(xutm); ys.append(yutm)
        poly_utm = Polygon(list(zip(xs, ys)))
        if not poly_utm.is_valid:
            poly_utm = poly_utm.buffer(0)
        area_km2 = poly_utm.area / 1e6
        cells_utm.append(poly_utm)
        centers_lon.append(lon)
        centers_lat.append(lat)
        areas_km2.append(area_km2)
        pop_vals.append(z)
    return cells_utm, np.array(centers_lon), np.array(centers_lat), np.array(areas_km2), np.array(pop_vals)

# Existing users per cell

def compute_user_features(users_df, cells_utm, centers_lon, centers_lat, t_ll_to_utm, lam_km=0.7, decay_radius_km=1.5):
    ux, uy = t_ll_to_utm.transform(users_df['lon'].to_numpy(), users_df['lat'].to_numpy())
    um = users_df['user_metric'].to_numpy()
    counts = np.zeros(len(cells_utm), dtype=int)
    sums   = np.zeros(len(cells_utm), dtype=float)

    bounds = np.array([c.bounds for c in cells_utm])

    for i in range(len(ux)):
        x, y, val = ux[i], uy[i], um[i]
        hits = np.where((x >= bounds[:,0]) & (x <= bounds[:,2]) & (y >= bounds[:,1]) & (y <= bounds[:,3]))[0]
        for ci in hits:
            if cells_utm[ci].covers(Point(x, y)):
                counts[ci] += 1
                sums[ci]   += val
                break
    means = np.where(counts>0, sums/np.maximum(counts,1), np.nan)

    u_np = users_df[['lat','lon','user_metric']].to_numpy()
    decay_vals = []
    for lonc, latc in zip(centers_lon, centers_lat):
        d_km = haversine_km(lonc, latc, u_np[:,1], u_np[:,0])
        mask = d_km <= decay_radius_km
        if not np.any(mask):
            decay_vals.append(np.nan)
        else:
            w = np.exp(-d_km[mask]/max(lam_km,1e-6))
            vals = u_np[:,2][mask]
            decay_vals.append(float(np.sum(w*vals)/np.sum(w)))
    return counts, means, np.array(decay_vals)

# Competitors

def competitor_decay(centers_lon, centers_lat, comp_df, lam_km=0.7):
    if comp_df is None or len(comp_df)==0:
        return np.zeros(len(centers_lon))
    c_lon = centers_lon; c_lat = centers_lat
    p_lon = comp_df['lon'].to_numpy(); p_lat = comp_df['lat'].to_numpy()
    dec = np.zeros(len(c_lon), dtype=float)
    for i in range(len(c_lon)):
        d = haversine_km(c_lon[i], c_lat[i], p_lon, p_lat)
        w = np.exp(-d/ max(lam_km,1e-6))
        dec[i] = float(np.sum(w))
    return dec

# Calibrating K

def choose_k_by_correlation(centers_lon, centers_lat, users_counts, users_decay_norm, pop_norm, stores_df, k_candidates):
    """Finding optimal K for the formula """
    g_lat = np.array(centers_lat); g_lon = np.array(centers_lon)
    s_lat = stores_df['lat'].to_numpy(); s_lon = stores_df['lon'].to_numpy(); s_metric = stores_df['store_metric'].to_numpy()
    nearest_idx = []
    for i in range(len(s_lat)):
        d2 = (g_lat - s_lat[i])**2 + (g_lon - s_lon[i])**2
        nearest_idx.append(int(np.argmin(d2)))
    best_k, best_r = None, -999.0
    for k in k_candidates:
        alpha = users_counts / (users_counts + k)
        blended = alpha * users_decay_norm + (1.0 - alpha) * pop_norm
        vals = blended[nearest_idx]
        r = pd.Series(vals).corr(pd.Series(s_metric))
        if pd.notnull(r) and r > best_r:
            best_r, best_k = r, k
    return best_k, best_r

# ------------------------- Main -------------------------

def run(cfg):
    # WGS84 to UTM 36N (for distance calc)
    t_ll_to_utm = Transformer.from_crs('EPSG:4326', 'EPSG:32636', always_xy=True)

    users = load_users(cfg['users_path'])
    stores = load_stores(cfg['stores_path'])
    kyiv_poly = load_kyiv_polygon(cfg['kyiv_geojson_path'])

    pop = load_population_in_kyiv(cfg['population_csv_path'], kyiv_poly)
    if pop.empty:
        raise ValueError('No population points found inside Kyiv polygon; check CSV path and CRS.')
    cells_utm, centers_lon, centers_lat, areas_km2, pop_density = grid_from_population_30arc(pop, kyiv_poly, t_ll_to_utm)
    pop_norm = normalize(pd.Series(pop_density))
    pop_count_est = pop_density * areas_km2

    counts, means, decay_vals = compute_user_features(
        users, cells_utm, centers_lon, centers_lat,
        t_ll_to_utm,
        lam_km=cfg.get('decay_lambda_km',0.7),
        decay_radius_km=cfg.get('decay_radius_km',1.5)
    )
    users_decay_norm = normalize(pd.Series(decay_vals))

    k_candidates = cfg.get('k_candidates', list(range(5,101,5)))
    k_opt, r_best = choose_k_by_correlation(
        centers_lon, centers_lat,
        counts.astype(float), users_decay_norm.to_numpy(), pop_norm.to_numpy(), stores, k_candidates
    )
    k_used = int(k_opt) if k_opt is not None else int(np.median(counts))
    alpha = counts.astype(float) / (counts.astype(float) + k_used)
    demand_blended = alpha * users_decay_norm + (1.0 - alpha) * pd.Series(pop_norm)

    stops_above = load_transit_aboveground(cfg['stops_src_aboveground'], cfg.get('stops_mapping')) if cfg.get('stops_src_aboveground') else None
    stops_metro  = load_metro_exits_geojson(cfg['metro_exits_src']) if cfg.get('metro_exits_src') else None

    transit_points = []
    if stops_above is not None and len(stops_above)>0:
        transit_points.append(stops_above)
    if stops_metro is not None and len(stops_metro)>0:
        transit_points.append(stops_metro)
    stops_all = pd.concat(transit_points, ignore_index=True) if transit_points else pd.DataFrame(columns=['lon','lat','type','line_id'])

    wcfg = cfg.get('transit_type_weights', {'metro_exit': 4.0, 'other': 1.0})
    def _w(t):
        if pd.isna(t):
            return float(wcfg.get('other',1.0))
        t = str(t).lower()
        if 'metro_exit' in t:
            return float(wcfg.get('metro_exit', 4.0))
        return float(wcfg.get('other', 1.0))
    if len(stops_all)>0:
        stops_all['weight'] = stops_all['type'].apply(_w)

    parking_df = load_parking_arcgis(cfg['parking_src']) if cfg.get('parking_src') else None

    comps_df = None
    if cfg.get('competitors_src'):
        try:
            comps_df = load_overpass_supermarkets(cfg['competitors_src'])
        except Exception as e:
            print(e)
            comps_df = None

    ex_names    = [ _normalize_name(x) for x in cfg.get("brand_exclusion_names", []) ]
    ex_radius_m = int(cfg.get("brand_exclusion_radius_m", 500))

    if comps_df is not None and len(comps_df) > 0:
        if "name" in comps_df.columns:
            comps_df["name_norm"] = comps_df["name"].apply(_normalize_name)
        else:
            comps_df["name_norm"] = ""

        comps_ex = comps_df.loc[ comps_df["name_norm"].isin(ex_names), ["lon", "lat", "name"] ].copy()
    else:
        comps_ex = None

    r_transit_m = cfg.get('radii_m_transit', [100, 250, 500])
    r_parking_m = cfg.get('radii_m_parking', [100, 200, 300])
    r_comp_m    = cfg.get('radii_m_competitors', [500, 1000, 1500])
    r_store_m   = cfg.get('radii_m_stores', [500, 1000, 1500])

    transit_aggr = aggregate_poi_by_radius_m(centers_lat, centers_lon, stops_all, t_ll_to_utm, value_col='weight', radii_m=r_transit_m, mode='sum') if len(stops_all)>0 else None
    parking_aggr = aggregate_poi_by_radius_m(centers_lat, centers_lon, parking_df, t_ll_to_utm, value_col='cars', radii_m=r_parking_m, mode='sum') if (parking_df is not None and len(parking_df)>0) else None
    comp_aggr    = aggregate_poi_by_radius_m(centers_lat, centers_lon, comps_df, t_ll_to_utm, value_col=None, radii_m=r_comp_m, mode='count') if (comps_df is not None and len(comps_df)>0) else None

    store_aggr   = aggregate_poi_by_radius_m(centers_lat, centers_lon, stores[['lon','lat']], t_ll_to_utm, value_col=None, radii_m=r_store_m, mode='count')

    s_np = stores[['lat','lon']].to_numpy()
    dist_store = []
    for lonc, latc in zip(centers_lon, centers_lat):
        d = haversine_km(lonc, latc, s_np[:,1], s_np[:,0])
        dist_store.append(float(np.min(d)) if len(d) else np.nan)
    whitespace_norm = normalize(pd.Series(dist_store))

    if comp_aggr is not None:
        first_key = f"m_{int(r_comp_m[0])}"
        competition_score = 1.0 - normalize(pd.Series(comp_aggr[first_key]))
    else:
        competition_score = pd.Series(np.ones(len(centers_lon)))

    if comps_ex is not None and len(comps_ex) > 0:
        brand_counts = aggregate_poi_by_radius_m(
            centers_lat, centers_lon,
            comps_ex, t_ll_to_utm,
            value_col=None,
            radii_m=[ex_radius_m],
            mode='count'
        )
        brand_key        = f"m_{ex_radius_m}"
        brand_count_arr  = brand_counts.get(brand_key, np.zeros(len(centers_lat), dtype=float))
        brand_flag_arr   = (brand_count_arr > 0).astype(int)
    else:
        brand_count_arr = np.zeros(len(centers_lat), dtype=float)
        brand_flag_arr  = np.zeros(len(centers_lat), dtype=int)

    records = {
        'lat': centers_lat,
        'lon': centers_lon,
        'cell_area_km2': areas_km2,
        'population_density': pop_density,
        'population_density_norm': pop_norm,
        'population_est_count': pop_count_est,
        'users_cell_count': counts,
        'users_cell_mean': means,
        'users_decay1km': decay_vals,
        'users_decay1km_norm': users_decay_norm,
        'demand_blended': demand_blended,
        'dist_to_store_km': dist_store,
        'whitespace_norm': whitespace_norm,
        'k_used': k_used,
        'competitors_decay': competitor_decay(centers_lon, centers_lat, comps_df, lam_km=cfg.get('comp_decay_lambda_km',0.7)) if (comps_df is not None and len(comps_df)>0) else np.zeros(len(centers_lon))
    }

    transit_weighted_norm = None
    if transit_aggr is not None:
        w_transit = _weights_or_equal(len(r_transit_m), cfg.get("weights_radii_transit"))
        transit_cols = []
        for i, r in enumerate(r_transit_m):
            key = f"transit_weighted_m_{int(r)}"
            records[key] = transit_aggr[f"m_{int(r)}"]
            transit_cols.append(key)
            
        per_radius_norm = []
        for key in transit_cols:
            per_radius_norm.append(normalize(pd.Series(records[key])).to_numpy())

        transit_matrix = np.vstack(per_radius_norm).T  # n_cells, n_radii
        transit_weighted_norm = pd.Series(transit_matrix.dot(w_transit))
        records["transit_weighted_norm"] = transit_weighted_norm

    parking_capacity_norm = None
    if parking_aggr is not None:
        w_parking = _weights_or_equal(len(r_parking_m), cfg.get("weights_radii_parking"))
        parking_cols = []
        for i, r in enumerate(r_parking_m):
            key = f"parking_capacity_m_{int(r)}"
            records[key] = parking_aggr[f"m_{int(r)}"]
            parking_cols.append(key)

        per_radius_norm = []
        for key in parking_cols:
            per_radius_norm.append(normalize(pd.Series(records[key])).to_numpy())

        parking_matrix = np.vstack(per_radius_norm).T
        parking_capacity_norm = pd.Series(parking_matrix.dot(w_parking))
        records["parking_capacity_norm"] = parking_capacity_norm

    competition_score = None
    if comp_aggr is not None:
        w_comp = _weights_or_equal(len(r_comp_m), cfg.get("weights_radii_competitors"))
        comp_cols = []
        for i, r in enumerate(r_comp_m):
            key = f"competitor_count_m_{int(r)}"
            records[key] = comp_aggr[f"m_{int(r)}"]
            comp_cols.append(key)

        per_radius_norm = []
        for key in comp_cols:
            per_radius_norm.append(normalize(pd.Series(records[key])).to_numpy())

        comp_matrix = np.vstack(per_radius_norm).T
        comp_norm_blend = pd.Series(comp_matrix.dot(w_comp))

        competition_score = 1.0 - normalize(comp_norm_blend)
        records["competition_score"] = competition_score
    else:
        records["competition_score"] = pd.Series(np.ones(len(centers_lon)))

    for r in r_store_m:
        key = f"store_count_m_{int(r)}"
        records[key] = store_aggr[f"m_{int(r)}"]

    records[f"brand_exclusion_count_m_{ex_radius_m}"]  = brand_count_arr
    records[f"brand_exclusion_within_m_{ex_radius_m}"] = brand_flag_arr  # 1/0

    out = pd.DataFrame(records)

    # Default score 
    w = cfg.get('default_weights', {
        'demand_blended': 0.50,
        'parking_capacity_norm': 0.05,
        'transit_weighted_norm': 0.2,
        'competition_score': 0.2,
        'whitespace_norm': 0.05
    })
    out['default_score'] = (
        out['demand_blended'] * w.get('demand_blended',0.0) +
        out.get('parking_capacity_norm', pd.Series(np.zeros(len(out)))) * w.get('parking_capacity_norm',0.0) +
        out.get('transit_weighted_norm', pd.Series(np.zeros(len(out)))) * w.get('transit_weighted_norm',0.0) +
        out['competition_score'] * w.get('competition_score',0.0) +
        out['whitespace_norm'] * w.get('whitespace_norm',0.0)
    )

    out["cell_id"] = out.apply(lambda r: f"{r['lon']:.6f}_{r['lat']:.6f}", axis=1)

    out.to_csv('candidate_features.csv', index=False)

    col_flag = f"brand_exclusion_within_m_{ex_radius_m}"
    filtered = out.loc[out[col_flag] == 0].copy()
    filtered.to_csv('candidate_features_no_brand.csv', index=False)

    topN = cfg.get('top_n', 15)
    filtered.sort_values('default_score', ascending=False).head(topN)\
            .to_csv('proposed_store_locations_filtered.csv', index=False)


    topN = cfg.get('top_n', 15)
    out.sort_values('default_score', ascending=False).head(topN).to_csv('proposed_store_locations.csv', index=False)

    if len(stops_all)>0:
        stops_all[['lat','lon','type','line_id','weight']].to_csv('transit_points.csv', index=False)
    
    geojson = cells_to_geojson(filtered, id_col="cell_id" if "cell_id" in out.columns else None)
    
    with open("candidate_cells.geojson", "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False)

    plt.figure(figsize=(8,8))
    plt.scatter(out['lon'], out['lat'], s=6, alpha=0.3, label='Grid centers (30 arcsec)')
    plt.scatter(stores['lon'], stores['lat'], s=30, label='Stores')
    top = out.sort_values('default_score', ascending=False).head(topN)
    plt.scatter(top['lon'], top['lat'], s=60, marker='X', color='red', label='Proposed')
    plt.legend(); plt.title('Kyiv 30" grid centers, stores, proposed')
    plt.xlabel('Longitude'); plt.ylabel('Latitude'); plt.tight_layout()
    plt.savefig('map_grid_proposed_30arc.png', dpi=150)

    print('Done: candidate_features.csv, proposed_store_locations.csv, transit_points.csv, map_grid_proposed_30arc.png')

if __name__ == '__main__':
    import os
    ap = argparse.ArgumentParser()
    default_cfg = os.path.join(os.path.dirname(__file__), 'cfg.json')
    ap.add_argument('--config', default=default_cfg)
    args = ap.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    run(cfg)
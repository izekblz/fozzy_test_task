
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from site_selection import haversine_km

stores_raw = pd.read_csv('./data/original/stores.csv', sep=';', header=0)
users_raw = pd.read_csv('./data/original/Users.csv', sep=';', header=0)

stores = stores_raw.copy()
stores = stores[[col for col in stores.columns if col.strip() != '']]
stores.columns = [c.strip().lower().replace(' ', '_') for c in stores.columns]
metric_cols = [c for c in stores.columns if 'metric' in c]
store_metric_col = metric_cols[0]
stores = stores.rename(columns={store_metric_col: 'store_metric', 'long': 'lon'})

users = users_raw.copy()
users.columns = [c.strip().lower().replace(' ', '_') for c in users.columns]
user_metric_col = None
for c in users.columns:
    if c.startswith('metric') or 'metric' in c:
        user_metric_col = c
        break
if user_metric_col is None:
    raise ValueError(f"No metric column found in users: {users.columns}")
users = users.rename(columns={user_metric_col: 'user_metric'})

for df, kind in [(stores, 'stores'), (users, 'users')]:
    for col in ['lat', 'lon']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    metric_name = 'store_metric' if kind=='stores' else 'user_metric'
    df[metric_name] = pd.to_numeric(df[metric_name], errors='coerce')
    df.dropna(subset=['lat','lon', metric_name], inplace=True)

stores_coords = stores[['lat','lon']].to_numpy()
store_metrics = stores['store_metric'].to_numpy()

batch_size = 5000           #ram shortages are wild these days
nearest_store_metric = []
nearest_store_dist_km = []

for start in range(0, len(users), batch_size):
    end = min(start+batch_size, len(users))
    u = users.iloc[start:end]
    dists = haversine_km(u['lat'].to_numpy()[:,None], u['lon'].to_numpy()[:,None],
                      stores_coords[:,0][None,:], stores_coords[:,1][None,:])
    idx_min = np.argmin(dists, axis=1)
    nearest_store_metric.append(store_metrics[idx_min])
    nearest_store_dist_km.append(dists[np.arange(len(u)), idx_min])

users = users.copy()
users['nearest_store_metric'] = np.concatenate(nearest_store_metric)
users['nearest_store_dist_km'] = np.concatenate(nearest_store_dist_km)

corr_user_store = users['user_metric'].corr(users['nearest_store_metric'])

RADIUS_KM_CLOSE = 0.5
RADIUS_KM_FAR = 1.5

store_local_mean_far = []
store_local_sum_far = []
store_local_count_far = []
store_local_mean_close = []
store_local_sum_close = []
store_local_count_close = []

users_coords = users[['lat','lon']].to_numpy()
users_metrics = users['user_metric'].to_numpy()

for i in range(len(stores)):
    s_lat, s_lon = stores_coords[i]
    dists = haversine_km(s_lat, s_lon, users_coords[:,0], users_coords[:,1])
    mask_far = dists <= RADIUS_KM_FAR
    mask_close = dists <= RADIUS_KM_CLOSE
    if np.any(mask_far):
        store_local_mean_far.append(users_metrics[mask_far].mean())
        store_local_sum_far.append(users_metrics[mask_far].sum())
        store_local_count_far.append(mask_far.sum())
    else:
        store_local_mean_far.append(np.nan)
        store_local_sum_far.append(0.0)
        store_local_count_far.append(0)
    if np.any(mask_close):
        store_local_mean_close.append(users_metrics[mask_close].mean())
        store_local_sum_close.append(users_metrics[mask_close].sum())
        store_local_count_close.append(mask_close.sum())
    else:
        store_local_mean_close.append(np.nan)
        store_local_sum_close.append(0.0)
        store_local_count_close.append(0)

stores['local_user_mean_close'] = store_local_mean_close
stores['local_user_sum_close'] = store_local_sum_close
stores['local_user_count_close'] = store_local_count_close
stores['local_user_mean_far'] = store_local_mean_far
stores['local_user_sum_far'] = store_local_sum_far
stores['local_user_count_far'] = store_local_count_far

stores_corr_far = stores[['store_metric','local_user_mean_far']].dropna()
corr_store_local_far = stores_corr_far['store_metric'].corr(stores_corr_far['local_user_mean_far'])
stores_corr_close = stores[['store_metric','local_user_mean_close']].dropna()
corr_store_local_close = stores_corr_close['store_metric'].corr(stores_corr_close['local_user_mean_close'])

# Visuals

plt.style.use('ggplot')

fig1, ax1 = plt.subplots(figsize=(10,8))     #user metric vs nearest store metric
ax1.scatter(users['nearest_store_metric'], users['user_metric'], s=8, alpha=0.4)
ax1.set_title('Кореляція: метрика користувача vs метрика найближчого магазину\n(Pearson r = %.3f)' % corr_user_store)
ax1.set_xlabel('Метрика найближчого магазину')
ax1.set_ylabel('Метрика користувача')
fig1.tight_layout()
fig1.savefig('corr_user_vs_nearest_store.png', dpi=150)

fig2, ax2 = plt.subplots(figsize=(10,8))     #store metric vs local mean user metric within RADIUS_KM disctance
valid = ~stores['local_user_mean_close'].isna()
ax2.scatter(stores.loc[valid, 'local_user_mean_close'], stores.loc[valid, 'store_metric'], s=30, alpha=0.6)
ax2.set_title('Кореляція: середня локальна метрика користувачів (%.1f км) vs метрика магазину\n(Pearson r = %.3f)' % (RADIUS_KM_CLOSE, corr_store_local_close))
ax2.set_xlabel('Середня локальна метрика користувачів')
ax2.set_ylabel('Метрика магазину')
fig2.tight_layout()
fig2.savefig('corr_store_vs_local_users_close.png', dpi=150)

fig3, ax3 = plt.subplots(figsize=(10,8))     #store metric vs local mean user metric within RADIUS_KM disctance
valid = ~stores['local_user_mean_far'].isna()
ax3.scatter(stores.loc[valid, 'local_user_mean_far'], stores.loc[valid, 'store_metric'], s=30, alpha=0.6)
ax3.set_title('Кореляція: середня локальна метрика користувачів (%.1f км) vs метрика магазину\n(Pearson r = %.3f)' % (RADIUS_KM_FAR, corr_store_local_far))
ax3.set_xlabel('Середня локальна метрика користувачів')
ax3.set_ylabel('Метрика магазину')
fig3.tight_layout()
fig3.savefig('corr_store_vs_local_users_far.png', dpi=150)

summary = {
    'num_stores': len(stores),
    'num_users': len(users),
    'corr_user_nearest_store': float(corr_user_store) if pd.notnull(corr_user_store) else None,
    'corr_store_local_users_mean_%skm' % RADIUS_KM_CLOSE: float(corr_store_local_close) if pd.notnull(corr_store_local_close) else None,
    'corr_store_local_users_mean_%skm' % RADIUS_KM_FAR: float(corr_store_local_far) if pd.notnull(corr_store_local_far) else None,
}
print(summary)
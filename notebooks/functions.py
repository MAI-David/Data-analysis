import pandas as pd
import numpy as np
import time
import re
import random
import os
from collections import namedtuple

# Machine Learning & Stats
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import layers, models, constraints, callbacks, initializers, regularizers
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold, cross_validate
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# =========================================================
# 1. GLOBAL CONFIG & STRUCTURES
# =========================================================
# Added 'top_features' to the tuple for easy inspection
ModelResult = namedtuple('ModelResult', ['model', 'rmse', 'r2', 'best_params', 'runtime', 'top_features'])
NNResult = namedtuple('NNResult', ['X_train_elite', 'X_test_elite', 'feature_names', 'rmse', 'n_features'])
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

def set_global_seeds(seed=42):
    """Ensure reproducibility across TF, Numpy, and Python"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"✅ Global seeds set to {seed}")

# =========================================================
# 2. NEURAL NETWORK SELECTOR (The "Gatekeeper")
# =========================================================
class GatekeeperLayer(layers.Layer):
    def __init__(self, num_features, l1_penalty=0.01, **kwargs):
        super(GatekeeperLayer, self).__init__(**kwargs)
        self.num_features = num_features
        self.l1_penalty = l1_penalty

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(self.num_features,),
            initializer=initializers.HeNormal(),
            trainable=True,
            constraint=constraints.NonNeg(),
            regularizer=regularizers.l1(self.l1_penalty)
        )

    def call(self, inputs):
        return inputs * self.w

def nn_feature_search(X_train, X_test, Y_train, target_range=(50, 1250)):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_tf = X_train_scaled.astype('float32')
    y_train_tf = Y_train.values.astype('float32')
    penalties = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    repeats = 7
    champion = {'rmse': float('inf'), 'weights': None, 'n_features': 0, 'penalty': 0}

    print(f"🔬 Starting NN Sparsity Search on {X_train.shape[1]} features...")
    print(f"{'Penalty':<8} | {'Run':<4} | {'Features':<10} | {'Val RMSE':<10} | {'Status'}")
    print("-" * 65)

    for p in penalties:
        for i in range(repeats):
            tf.keras.backend.clear_session()
            inputs = layers.Input(shape=(X_train_tf.shape[1],))
            gate = GatekeeperLayer(X_train_tf.shape[1], l1_penalty=p)(inputs)
            x = layers.Dense(32, activation='relu')(gate)
            outputs = layers.Dense(1, activation='linear')(x)
            model = models.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=tf.keras.optimizers.Adam(0.005), loss='mse')

            history = model.fit(X_train_tf, y_train_tf, epochs=200, batch_size=64,
                                validation_split=0.2, verbose=0,
                                callbacks=[callbacks.EarlyStopping(patience=20, restore_best_weights=True)])

            weights = model.layers[1].get_weights()[0]
            n_feats = np.sum(weights > 1e-3)
            val_rmse = min(history.history['val_loss']) ** 0.5

            status = "❌"
            if target_range[0] < n_feats < target_range[1]:
                status = "✅"
                if val_rmse < champion['rmse']:
                    champion.update({'rmse': val_rmse, 'weights': weights, 'n_features': n_feats, 'penalty': p})
                    status = "🏆 NEW BEST"
            print(f"{p:<8} | {i+1:<4} | {n_feats:<10} | {val_rmse:<10.2f} | {status}")

    if champion['weights'] is not None:
        df_imp = pd.DataFrame({'Bacteria': X_train.columns, 'Score': champion['weights']})
        elite_names = df_imp[df_imp['Score'] > 1e-3].sort_values('Score', ascending=False)['Bacteria'].tolist()
        return NNResult(X_train[elite_names], X_test[elite_names], elite_names, champion['rmse'], champion['n_features'])
    return None

# =========================================================
# 3. CLASSICAL ML BENCHMARKS (XGB & RF)
# =========================================================
def xgboost_benchmark(X_train_df, X_test_df, y_train, y_test, label="Dataset"):
    print(f"🚀 Initializing XGBoost Engine: {label}")
    X_train_clean = X_train_df.copy()
    X_test_clean = X_test_df.copy()
    X_train_clean.columns = [re.sub('[^A-Za-z0-9_]+', '', str(col)) for col in X_train_clean.columns]
    X_test_clean.columns = [re.sub('[^A-Za-z0-9_]+', '', str(col)) for col in X_test_clean.columns]

    search_xgb = RandomizedSearchCV(
        estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1),
        param_distributions={"n_estimators": [500, 1000], "learning_rate": [0.01, 0.05], "max_depth": [3, 4, 5],
                             "subsample": [0.7, 0.8], "colsample_bytree": [0.1, 0.2], "reg_alpha": [0.1, 0.5, 1.0],
                             "reg_lambda": [1.0, 5.0]},
        n_iter=15, cv=5, scoring="neg_root_mean_squared_error", random_state=42, n_jobs=-1, verbose=1
    )

    start_time = time.time()
    search_xgb.fit(X_train_clean, y_train)
    elapsed = time.time() - start_time

    best_model = search_xgb.best_estimator_
    y_pred = best_model.predict(X_test_clean)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Calculate Top Drivers
    feat_df = pd.DataFrame({'Bacteria': X_train_df.columns, 'Importance': best_model.feature_importances_})
    top_drivers = feat_df.sort_values('Importance', ascending=False).head(20).reset_index(drop=True)

    print(f"\n✅ {label} Complete ({elapsed:.1f}s) | R2: {r2:.3f}")

    return ModelResult(best_model, rmse, r2, search_xgb.best_params_, elapsed, top_drivers)


def random_forest_benchmark(X_train_df, X_test_df, y_train, y_test, label="Dataset"):
    print(f"🌲 Initializing Random Forest Engine: {label}")
    search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
        param_distributions={"n_estimators": [300, 500, 800, 1000], "max_depth": [None, 10, 20, 40],
                             "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4],
                             "max_features": ["sqrt", "log2"]},
        n_iter=20, cv=5, scoring="neg_mean_squared_error", random_state=42, n_jobs=-1, verbose=1
    )
    start_time = time.time()
    search.fit(X_train_df, y_train)
    elapsed = time.time() - start_time

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test_df)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Calculate Top Drivers
    feat_df = pd.DataFrame({'Bacteria': X_train_df.columns, 'Importance': best_model.feature_importances_})
    top_drivers = feat_df.sort_values('Importance', ascending=False).head(20).reset_index(drop=True)

    print(f"\n✅ {label} RF Complete ({elapsed:.1f}s) | R2: {r2:.3f}")

    return ModelResult(best_model, rmse, r2, search.best_params_, elapsed, top_drivers)


# =========================================================
# 4. REPEATED CV BATTLE (5x5 Arena)
# =========================================================
def final_battle(datasets_dict, y_train, n_splits=5, n_repeats=5, xgb_params=None, rf_params=None):
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    default_xgb = {
        'n_estimators': 1000, 'learning_rate': 0.01, 'max_depth': 3,
        'subsample': 0.7, 'colsample_bytree': 0.2, 'reg_alpha': 0.1,
        'reg_lambda': 1.0, 'n_jobs': -1, 'random_state': 42
    }

    default_rf = {
        'n_estimators': 1000, 'max_depth': None,
        'max_features': 'sqrt', 'n_jobs': -1, 'random_state': 42
    }

    current_xgb_params = xgb_params if xgb_params else default_xgb
    current_rf_params = rf_params if rf_params else default_rf

    models_to_test = {
        "XGBoost": xgb.XGBRegressor(**current_xgb_params),
        "Random Forest": RandomForestRegressor(**current_rf_params)
    }

    battle_results = []
    print(f"⚔️ Starting Battle Arena ({n_splits}x{n_repeats} = {n_splits * n_repeats} runs)")
    # Expanded headers to include R2
    print(f"{'Dataset':<20} | {'Model':<15} | {'Avg RMSE':<12} | {'Avg R2':<10} | {'Std Dev':<10}")
    print("-" * 85)

    for data_name, X_data in datasets_dict.items():
        X_clean = X_data.copy()
        X_clean.columns = [re.sub('[^A-Za-z0-9_]+', '', str(col)) for col in X_clean.columns]

        for model_name, model_obj in models_to_test.items():
            # Modified scoring to capture both RMSE and R2
            cv_metrics = cross_validate(
                model_obj, X_clean, y_train,
                cv=rkf,
                scoring={'rmse': 'neg_root_mean_squared_error', 'r2': 'r2'},
                n_jobs=-1
            )

            mean_rmse = -np.mean(cv_metrics['test_rmse'])
            std_rmse = np.std(cv_metrics['test_rmse'])
            mean_r2 = np.mean(cv_metrics['test_r2'])

            # Print updated table row
            print(f"{data_name:<20} | {model_name:<15} | {mean_rmse:.2f} days   | {mean_r2:.3f}    | ±{std_rmse:.2f}")

            battle_results.append({
                'Dataset': data_name,
                'Model': model_name,
                'RMSE_Mean': mean_rmse,
                'RMSE_Std': std_rmse,
                'R2_Mean': mean_r2
            })

    return battle_results
#!/usr/bin/env python3
# Hypothesis: 12h Camarilla H3/L3 breakout with 1d ADX trend filter and 12h volume confirmation.
# Long when price breaks above H3 with 1d ADX > 25 (trending) and 12h volume > 1.8x 20-period average.
# Short when price breaks below L3 with 1d ADX > 25 (trending) and 12h volume > 1.8x 20-period average.
# Exit on opposite Camarilla level (L3 for longs, H3 for shorts).
# Uses discrete position sizing (0.25) to minimize fee churn. ADX filter ensures trend alignment,
# reducing false breakouts in ranging markets. Target: 50-150 total trades over 4 years = 12-37/year for 12h.
# Works in bull/bear: 1d ADX confirms trend strength, Camarilla H3/L3 provides tight structure, volume confirms momentum.

name = "12h_Camarilla_H3L3_Breakout_1dADX_12hVolumeConfirm"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 12h Indicators (LTF) ---
    # 12h volume confirmation: > 1.8x 20-period average (tight filter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_12h = volume > (1.8 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ADX(14)
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                               np.abs(low_1d[1:] - close_1d[:-1])))
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di_14 = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr_14)
    minus_di_14 = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr_14)
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx_14 = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # --- 12h Camarilla Pivot Points (Prior Day OHLC) ---
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    df_1d_pivot = get_htf_data(prices, '1d')
    if len(df_1d_pivot) == 0:
        return np.zeros(n)
    
    # Precompute prior day's OHLC for each 12h bar
    open_time = prices['open_time']
    prior_day_start = open_time - pd.Timedelta(days=1)
    prior_day_start = prior_day_start.dt.normalize()  # Start of prior day
    
    for i in range(n):
        pd_ts = prior_day_start.iloc[i]
        day_mask = (df_1d_pivot['open_time'] >= pd_ts) & (df_1d_pivot['open_time'] < pd_ts + pd.Timedelta(days=1))
        if day_mask.any():
            day_data = df_1d_pivot.loc[day_mask]
            high_val = day_data['high'].iloc[0]
            low_val = day_data['low'].iloc[0]
            close_val = day_data['close'].iloc[0]
            range_val = high_val - low_val
            camarilla_h3[i] = close_val + (range_val * 1.1 / 4)  # H3
            camarilla_l3[i] = close_val - (range_val * 1.1 / 4)  # L3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(adx_14_aligned[i]) or
            np.isnan(volume_confirm_12h[i]) or
            np.isnan(camarilla_h3[i]) or
            np.isnan(camarilla_l3[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above H3 + 1d ADX > 25 (trending) + 12h volume confirmation
            if (close[i] > camarilla_h3[i] and 
                adx_14_aligned[i] > 25 and 
                volume_confirm_12h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below L3 + 1d ADX > 25 (trending) + 12h volume confirmation
            elif (close[i] < camarilla_l3[i] and 
                  adx_14_aligned[i] > 25 and 
                  volume_confirm_12h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below L3
            if close[i] < camarilla_l3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above H3
            if close[i] > camarilla_h3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
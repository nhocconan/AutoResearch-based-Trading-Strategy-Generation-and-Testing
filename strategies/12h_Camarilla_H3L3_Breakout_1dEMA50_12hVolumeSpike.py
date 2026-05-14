#!/usr/bin/env python3
# Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA50 trend filter and 12h volume spike confirmation.
# Long when price breaks above H3 with price > 1d EMA50 (bullish trend) and 12h volume > 2.0x 20-period average.
# Short when price breaks below L3 with price < 1d EMA50 (bearish trend) and 12h volume > 2.0x 20-period average.
# Exit on opposite Camarilla level (L3 for longs, H3 for shorts).
# Uses H3/L3 for tighter structure, 1d EMA50 for strong trend filter (reduces whipsaw), and moderate volume threshold.
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag.

name = "12h_Camarilla_H3L3_Breakout_1dEMA50_12hVolumeSpike"
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
    # 12h volume confirmation: > 2.0x 20-period average (balanced filter)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_12h = volume > (2.0 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) - trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
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
        if (np.isnan(ema_50_aligned[i]) or
            np.isnan(volume_spike_12h[i]) or
            np.isnan(camarilla_h3[i]) or
            np.isnan(camarilla_l3[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above H3 + price > 1d EMA50 (bullish) + 12h volume spike
            if (close[i] > camarilla_h3[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike_12h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below L3 + price < 1d EMA50 (bearish) + 12h volume spike
            elif (close[i] < camarilla_l3[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike_12h[i]):
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
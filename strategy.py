#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA(50) trend filter and 1d volume spike filter.
# Long when price breaks above R1 with 4h EMA(50) bullish (close > EMA) and 1d volume > 2.0x 20-period average.
# Short when price breaks below S1 with 4h EMA(50) bearish (close < EMA) and 1d volume > 2.0x 20-period average.
# Exit on opposite Camarilla level (S1 for longs, R1 for shorts).
# Uses discrete position sizing (0.20) to minimize fee churn and volume spike filter to reduce false breakouts.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe.
# Works in bull/bear: 4h EMA ensures trend alignment, Camarilla R1/S1 provides tight structure, volume spike confirms momentum.

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_1dVolumeSpike"
timeframe = "1h"
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
    
    # --- 1h Indicators (LTF) ---
    # Session filter: 08-20 UTC (pre-compute hours array)
    hours = prices.index.hour
    
    # --- 4h Indicators (HTF) ---
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # 4h EMA(50)
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # 1d volume spike: > 2.0x 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_20_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # --- 1h Camarilla Pivot Points (Prior Day OHLC) ---
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    df_1d_pivot = get_htf_data(prices, '1d')
    if len(df_1d_pivot) == 0:
        return np.zeros(n)
    
    # Precompute prior day's OHLC for each 1h bar
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
            camarilla_r1[i] = close_val + (range_val * 1.1 / 4)  # R1
            camarilla_s1[i] = close_val - (range_val * 1.1 / 4)  # S1
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data or outside session (08-20 UTC)
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or
            np.isnan(camarilla_r1[i]) or
            np.isnan(camarilla_s1[i]) or
            hours[i] < 8 or hours[i] > 20):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 + 4h EMA bullish (close > EMA) + 1d volume spike
            if (close[i] > camarilla_r1[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_spike_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 + 4h EMA bearish (close < EMA) + 1d volume spike
            elif (close[i] < camarilla_s1[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_spike_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1
            if close[i] < camarilla_s1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above R1
            if close[i] > camarilla_r1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
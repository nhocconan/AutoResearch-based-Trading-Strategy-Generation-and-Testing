#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot Breakout with 1d Volume Confirmation and ADX Trend Filter
# Takes long when price breaks above 12h Camarilla H4 level with 1d volume spike and ADX > 20
# Takes short when price breaks below 12h Camarilla L4 level with 1d volume spike and ADX > 20
# Exits when price returns to 12h Camarilla Pivot point (central level)
# Designed to capture institutional breakouts with volume confirmation, avoiding false breakouts in chop
# Target: 25-75 trades per symbol over 4 years (6-19/year)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h Camarilla levels (using previous period's high-low-close)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Shift to use previous period's data (avoid look-ahead)
    prev_high_12h = np.concatenate([[np.nan], high_12h[:-1]])
    prev_low_12h = np.concatenate([[np.nan], low_12h[:-1]])
    prev_close_12h = np.concatenate([[np.nan], close_12h[:-1]])
    
    # Calculate Camarilla levels for each period
    camarilla_h4 = np.full_like(close_12h, np.nan)
    camarilla_l4 = np.full_like(close_12h, np.nan)
    camarilla_pivot = np.full_like(close_12h, np.nan)
    
    for i in range(len(close_12h)):
        if not (np.isnan(prev_high_12h[i]) or np.isnan(prev_low_12h[i]) or np.isnan(prev_close_12h[i])):
            range_val = prev_high_12h[i] - prev_low_12h[i]
            camarilla_pivot[i] = (prev_high_12h[i] + prev_low_12h[i] + prev_close_12h[i]) / 3
            camarilla_h4[i] = camarilla_pivot[i] + 1.1 * range_val / 2
            camarilla_l4[i] = camarilla_pivot[i] - 1.1 * range_val / 2
    
    # Calculate 1d ADX for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_12h, camarilla_pivot)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 60  # for Camarilla and ADX calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_1d_current = vol_1d[i] if i < len(vol_1d) else vol_1d[-1]
        
        if position == 0:
            # Long setup: break above Camarilla H4 with volume spike and trend
            if (price > camarilla_h4_aligned[i] and 
                vol_1d_current > 1.5 * vol_ma_1d_aligned[i] and  # Volume spike
                adx_aligned[i] > 20):                           # Trend filter
                position = 1
                signals[i] = position_size
            # Short setup: break below Camarilla L4 with volume spike and trend
            elif (price < camarilla_l4_aligned[i] and 
                  vol_1d_current > 1.5 * vol_ma_1d_aligned[i] and  # Volume spike
                  adx_aligned[i] > 20):                           # Trend filter
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Pivot point
            if abs(price - camarilla_pivot_aligned[i]) < 0.001 * camarilla_pivot_aligned[i]:  # Within 0.1% of pivot
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Pivot point
            if abs(price - camarilla_pivot_aligned[i]) < 0.001 * camarilla_pivot_aligned[i]:  # Within 0.1% of pivot
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_Breakout_1dVolume_ADX"
timeframe = "12h"
leverage = 1.0
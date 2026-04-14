#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot Breakout with 1d Volume Spike and ADX Trend Filter
# Takes long when price breaks above Camarilla H3 with 1d volume spike and ADX > 25
# Takes short when price breaks below Camarilla L3 with 1d volume spike and ADX > 25
# Exits when price crosses back to Camarilla pivot (midpoint)
# Camarilla levels are effective in trending markets; volume confirms strength; ADX filters chop
# Target: 25-50 trades per symbol over 4 years (6-12.5/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels (H3, L3, and pivot)
    high_1d = df_4h['high'].values  # Use 4h high/low/close for pivot calculation as per Camarilla rules
    low_1d = df_4h['low'].values
    close_1d = df_4h['close'].values
    
    # Daily range (using 4h bars to calculate daily pivot - but we need actual daily data)
    # Let's use the 1d data for proper daily pivot
    high_1d_actual = df_1d['high'].values
    low_1d_actual = df_1d['low'].values
    close_1d_actual = df_1d['close'].values
    
    # Camarilla calculation: based on previous day's range
    # We'll calculate for each 1d bar, then align
    range_1d = high_1d_actual - low_1d_actual
    camarilla_pivot = (high_1d_actual + low_1d_actual + close_1d_actual) / 3
    camarilla_H3 = camarilla_pivot + (range_1d * 1.1 / 2)  # H3 = pivot + 1.1*(range)/2
    camarilla_L3 = camarilla_pivot - (range_1d * 1.1 / 2)  # L3 = pivot - 1.1*(range)/2
    
    # Calculate 1d ADX for trend strength
    high_1d_adx = df_1d['high'].values
    low_1d_adx = df_1d['low'].values
    close_1d_adx = df_1d['close'].values
    
    # True Range
    tr1 = high_1d_adx[1:] - low_1d_adx[1:]
    tr2 = np.abs(high_1d_adx[1:] - close_1d_adx[:-1])
    tr3 = np.abs(low_1d_adx[1:] - close_1d_adx[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d_adx[1:] - high_1d_adx[:-1]) > (low_1d_adx[:-1] - low_1d_adx[1:]), 
                       np.maximum(high_1d_adx[1:] - high_1d_adx[:-1], 0), 0)
    dm_minus = np.where((low_1d_adx[:-1] - low_1d_adx[1:]) > (high_1d_adx[1:] - high_1d_adx[:-1]), 
                        np.maximum(low_1d_adx[:-1] - low_1d_adx[1:], 0), 0)
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
    
    # Align indicators to 4h timeframe
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for ADX and volume calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_H3_aligned[i]) or np.isnan(camarilla_L3_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_1d_current = vol_1d[i] if i < len(vol_1d) else vol_1d[-1]
        
        if position == 0:
            # Long setup: break above Camarilla H3 with volume spike and strong trend
            if (price > camarilla_H3_aligned[i] and 
                vol_1d_current > 1.5 * vol_ma_1d_aligned[i] and  # Volume spike
                adx_aligned[i] > 25):                           # Strong trend
                position = 1
                signals[i] = position_size
            # Short setup: break below Camarilla L3 with volume spike and strong trend
            elif (price < camarilla_L3_aligned[i] and 
                  vol_1d_current > 1.5 * vol_ma_1d_aligned[i] and  # Volume spike
                  adx_aligned[i] > 25):                           # Strong trend
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to pivot level
            if price < camarilla_pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to pivot level
            if price > camarilla_pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Camarilla_Pivot_Breakout_1dVolume_ADX"
timeframe = "4h"
leverage = 1.0
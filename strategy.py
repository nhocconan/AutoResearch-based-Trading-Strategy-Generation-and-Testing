#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot long/short with 1d volume confirmation and 1d ADX trend filter.
Uses 4h Camarilla levels (based on prior day's range) for precise entry/exit, 1d volume and ADX to filter noise.
Aims for 20-40 trades/year (80-160 total over 4 years) with discrete sizing to minimize fee drag.
Works in bull via longs at L3/L4, in bear via shorts at H3/H4, avoids whipsaws with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivot, volume, and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels for each 1d bar
    range_val = prev_high - prev_low
    camarilla_h4 = prev_close + range_val * 1.1 / 2
    camarilla_h3 = prev_close + range_val * 1.1 / 4
    camarilla_l3 = prev_close - range_val * 1.1 / 4
    camarilla_l4 = prev_close - range_val * 1.1 / 2
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
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
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / (tr_14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d data to 4h
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(30, n):
        # Get aligned values
        h4 = camarilla_h4_aligned[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        l4 = camarilla_l4_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        adx_val = adx_aligned[i]
        
        # Check for NaN values
        if (np.isnan(h4) or np.isnan(h3) or np.isnan(l3) or np.isnan(l4) or 
            np.isnan(vol_ma) or np.isnan(adx_val)):
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma
        
        # ADX trend filter (> 20)
        trend_filter = adx_val > 20
        
        if position == 0:  # No position - look for entries
            if volume_confirm and trend_filter:
                # Long: price crosses above L3
                if close[i] > l3 and close[i-1] <= l3:
                    position = 1
                    signals[i] = position_size
                # Short: price crosses below H3
                elif close[i] < h3 and close[i-1] >= h3:
                    position = -1
                    signals[i] = -position_size
        elif position == 1:  # Long position - exit when price crosses below L4
            if close[i] < l4 and close[i-1] >= l4:
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit when price crosses above H4
            if close[i] > h4 and close[i-1] <= h4:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_1dVol_ADX"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Supertrend_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === Daily Supertrend Trend Filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ATR(10)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Basic upper and lower bands
    hl2 = (high_1d + low_1d) / 2
    upper_band = hl2 + (3.0 * atr)
    lower_band = hl2 - (3.0 * atr)
    
    # Final upper and lower bands
    final_upper = upper_band.copy()
    final_lower = lower_band.copy()
    for i in range(1, len(close_1d)):
        if close_1d[i-1] > final_upper[i-1]:
            final_upper[i] = max(upper_band[i], final_upper[i-1])
        else:
            final_upper[i] = upper_band[i]
        if close_1d[i-1] < final_lower[i-1]:
            final_lower[i] = min(lower_band[i], final_lower[i-1])
        else:
            final_lower[i] = lower_band[i]
    
    # Supertrend direction
    supertrend_dir = np.ones_like(close_1d)  # 1 for uptrend, -1 for downtrend
    for i in range(1, len(close_1d)):
        if close_1d[i] > final_upper[i-1]:
            supertrend_dir[i] = 1
        elif close_1d[i] < final_lower[i-1]:
            supertrend_dir[i] = -1
        else:
            supertrend_dir[i] = supertrend_dir[i-1]
            if supertrend_dir[i] == 1 and final_lower[i] < final_lower[i-1]:
                final_lower[i] = final_lower[i-1]
            if supertrend_dir[i] == -1 and final_upper[i] > final_upper[i-1]:
                final_upper[i] = final_upper[i-1]
    
    # Align Supertrend to 6h
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_1d, supertrend_dir)
    
    # === 6h Price and Volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume ratio with proper initialization
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = close[i]
        vol_ratio_val = vol_ratio[i]
        supertrend_val = supertrend_dir_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(supertrend_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Uptrend + volume confirmation
            if (supertrend_val == 1 and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Downtrend + volume confirmation
            elif (supertrend_val == -1 and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Downtrend or volume dries up
            if supertrend_val == -1 or vol_ratio_val < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Uptrend or volume dries up
            if supertrend_val == 1 or vol_ratio_val < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
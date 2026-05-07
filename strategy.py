#!/usr/bin/env python3
name = "4h_Keltner_Breakout_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR and EMA20 (midline)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA20 (midline)
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate 1d ATR(20)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d_arr, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d_arr, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_20_1d = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_20_1d)
    
    # Calculate Keltner Channels (upper/lower)
    upper_keltner = ema_20_1d_aligned + (2.0 * atr_20_1d_aligned)
    lower_keltner = ema_20_1d_aligned - (2.0 * atr_20_1d_aligned)
    
    # Get 1d volume for volume spike filter
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Calculate 4h volume ratio (current volume vs 20-period average)
    vol_ma20_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma20_4h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_20_1d_aligned[i]) or 
            np.isnan(upper_keltner[i]) or 
            np.isnan(lower_keltner[i]) or 
            np.isnan(vol_ma20_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Keltner + volume spike
            if (close[i] > upper_keltner[i] and 
                vol_ratio[i] > 1.8):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Keltner + volume spike
            elif (close[i] < lower_keltner[i] and 
                  vol_ratio[i] > 1.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below midline (EMA20)
            if close[i] < ema_20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above midline (EMA20)
            if close[i] > ema_20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
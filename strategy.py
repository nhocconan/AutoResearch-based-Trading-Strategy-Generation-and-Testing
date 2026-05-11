#!/usr/bin/env python3
name = "4h_Equidistance_Reversion_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data for equidistance channel and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily EMA(20) as middle line
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    # Daily ATR(10) for channel width
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = close_1d[0]
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - prev_close_1d)
    tr3 = np.abs(low_1d - prev_close_1d)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10_1d = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    # Equidistance channel: ±1.5 * ATR around EMA20
    upper_eq = ema_20_1d + 1.5 * atr_10_1d
    lower_eq = ema_20_1d - 1.5 * atr_10_1d
    
    # Align to 4h
    upper_eq_aligned = align_htf_to_ltf(prices, df_1d, upper_eq)
    lower_eq_aligned = align_htf_to_ltf(prices, df_1d, lower_eq)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume filter: 20-period average on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_eq_aligned[i]) or np.isnan(lower_eq_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold - avoid low-volume false signals
        volume_surge = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: Price breaks below lower equidistance channel with volume (mean reversion long)
            # Short: Price breaks above upper equidistance channel with volume (mean reversion short)
            if (close[i] < lower_eq_aligned[i] and volume_surge):
                signals[i] = 0.25
                position = 1
            elif (close[i] > upper_eq_aligned[i] and volume_surge):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to EMA20 (equilibrium)
            if position == 1:
                # Exit long: price returns to EMA20
                if close[i] >= ema_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to EMA20
                if close[i] <= ema_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
name = "12h_1D_Keltner_BandBreak_Trend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # 12h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Keltner Channel from 1D data
    # Keltner Channel: EMA ± (ATR * multiplier)
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # True Range for ATR
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first day
    atr_10_1d = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    multiplier = 1.5
    upper_keltner = ema_20_1d + (atr_10_1d * multiplier)
    lower_keltner = ema_20_1d - (atr_10_1d * multiplier)
    
    # Align to 12h timeframe
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    
    # Volume ratio (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_aligned[i]) or np.isnan(upper_keltner_aligned[i]) or 
            np.isnan(lower_keltner_aligned[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold
        volume_surge = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: Price breaks above upper Keltner band with volume surge
            if (close[i] > upper_keltner_aligned[i] and volume_surge):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Keltner band with volume surge
            elif (close[i] < lower_keltner_aligned[i] and volume_surge):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: Price returns below EMA (middle of Keltner channel)
                if close[i] < ema_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: Price returns above EMA (middle of Keltner channel)
                if close[i] > ema_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals